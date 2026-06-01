'use strict';
/* ══════════════════════════════════════════════
   UCL RAG — Optimised Raymarching
   Removed R/B chroma march loops (-96 steps/px)
   Steps 80→56, pixelRatio locked to 1.
   Visual: ~identical. Perf: ~3x faster.
══════════════════════════════════════════════ */

let scrollRatio = 0, smoothScroll = 0, scrollVel = 0, lastScroll = 0;
window.addEventListener('scroll', () => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  scrollRatio = max > 0 ? window.scrollY / max : 0;
  scrollVel = scrollRatio - lastScroll;
  lastScroll = scrollRatio;
}, { passive: true });

const panels = [
  { el: document.getElementById('p0'), start: -0.1, peak: 0.00, end: 0.22 },
  { el: document.getElementById('p1'), start: 0.18, peak: 0.28, end: 0.42 },
  { el: document.getElementById('p2'), start: 0.38, peak: 0.50, end: 0.62 },
  { el: document.getElementById('p3'), start: 0.58, peak: 0.70, end: 0.82 },
  { el: document.getElementById('p4'), start: 0.82, peak: 0.88, end: 1.00 },
];
const sectionNames = ['INTRO','QUERY','VECTOR','GRAPH','ASK'];
const progressBar  = document.getElementById('progress-bar');
const sectionLabel = document.getElementById('section-label');

function updatePanels(s) {
  panels.forEach((p, i) => {
    if (!p.el) return;
    let op = 0;
    if (s >= p.start && s <= p.end) {
      const fadeIn  = (s - p.start) / Math.max(p.peak - p.start, 0.001);
      const fadeOut = 1 - (s - p.peak) / Math.max(p.end - p.peak, 0.001);
      op = s <= p.peak ? Math.min(fadeIn, 1) : Math.max(Math.min(fadeOut, 1), 0);
    }
    if (i === 4 && s >= p.peak) op = 1;
    p.el.style.opacity = op.toFixed(3);
    p.el.style.pointerEvents = (i === 4 && op > 0.5) ? 'auto' : 'none';
  });
  const si = Math.min(Math.floor(s * 5), 4);
  if (sectionLabel) sectionLabel.textContent = String(si).padStart(2,'0') + ' / ' + sectionNames[si];
  if (progressBar)  progressBar.style.width = (s * 100).toFixed(1) + '%';
}
updatePanels(0);

(function () {
  const s = document.createElement('script');
  s.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
  s.onload = () => {
    const UNPKG = 'https://unpkg.com/three@0.128.0/examples/js/';
    function seq(srcs, done) {
      if (!srcs.length) { done(); return; }
      const sc = document.createElement('script');
      sc.src = srcs[0];
      sc.onload  = () => seq(srcs.slice(1), done);
      sc.onerror = () => seq(srcs.slice(1), done);
      document.head.appendChild(sc);
    }
    seq([
      UNPKG + 'postprocessing/EffectComposer.js',
      UNPKG + 'postprocessing/RenderPass.js',
      UNPKG + 'shaders/LuminosityHighPassShader.js',
      UNPKG + 'postprocessing/UnrealBloomPass.js',
      UNPKG + 'shaders/CopyShader.js',
      UNPKG + 'postprocessing/ShaderPass.js',
    ], init);
  };
  document.head.appendChild(s);

  function init() {
    const canvas = document.getElementById('c');
    if (!canvas || typeof THREE === 'undefined') return;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    renderer.setPixelRatio(1); // locked — biggest single perf win
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    const scene  = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    let composer = null;
    if (typeof THREE.EffectComposer !== 'undefined') {
      const rp    = new THREE.RenderPass(scene, camera);
      const bloom = new THREE.UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight), 1.2, 0.8, 0.1);
      composer = new THREE.EffectComposer(renderer);
      composer.addPass(rp);
      composer.addPass(bloom);
    }

    const vert = 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position,1.0); }';

    const frag = `
      precision highp float;
      uniform float uTime;
      uniform float uScroll;
      uniform float uAspect;
      varying vec2 vUv;

      mat2 rot2(float a){float c=cos(a),s=sin(a);return mat2(c,-s,s,c);}
      vec3 hsv(float h,float s,float v){
        vec4 K=vec4(1.,2./3.,1./3.,3.);
        vec3 p=abs(fract(vec3(h)+K.xyz)*6.-K.www);
        return v*mix(K.xxx,clamp(p-K.xxx,0.,1.),s);
      }
      float sdSphere(vec3 p,float r){return length(p)-r;}
      float sdBox(vec3 p,vec3 b){vec3 q=abs(p)-b;return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.);}
      float sdTorus(vec3 p,float r1,float r2){return length(vec2(length(p.xz)-r1,p.y))-r2;}
      float sdOct(vec3 p,float s){p=abs(p);return(p.x+p.y+p.z-s)*0.57735027;}
      float smin(float a,float b,float k){float h=max(k-abs(a-b),0.)/k;return min(a,b)-h*h*k*.25;}
      vec3 rep(vec3 p,vec3 c){return mod(p+c*.5,c)-c*.5;}

      float map(vec3 p){
        float s=uScroll,t=uTime;
        float warp=1.0+s*2.0;
        p.x+=sin(p.z*.18+t*.38)*warp;
        p.y+=cos(p.z*.14+t*.29)*warp*.8;
        p.xy=rot2(p.z*.025+s*3.14159)*p.xy;
        float cxy=mix(4.2,2.8,s),cz=mix(5.5,3.5,s);
        vec3 q=rep(p,vec3(cxy,cxy,cz));
        float sp=sdSphere(q,1.0-s*.35);
        float bx=sdBox(q,vec3(0.58+sin(t*.4)*.05,0.58,0.58));
        float tr=sdTorus(q.xzy,0.72,0.24+sin(t*.35)*.04);
        float oc=sdOct(q,0.9);
        float sp2=sdSphere(q,0.18);
        float b1=smoothstep(0.,.25,s),b2=smoothstep(.25,.5,s),b3=smoothstep(.5,.75,s),b4=smoothstep(.75,1.,s);
        float d=mix(sp,bx,b1); d=mix(d,tr,b2); d=mix(d,oc,b3); d=mix(d,sp2,b4);
        vec3 q2=rep(p+vec3(cxy*.5,cxy*.5,cz*.5),vec3(cxy,cxy,cz));
        return smin(d,sdSphere(q2,.12+sin(p.z*.3+t)*.04),.5);
      }
      vec3 calcNormal(vec3 p){
        float e=0.001;
        return normalize(vec3(
          map(p+vec3(e,0,0))-map(p-vec3(e,0,0)),
          map(p+vec3(0,e,0))-map(p-vec3(0,e,0)),
          map(p+vec3(0,0,e))-map(p-vec3(0,0,e))
        ));
      }

      void main(){
        float s=uScroll,t=uTime;
        vec2 uv=(vUv-.5)*vec2(uAspect,1.);
        vec3 ro=vec3(0.,0.,s*70.);
        ro.x+=sin(t*.22)*.35; ro.y+=cos(t*.17)*.25;
        float fov=1.35-s*.15;
        vec3 rd=normalize(vec3(uv,-fov));

        /* ── Single march pass (was 3×) ── */
        float tm=.1; bool hit=false;
        for(int i=0;i<56;i++){        /* 80→56 steps */
          float d=map(ro+rd*tm);
          if(d<0.0015*(1.+tm*.05)){hit=true;break;}
          if(tm>55.)break;
          tm+=d*.95;
        }

        vec3 col=vec3(0.);
        if(hit){
          vec3 p=ro+rd*tm,n=calcNormal(p);
          float hue=fract(dot(n*.5+.5,vec3(.333))+t*.04+s*.5+tm*.008);
          float sat=mix(0.95,.1,smoothstep(.7,1.,s));
          col=hsv(hue,sat,1.);
          vec3 ldir=normalize(vec3(sin(t*.4),cos(t*.3)*.5+.5,-.5));
          float diff=max(dot(n,ldir),0.);
          float spec=pow(max(dot(reflect(-ldir,n),-rd),0.),64.);
          col=col*(.2+diff*.8)+spec*.8;
          float fres=pow(1.-abs(dot(n,-rd)),3.5);
          col+=hsv(hue+.12,sat*.6,1.)*fres*1.4;
          col=mix(col,vec3(0.),1.-exp(-tm*.038));
        }

        col*=smoothstep(.88,.22,length(vUv-.5));
        col*=1.-sin(vUv.y*580.)*.5*.018;
        float g=fract(sin(dot(vUv*1000.+fract(t*.07),vec2(12.9898,78.233)))*43758.5453)*2.-1.;
        col+=g*.018;
        gl_FragColor=vec4(max(col,0.),1.);
      }
    `;

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uTime:   { value: 0 },
        uScroll: { value: 0 },
        uAspect: { value: window.innerWidth / window.innerHeight },
      },
      vertexShader: vert, fragmentShader: frag,
    });
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

    window.addEventListener('resize', () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      if (composer) composer.setSize(window.innerWidth, window.innerHeight);
      mat.uniforms.uAspect.value = window.innerWidth / window.innerHeight;
    });

    const clock = new THREE.Clock();
    (function animate() {
      requestAnimationFrame(animate);
      smoothScroll += (scrollRatio - smoothScroll) * 0.045;
      scrollVel *= 0.85;
      mat.uniforms.uTime.value   = clock.getElapsedTime();
      mat.uniforms.uScroll.value = smoothScroll;
      updatePanels(smoothScroll);
      if (composer) composer.render();
      else renderer.render(scene, camera);
    })();
  }
})();
