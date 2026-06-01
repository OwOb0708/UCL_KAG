const statusBox = document.getElementById("statusBox");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");

function setStatus(message) {
  statusBox.textContent = message;
}

// 輪詢索引狀態，索引完成前每 5 秒查一次
async function pollIndexingStatus() {
  try {
    const res = await fetch("api/status");
    const data = await res.json();
    if (data.indexing_status === "indexing") {
      setStatus(
        "⏳ 背景索引中（資料量大時需數分鐘），系統仍可正常查詢已完成的資料...",
      );
      setTimeout(pollIndexingStatus, 5000);
    } else if (data.indexing_status === "ready") {
      setStatus("系統就緒。");
    } else {
      setStatus(`索引狀態：${data.indexing_status}`);
    }
  } catch {
    setTimeout(pollIndexingStatus, 8000);
  }
}
pollIndexingStatus();

function scoreBar(score) {
  const pct = Math.round(score * 100);
  const color = score >= 0.7 ? "#006d77" : score >= 0.4 ? "#e29578" : "#aaa";
  return `<span class="score-bar-wrap">
    <span class="score-bar-fill" style="width:${pct}%;background:${color}"></span>
  </span><span class="score-num">${pct}%</span>`;
}

function shortSource(source) {
  // "gdrive:fileId:filename.pdf" → "filename.pdf"
  const parts = source.split(":");
  return parts[parts.length - 1] || source;
}

function appendUserMessage(text) {
  const article = document.createElement("article");
  article.className = "msg user";
  article.innerHTML = `<div class="role">你</div><div class="msg-text">${escHtml(text)}</div>`;
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendBotMessage(result) {
  const article = document.createElement("article");
  article.className = "msg bot";

  // ── Answer ──
  const answerHtml = `<div class="role">助理</div>
    <div class="msg-text">${formatAnswer(result.answer, result.citations || [])}</div>`;

  // ── Vector / Hybrid Search Results ──
  let vectorHtml = "";
  if (result.citations?.length) {
    const rows = result.citations
      .map(
        (c, i) => `
      <tr>
        <td class="src-idx">${i + 1}</td>
        <td class="src-name" title="${escHtml(c.source)}">${escHtml(shortSource(c.source))}</td>
        <td class="src-score">${scoreBar(c.score)}</td>
        <td class="src-excerpt">${escHtml(c.excerpt.slice(0, 120))}…</td>
      </tr>`,
      )
      .join("");

    vectorHtml = `
      <details class="evidence-block" open>
        <summary>
          <span class="ev-label vector-label">向量 + 關鍵字搜尋</span>
          <span class="ev-count">${result.citations.length} 筆</span>
        </summary>
        <table class="ev-table">
          <thead><tr><th>#</th><th>來源</th><th>相關度</th><th>摘要</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </details>`;
  }

  // ── KAG Knowledge Graph Results (SPO triples) ──
  let graphHtml = "";
  if (result.graph_facts?.length) {
    const rows = result.graph_facts
      .map((f, i) => {
        const src = f.source ? `<span class="doc-date" title="${escHtml(f.source)}">${escHtml(shortSource(f.source))}</span>` : `<span class="no-date">—</span>`;
        return `<tr>
        <td class="src-idx">${i + 1}</td>
        <td class="src-name">${escHtml(f.subject)}</td>
        <td class="src-score">${escHtml(f.predicate)}</td>
        <td class="src-excerpt">${escHtml(f.object)}</td>
        <td class="src-name">${src}</td>
      </tr>`;
      })
      .join("");

    graphHtml = `
      <details class="evidence-block">
        <summary>
          <span class="ev-label graph-label">知識圖譜 (KAG · Neo4j)</span>
          <span class="ev-count">${result.graph_facts.length} 筆</span>
        </summary>
        <table class="ev-table">
          <thead><tr><th>#</th><th>主體</th><th>關係</th><th>客體</th><th>來源</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </details>`;
  }

  article.innerHTML = answerHtml + vectorHtml + graphHtml;
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function formatAnswer(text, citations = []) {
  // Split into blocks: markdown tables vs normal text
  const lines = text.split("\n");
  const outputParts = [];
  let i = 0;

  while (i < lines.length) {
    // Detect markdown table: current line and next line both look like table rows
    if (
      lines[i].trim().startsWith("|") &&
      i + 1 < lines.length &&
      /^\|[-| :]+\|/.test(lines[i + 1].trim())
    ) {
      // Collect all table lines
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      outputParts.push(renderMarkdownTable(tableLines, citations));
    } else {
      // Normal line
      const escaped = escHtml(lines[i])
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(
          /(https?:\/\/[^\s<]+)/g,
          '<a href="$1" target="_blank" rel="noopener">$1</a>',
        );
      outputParts.push(renderCitationBadges(escaped, citations));
      i++;
    }
  }

  // Join: tables are block elements, text lines use <br>
  let html = "";
  for (let j = 0; j < outputParts.length; j++) {
    if (outputParts[j].startsWith("<table")) {
      if (html && !html.endsWith("<br>")) html += "<br>";
      html += outputParts[j];
      if (j + 1 < outputParts.length) html += "<br>";
    } else {
      html += outputParts[j];
      if (j + 1 < outputParts.length) html += "<br>";
    }
  }
  return html;
}

// 將 【N†...】 標註替換成可 hover 的來源 badge
function renderCitationBadges(html, citations) {
  return html.replace(/【(\d+)[†]([^】]*)】/g, (match, numStr, label) => {
    const idx = parseInt(numStr, 10) - 1; // LLM 從 1 開始計數
    const citation = citations[idx];
    if (!citation)
      return `<sup class="cite-badge cite-unknown" title="來源 ${numStr}">[${numStr}]</sup>`;
    const filename = shortSource(citation.source);
    const excerpt = escHtml(citation.excerpt.slice(0, 200));
    return `<sup class="cite-badge" title="${escHtml(filename)}&#10;${excerpt}" data-idx="${idx + 1}">[${numStr}]</sup>`;
  });
}

function renderMarkdownTable(lines, citations = []) {
  // lines[0] = header row, lines[1] = separator, lines[2+] = data rows
  const parseRow = (line) =>
    line
      .trim()
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((c) => c.trim());

  const headers = parseRow(lines[0]);
  const dataRows = lines.slice(2).map(parseRow);

  const renderCell = (c) =>
    renderCitationBadges(
      escHtml(c)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/&lt;br\s*\/?&gt;/gi, "<br>")
        .replace(/\n/g, "<br>"),
      citations,
    );

  const thCells = headers.map((h) => `<th>${renderCell(h)}</th>`).join("");
  const trRows = dataRows
    .map((cols) => {
      const tds = cols.map((c) => `<td>${renderCell(c)}</td>`).join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");

  return `<table class="md-table"><thead><tr>${thCells}</tr></thead><tbody>${trRows}</tbody></table>`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "請求失敗");
  return data;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  appendUserMessage(message);
  messageInput.value = "";
  messageInput.style.height = "auto";

  const submitBtn = chatForm.querySelector("button[type=submit]");
  submitBtn.disabled = true;
  setStatus("回答生成中...");

  try {
    const result = await postJson("api/chat", { message, top_k: 6 });
    appendBotMessage(result);
    setStatus("完成。");
  } catch (error) {
    const errArticle = document.createElement("article");
    errArticle.className = "msg bot error";
    errArticle.innerHTML = `<div class="role">助理</div><div class="msg-text">查詢失敗：${escHtml(error.message)}</div>`;
    chatLog.appendChild(errArticle);
    setStatus("發生錯誤。");
  } finally {
    submitBtn.disabled = false;
  }
});

// Auto-resize textarea
messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = messageInput.scrollHeight + "px";
});

// Ctrl+Enter to submit
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});
