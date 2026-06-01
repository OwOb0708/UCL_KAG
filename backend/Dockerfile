FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gettext-base \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application source
COPY backend/ /app/

# Patch KAG SDK bug: trailing comma in _invoke wraps triples in a tuple, skipping all edges
RUN sed -i \
    's/triples = (self\.triples_extraction(passage, filtered_entities),)/triples = self.triples_extraction(passage, filtered_entities)/' \
    /usr/local/lib/python3.10/site-packages/kag/builder/component/extractor/schema_free_extractor.py

# Patch KAG SDK bug: kag_flow.py default_flow_component uses old unregistered type names;
# the registry now only accepts the _legacy variants.
RUN sed -i \
    's/"type": "kg_fr_open_spg"/"type": "kg_fr_open_spg_legacy"/g; \
     s/"type": "kg_cs_open_spg"/"type": "kg_cs_open_spg_legacy"/g; \
     s/"type": "rc_open_spg"/"type": "rc_open_spg_legacy"/g' \
    /usr/local/lib/python3.10/site-packages/kag/solver/executor/retriever/local_knowledge_base/kag_retriever/kag_flow.py

# Patch KAG SDK bug: ppr_chunk_retriever is a RetrieverABC but kag_flow.py passes it
# in the kg_fr config as ppr_chunk_retriever_tool (typed ToolABC), causing a registry mismatch.
# Removing it lets KgFreeRetrieverWithOpenSPG use its own fallback constructor.
RUN python3 -c \
    "import re; p='/usr/local/lib/python3.10/site-packages/kag/solver/executor/retriever/local_knowledge_base/kag_retriever/kag_flow.py'; \
    s=open(p).read(); \
    s=re.sub(r'[ ]+\"ppr_chunk_retriever_tool\":\s*\{[^}]+\},\n', '', s, flags=re.DOTALL); \
    open(p,'w').write(s); print('ppr_chunk_retriever_tool patch OK')"

# Patch KAG SDK bug: default_kg_fr_retriever.py calls PprChunkRetriever with old API
# (invoke(queries=..., start_entities=...) -> tuple). New API is invoke(task) -> RetrieverOutput.
RUN python3 -c \
    "path='/usr/local/lib/python3.10/site-packages/kag/solver/executor/retriever/local_knowledge_base/kag_retriever/kag_component/kg_fr/default_kg_fr_retriever.py'; \
    old='        chunks, match_spo = self.ppr_chunk_retriever_tool.invoke(\n            queries=ppr_queries,\n            start_entities=entities,\n            top_k=self.top_k,\n        )'; \
    new=\"        _ppr_out = self.ppr_chunk_retriever_tool.invoke(executor_task, top_k=self.top_k)\n        chunks = getattr(_ppr_out, 'chunks', []) or []\n        match_spo = []\"; \
    s=open(path).read(); open(path,'w').write(s.replace(old,new)); print('kg_fr invoke patch OK')"

# Patch KAG SDK: main_solver.py error message from Simplified to Traditional Chinese
RUN sed -i \
    's/抱歉，处理查询 {query} 时发生异常。错误：{str(e)}, 请重试。/抱歉，處理查詢 {query} 時發生異常。錯誤：{str(e)}，請重試。/' \
    /usr/local/lib/python3.10/site-packages/kag/solver/main_solver.py

# kag_config.yaml is mounted as a template at /app/kag_config.yaml.template
# entrypoint.sh substitutes env vars and writes the final /app/kag_config.yaml

RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV KAG_CONFIG_PATH=/app/kag_config.yaml

ENTRYPOINT ["/app/entrypoint.sh"]
