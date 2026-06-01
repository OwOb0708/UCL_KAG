from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path


class KAGService:
    """Async wrapper around the KAG builder and solver pipelines.

    Builder pipeline (schema-constrained):
        TXTReader → LengthSplitter → SchemaConstraintExtractor
        (UclLabNERPrompt + UclLabRelationPrompt) → BatchVectorizer → KGWriter

    Solver pipeline (KAG graph reasoning):
        think_pipeline → kag_static_pipeline
          ├── lf_kag_static_planner  (task decomposition)
          ├── kag_hybrid_executor    (KG logical-form query + reading comprehension)
          │     flow: "kg_fr -> rc"
          │     kg_fr: entity-linking + SPO logical-form → Neo4j graph traversal
          │     rc  : vector-chunk context + LLM reading comprehension
          ├── py_code_based_math_executor
          ├── kag_deduce_executor
          ├── kag_output_executor
          └── llm_index_generator    (final answer in Traditional Chinese)

    Critical: index_list must include "chunk_index" so the rc step has
    vector chunks to ground its answers.  An empty index_list silently
    disables ALL retrieval and is the primary cause of poor performance.
    """

    def __init__(
        self,
        openspg_host: str,
        project_id: int,
        namespace: str,
        schema_path: str,
        openai_base_url: str,
        openai_api_key: str,
        chat_model: str,
        planner_model: str,
        embedding_model: str,
        embedding_dimensions: int,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
    ) -> None:
        self._host = openspg_host
        self._project_id = project_id
        self._namespace = namespace
        self._schema_path = schema_path

        # Neo4j – stored so _setup_project and solve() can reference them
        self._neo4j_uri = neo4j_uri or "bolt://neo4j:7687"
        self._neo4j_user = neo4j_user or "neo4j"
        self._neo4j_password = neo4j_password or "neo4j@openspg"

        # Big model: reading comprehension + final answer (quality-critical)
        self._llm_kwargs = dict(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=chat_model,
            temperature=0.1,
            timeout=300,
        )
        # Small model: planning / logical-form rewriting / deduction / output formatting
        self._planner_llm_kwargs = dict(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=planner_model,
            temperature=0.1,
            timeout=120,
        )
        self._embed_kwargs = dict(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=embedding_model,
            vector_dimensions=embedding_dimensions,
        )
        self._initialized = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._initialized:
            return

        os.environ["KAG_PROJECT_ID"] = str(self._project_id)
        os.environ["KAG_PROJECT_HOST_ADDR"] = self._host

        config_path = os.environ.get("KAG_CONFIG_PATH")
        from kag.common.conf import init_env
        init_env(config_file=config_path)

        await asyncio.get_event_loop().run_in_executor(None, self._setup_project)
        self._initialized = True

    # ── builder ──────────────────────────────────────────────────────────────

    async def build_document(self, text: str, source: str) -> bool:
        if not self._initialized:
            await self.initialize()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_build, text, source)

    def _sync_build(self, text: str, source: str) -> bool:
        """Schema-constrained KG builder.

        Pipeline:
            TXTReader → LengthSplitter
            → SchemaConstraintExtractor (UclLabNERPrompt + UclLabRelationPrompt)
            → BatchVectorizer → KGWriter

        SchemaConstraintExtractor enforces that:
        - NER output is filtered to the 12 UCLLab entity types
        - Relation predicates are limited to the 19 defined in RELATION_SCHEMA
        This prevents LLM-invented types from polluting the graph.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(f"# Source: {source}\n\n{text}")
            tmp = f.name

        try:
            from kag.interface.common.llm_client import LLMClient
            from kag.interface.common.vectorize_model import VectorizeModelABC
            from kag.builder.default_chain import DefaultUnstructuredBuilderChain
            from kag.builder.component.reader.txt_reader import TXTReader
            from kag.builder.component.splitter.length_splitter import LengthSplitter
            from kag.builder.component.extractor.schema_constraint_extractor import SchemaConstraintExtractor
            from kag.builder.component.vectorizer.batch_vectorizer import BatchVectorizer
            from kag.builder.component.writer.kg_writer import KGWriter
            from app.prompts.ucl_lab_ner import UclLabNERPrompt
            from app.prompts.ucl_lab_relation import UclLabRelationPrompt

            llm = LLMClient.from_config({"type": "openai", **self._llm_kwargs})
            embed = VectorizeModelABC.from_config({"type": "openai", **self._embed_kwargs})

            chain = DefaultUnstructuredBuilderChain(
                reader=TXTReader(),
                splitter=LengthSplitter(split_length=1200, window_length=180),
                extractor=SchemaConstraintExtractor(
                    llm=llm,
                    ner_prompt=UclLabNERPrompt(language="zh"),
                    relation_prompt=UclLabRelationPrompt(language="zh"),
                ),
                vectorizer=BatchVectorizer(vectorize_model=embed),
                writer=KGWriter(project_id=self._project_id),
            )
            chain.invoke(tmp)
            return True
        except Exception as exc:
            print(f"[KAG] build error for {source}: {exc}")
            return False
        finally:
            Path(tmp).unlink(missing_ok=True)

    # ── solver ───────────────────────────────────────────────────────────────

    async def solve(self, question: str) -> tuple[str, list[dict], list[str]]:
        """KAG graph-reasoning solver.

        Uses think_pipeline (kag_static_pipeline) which performs:
          1. Task decomposition (planner)
          2. KG graph traversal via SPO logical forms (kg_fr step of kag_hybrid_executor)
          3. Vector-chunk retrieval for reading comprehension (rc step)
          4. Math / deduction / output formatting executors
          5. Final answer generation in Traditional Chinese

        Key fix vs naive RAG:
          - index_list: ["chunk_index"] enables vector chunk retrieval for rc step.
            Previously [] caused ALL retrieval to silently be skipped, making the
            think_pipeline behave worse than a basic LLM call.
          - kag_hybrid_executor flow "kg_fr->rc" traverses the schema-typed Neo4j
            graph (UCLLab.Person, UCLLab.Project, etc.) before reading comprehension,
            enabling true multi-hop KG reasoning.
        """
        if not self._initialized:
            await self.initialize()
        try:
            from kag.solver.main_solver import SolverMain
            solver = SolverMain()
            question = f"請務必使用繁體中文回答以下問題：{question}"

            task_key = f"0_{self._project_id}"
            llm_cfg = {"type": "openai", **self._llm_kwargs}
            planner_cfg = {"type": "openai", **self._planner_llm_kwargs}
            embed_cfg = {"type": "openai", **self._embed_kwargs}

            # ── KG hybrid executor ─────────────────────────────────────────
            # flow "kg_fr->rc":
            #   kg_fr: converts query → SPO logical form → Neo4j graph traversal
            #          finds schema-typed entities (Person, Project, Task …) and
            #          their relations, returns relevant subgraph triples
            #   rc   : combines KG triples + vector chunks → LLM reading comprehension
            #
            # lf_rewriter (kag_spo_lf): translates natural language into
            #   Subject-Predicate-Object patterns that can be executed against
            #   the typed Neo4j graph.  Uses the small/planner model to keep
            #   latency low.
            _graph_executor = {
                "type": "kag_hybrid_executor",
                "flow": "kg_fr->rc",
                "lf_rewriter": {
                    "type": "kag_spo_lf",
                    "llm_client": planner_cfg,
                    "lf_trans_prompt": {"type": "default_logic_form_plan"},
                    "kag_qa_task_config_key": task_key,
                },
                "llm_client": llm_cfg,
                "kag_qa_task_config_key": task_key,
            }

            # ── think_pipeline (kag_static_pipeline) ──────────────────────
            custom_pipeline = {
                "type": "kag_static_pipeline",
                "planner": {
                    "type": "lf_kag_static_planner",
                    "llm": planner_cfg,
                    "plan_prompt": {"type": "default_lf_static_planning"},
                    "rewrite_prompt": {"type": "default_rewrite_sub_task_query"},
                },
                "executors": [
                    _graph_executor,
                    {"type": "py_code_based_math_executor", "llm": planner_cfg},
                    {"type": "kag_deduce_executor",  "llm_module": planner_cfg},
                    {"type": "kag_output_executor",  "llm_module": planner_cfg},
                ],
                "generator": {
                    "type": "llm_index_generator",
                    "llm_client": llm_cfg,
                    "generated_prompt": {"type": "default_refer_generator_prompt"},
                    "enable_ref": True,
                },
            }

            params = {
                "usePipeline": "think_pipeline",
                "config": {
                    "llm": llm_cfg,
                    "chat_llm": llm_cfg,
                    "vectorize_model": embed_cfg,
                    "think_pipeline": custom_pipeline,
                    "kb": [{
                        "id": self._project_id,
                        "project": {
                            "id": self._project_id,
                            "host_addr": self._host,
                            "namespace": self._namespace,
                            "language": "zh",
                        },
                   