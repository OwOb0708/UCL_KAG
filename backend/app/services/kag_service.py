from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path


class KAGService:
    """Async wrapper around the KAG builder and solver pipelines.

    Builder  (schema-constrained KG construction):
        TXTReader → LengthSplitter → SchemaConstraintExtractor
        (UclLabNERPrompt + UclLabRelationPrompt) → BatchVectorizer → KGWriter

    Solver  (Cypher-first, KAG fallback):
        1. CypherSolver.solve()
               LLM → Cypher query → Neo4j direct execution → LLM formats answer
               ≈ 2 LLM calls, answers schema-typed 1/2/3-hop graph questions
               in seconds instead of 30 minutes.
        2. Fallback: KAG think_pipeline (vector chunk retrieval)
               Activated when Cypher returns empty results or execution fails.
               Handles open-ended / unstructured questions not suited for Cypher.
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

        self._neo4j_uri = neo4j_uri or "bolt://neo4j:7687"
        self._neo4j_user = neo4j_user or "neo4j"
        self._neo4j_password = neo4j_password or "neo4j@openspg"

        # Big model: final answer generation (quality-critical)
        self._llm_kwargs = dict(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=chat_model,
            temperature=0.1,
            timeout=300,
        )
        # Planner model: Cypher generation & reasoning steps (speed-critical)
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

        NER is restricted to 12 UCLLab entity types and 19 relation predicates
        defined in ucl_lab_ner.py and ucl_lab_relation.py.  The
        SchemaConstraintExtractor filters any LLM-invented types at parse time,
        so the graph stays clean across repeated ingestions.
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
        """Cypher-first solver with KAG fallback.

        Step 1 — CypherSolver (fast path):
            LLM generates a schema-aware Cypher query (1 planner-model call),
            executes it directly on Neo4j, LLM formats the answer (1 chat-model
            call).  This handles 1/2/3-hop schema-typed questions in ~5-30s.

        Step 2 — KAG think_pipeline (fallback):
            Activated when Cypher returns empty results or the query fails
            (e.g. open-ended / unstructured questions).  Uses chunk_index for
            vector retrieval + kag_hybrid_executor for KG reasoning.
        """
        if not self._initialized:
            await self.initialize()

        # ── Step 1: Cypher-first ─────────────────────────────────────────────
        try:
            from app.services.cypher_solver import CypherSolver
            cypher_solver = CypherSolver(
                neo4j_uri=self._neo4j_uri,
                neo4j_user=self._neo4j_user,
                neo4j_password=self._neo4j_password,
                llm_kwargs=self._llm_kwargs,
                planner_kwargs=self._planner_llm_kwargs,
            )
            answer, facts = await cypher_solver.solve(question)
            if answer:
                print(f"[KAG] Cypher path answered: {len(answer)} chars")
                return answer, facts, []
            print("[KAG] Cypher returned empty, falling back to KAG pipeline")
        except Exception as exc:
            print(f"[KAG] Cypher solver error: {exc}, falling back")

        # ── Step 2: KAG think_pipeline fallback ─────────────────────────────
        return await self._kag_pipeline_solve(question)

    async def _kag_pipeline_solve(self, question: str) -> tuple[str, list[dict], list[str]]:
        """KAG think_pipeline — used as fallback for non-graph questions.

        index_list: ["chunk_index"] enables vector-chunk retrievers for the
        rc (reading comprehension) step inside kag_hybrid_executor.
        Without this, rc has no grounding context (the original bug that
        caused 30-minute waits with empty / wrong answers).
        """
        try:
            from kag.solver.main_solver import SolverMain
            solver = SolverMain()
            q = f"請務必使用繁體中文回答以下問題：{question}"

            task_key = f"0_{self._project_id}"
            llm_cfg = {"type": "openai", **self._llm_kwargs}
            planner_cfg = {"type": "openai", **self._planner_llm_kwargs}
            embed_cfg = {"type": "openai", **self._embed_kwargs}

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
                        "vectorizer": embed_cfg,
                        "index_list": ["chunk_index"],
                    }],
                },
            }
            answer = await solver.ainvoke(
                project_id=self._project_id,
                tas