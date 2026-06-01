from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path


class KAGService:
    """Async wrapper around the KAG builder and solver pipelines."""

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
        self._llm_kwargs = dict(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=chat_model,
            temperature=0.1,
            timeout=300,
        )
        # Lightweight model for planning/reasoning/deduction steps (not final answer generation)
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
        if not self._initialized:
            await self.initialize()
        try:
            from kag.solver.main_solver import SolverMain
            solver = SolverMain()
            question = f"請務必使用繁體中文回答以下問題：{question}"

            task_key = f"0_{self._project_id}"
            # Big model: reading comprehension + final answer generation (quality-critical)
            llm_cfg = {"type": "openai", **self._llm_kwargs}
            # Small model: planning, logical form rewriting, deduction, output formatting
            planner_cfg = {"type": "openai", **self._planner_llm_kwargs}
            embed_cfg = {"type": "openai", **self._embed_kwargs}

            _graph_executor = {
                "type": "kag_hybrid_executor",
                "flow": "kg_fr->rc",
                "lf_rewriter": {
                    "type": "kag_spo_lf",
                    "llm_client": planner_cfg,  # logical form translation → small model
                    "lf_trans_prompt": {"type": "default_logic_form_plan"},
                    "kag_qa_task_config_key": task_key,
                },
                "llm_client": llm_cfg,  # reading comprehension over retrieved chunks → big model
                "kag_qa_task_config_key": task_key,
            }

            custom_pipeline = {
                "type": "kag_static_pipeline",
                "planner": {
                    "type": "lf_kag_static_planner",
                    "llm": planner_cfg,  # task decomposition → small model
                    "plan_prompt": {"type": "default_lf_static_planning"},
                    "rewrite_prompt": {"type": "default_rewrite_sub_task_query"},
                },
                "executors": [
                    _graph_executor,
                    {"type": "py_code_based_math_executor", "llm": planner_cfg},  # math/code → small model
                    {"type": "kag_deduce_executor", "llm_module": planner_cfg},   # deduction → small model
                    {"type": "kag_output_executor", "llm_module": planner_cfg},   # output format → small model
                ],
                "generator": {
                    "type": "llm_index_generator",
                    "llm_client": llm_cfg,  # final answer in Traditional Chinese → big model
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
                        "index_list": [],
                    }],
                },
            }
            answer = await solver.ainvoke(
                project_id=self._project_id,
                task_id=0,
                query=question,
                host_addr=self._host,
                params=params,
            )
            clean = re.sub(r"<reference[^>]*></reference>", "", str(answer)).strip() if answer else ""
            return clean, [], []
        except Exception as exc:
            print(f"[KAG] solve error: {exc}")
            return "", [], [str(exc)]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _setup_project(self) -> None:
        import yaml
        try:
            from knext.project.client import ProjectClient
            client = ProjectClient(host_addr=self._host)

            existing = client.get_by_namespace(namespace=self._namespace)
            if not existing:
                config_path = os.environ.get("KAG_CONFIG_PATH", "/app/kag_config.yaml")
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                vm = cfg.get("vectorize_model", {})
                cfg["vectorizer"] = {
                    "type": vm.get("client_type", "openai"),
                    "base_url": vm.get("base_url", ""),
                    "api_key": vm.get("api_key", ""),
                    "model": vm.get("model", ""),
                    "vector_dimensions": vm.get("vector_dimensions", 1536),
                }
                project = client.create(
                    name="UCL Lab KAG",
                    namespace=self._namespace,
                    config=cfg,
                    userNo="openspg",
                )
                print(f"[KAG] Project created with id={project.id}")
                self._project_id = project.id
                os.environ["KAG_PROJECT_ID"] = str(project.id)
            else:
                print(f"[KAG] Project already exists with id={existing.id}")
                self._project_id = existing.id
                os.environ["KAG_PROJECT_ID"] = str(existing.id)
        except Exception as exc:
            print(f"[KAG] project setup warning: {exc}")

        # Commit schema via knext CLI (reads KAG_PROJECT_ID and KAG_PROJECT_HOST_ADDR from env)
        schema_dir = str(Path(self._schema_path).parent)
        try:
            result = subprocess.run(
                ["knext", "schema", "commit"],
                capture_output=True, text=True, timeout=60,
                cwd=schema_dir,
                env={**os.environ, "KAG_PROJECT_ID": str(self._project_id),
                     "KAG_PROJECT_HOST_ADDR": self._host},
            )
            if result.returncode != 0:
                print(f"[KAG] schema commit: {result.stderr.strip() or result.stdout.strip()}")
        except FileNotFoundError:
            print("[KAG] knext CLI not found — skipping schema commit")
        except Exception as exc:
            print(f"[KAG] schema commit warning: {exc}")
        print("[KAG] Project/schema setup done.")
