from __future__ import annotations

"""Text-to-Cypher solver for UCLLab knowledge graph.

Design goal: answer schema-typed queries (including 2-hop graph traversals)
in seconds instead of minutes, by generating a Cypher query directly from
the user's question and executing it against Neo4j.

Flow:
    question
      → LLM generates Cypher  (1 call, small prompt, planner model)
      → neo4j.AsyncDriver executes query against `ucllab` database
      → if results: LLM formats natural-language answer (1 call, chat model)
      → if no results / Cypher error: fallback to KAG vector search

Two LLM calls total vs 6-10+ for think_pipeline  →  ~95% latency reduction
for questions that can be answered by the graph.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── UCLLab schema reference injected into every Cypher-generation prompt ────

_SCHEMA_PROMPT = """\
你操作的是 UCL 實驗室知識圖譜，Neo4j 資料庫名稱為 `ucllab`。

【節點類型 & 常用屬性】
- Person          : name, role
- Project         : name, category, status
- Task            : content, status, deadline
- MeetingRecord   : title, date, semester
- Decision        : content, status, deadline, decidedAt
- ResearchPlan    : name, planCode, category, status, startDate, endDate
- Patent          : name, status, region
- ComputeResource : name, resourceType, spec, status
- Equipment       : name, equipmentType, status
- Course          : name, courseCode, semester
- ExternalParty   : name, partyType
- Competition     : name, stage, submissionDeadline, prize
- Chunk           : content  （原始文字片段，盡量不要直接查 Chunk）

【關係 (方向: 主體 → 客體)】
- (Person)        -[:lead]->        (Project | Competition | ResearchPlan)
- (Person)        -[:members]->     (Project)
- (Person)        -[:consultant]->  (Project)
- (Person)        -[:assignedTo]->  (Task)
- (Task)          -[:belongsTo]->   (Project)
- (Person)        -[:admin]->       (ComputeResource)
- (Project)       -[:usedBy]->      (ComputeResource)
- (Person)        -[:manager]->     (Equipment)
- (Person)        -[:ta]->          (Course)
- (Person)        -[:instructor]->  (Course)
- (ResearchPlan)  -[:relatedProject]-> (Project)
- (Decision)      -[:executor]->    (Person)
- (Decision)      -[:affects]->     (Project)
- (MeetingRecord) -[:updatesProject]-> (Project)
- (Person)        -[:teamMembers]-> (Competition)

【重要規則】
1. 所有節點標籤前不需加 namespace，直接用 `Person`、`Project` 等。
2. 查詢時用 `=~` 做不區分大小寫的模糊比對，例如：
   WHERE n.name =~ '(?i).*宇軒.*'
3. 只輸出 Cypher 查詢語句，不要任何解釋、markdown 或程式碼區塊符號。
4. LIMIT 20 避免回傳過多結果。
5. 雙跳查詢範例：
   MATCH (p:Person)-[:assignedTo]->(t:Task)-[:belongsTo]->(proj:Project)
   WHERE p.name =~ '(?i).*宇軒.*'
   RETURN p.name AS person, t.content AS task, proj.name AS project
   LIMIT 20
"""

_CYPHER_GEN_SYSTEM = (
    "你是 Neo4j Cypher 查詢生成器。根據使用者問題和提供的 schema，"
    "生成一個能直接執行的 Cypher 查詢。只輸出 Cypher，不要任何其他文字。"
)

_FORMAT_SYSTEM = (
    "你是 UCL 實驗室助理，根據知識圖譜查詢結果，用繁體中文給出清晰完整的回答。"
    "如果結果為空，直接說明找不到相關資料。回答請簡潔有重點。"
)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some models add."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


class CypherSolver:
    """Schema-aware Text-to-Cypher solver for UCLLab knowledge graph.

    Steps:
        1. generate_cypher()  — 1 LLM call (planner model, fast)
        2. run_cypher()       — direct Neo4j query (milliseconds)
        3. format_answer()    — 1 LLM call (chat model)

    Total: ~2 LLM calls for any 2-hop graph question.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm_kwargs: dict,
        planner_kwargs: dict,
    ) -> None:
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password
        self._llm_kwargs = llm_kwargs        # big model: final answer
        self._planner_kwargs = planner_kwargs  # small model: cypher generation

    # ── public API ───────────────────────────────────────────────────────────

    async def solve(self, question: str) -> tuple[str, list[dict]]:
        """Return (answer, graph_facts).

        graph_facts is a list of dicts for the /api/chat GraphFact response.
        Returns ("", []) on total failure so the caller can fall back.
        """
        cypher = await self._generate_cypher(question)
        if not cypher:
            logger.warning("[Cypher] LLM returned empty query")
            return "", []

        logger.info(f"[Cypher] generated: {cypher}")

        rows, cols = await self._run_cypher(cypher)
        if rows is None:
            # query execution error
            logger.warning("[Cypher] query failed, falling back")
            return "", []

        facts = self._rows_to_facts(rows, cols)
        answer = await self._format_answer(question, rows, cols)
        return answer, facts

    # ── Cypher generation ────────────────────────────────────────────────────

    async def _generate_cypher(self, question: str) -> str:
        import httpx

        prompt = (
            f"{_SCHEMA_PROMPT}\n\n"
            f"使用者問題：{question}\n\n"
            "請生成對應的 Cypher 查詢："
        )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self._planner_kwargs['base_url']}/chat/completions",
                    headers={"Authorization": f"Bearer {self._planner_kwargs['api_key']}"},
                    json={
                        "model": self._planner_kwargs["model"],
                        "temperature": 0.0,
                        "max_tokens": 512,
                        "messages": [
                            {"role": "system", "content": _CYPHER_GEN_SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                return _strip_code_fences(raw)
        except Exception as exc:
            logger.error(f"[Cypher] generation error: {exc}")
            return ""

    # ── Neo4j execution ──────────────────────────────────────────────────────

    async def _run_cypher(
        self, cypher: str
    ) -> tuple[list[dict[str, Any]] | None, list[str]]:
        """Execute Cypher against ucllab database.

        Returns (rows, column_names) or (None, []) on error.
        """
        try:
            from neo4j import AsyncGraphDatabase

            async with AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            ) as driver:
                async with driver.session(database="ucllab") as session:
                    result = await session.run(cypher)
                    records = await result.data()
                    keys = list(records[0].keys()) if records else []
                    return records, keys
        except Exception as exc:
            logger.error(f"[Cypher] execution error: {exc}\nQuery: {cypher}")
            return None, []

    # ── answer formatting ────────────────────────────────────────────────────

    async def _format_answer(
        self, question: str, rows: list[dict], cols: list[str]
    ) -> str:
        import httpx

        if not rows:
            return "根據知識圖譜查詢，找不到與此問題相關的資料。"

        # Compact JSON representation of results for the LLM
        results_text = json.dumps(rows, ensure_ascii=False, indent=2)
        user_msg = (
            f"問題：{question}\n\n"
            f"知識圖譜查詢結果（JSON）：\n{results_text}\n\n"
            "請根據以上結果，用繁體中文給出清晰的回答："
        )

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self._llm_kwargs['base_url']}/chat/completions",
                    headers={"Authorization": f"Bearer {self._llm_kwargs['api_key']}"},
                    json={
                        "model": self._llm_kwargs["model"],
                        "temperature": 0.1,
                        "max_tokens": 1024,
                        "messages": [
                            {"role": "system", "content": _FORMAT_SYSTEM},
                            {"role": "user",   "content": user_msg},
                        ],
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"[Cypher] format error: {exc}")
            # Fallback: plain text dump
            lines = [f"查詢到 {len(rows)} 筆結果："]
            for row in rows[:10]:
                lines.append("  " + "、".join(f"{k}: {v}" for k, v in row.items()))
            return "\n".join(lines)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rows_to_facts(rows: list[dict], cols: list[str]) -> list[dict]:
        """Convert query rows into GraphFact-compatible dicts.

        Tries to produce (subject, predicate, object) triples.
        Falls back to (col_name, "=", value) pairs.
        """
        facts = []
        for row in rows[:20]:
            values = list(row.values())
            if len(values) >= 3:
                facts.append({
                    "subject": str(values[0]),
                    "predicate": str(cols[1]) if len(cols) > 1 else "relates_to",
                    "object": str(values[2]),
                    "source": "neo4j",
                })
            elif len(values) == 2:
                facts.append({
                    "subject": str(values[0]),
                    "predicate": cols[1] if len(cols) > 1 else "=",
                    "object": str(values[1]),
                    "source": "neo4j",
                })
            else:
                for k, v in row.items():
                    facts.append({
                        "subject": k,
                        "predicate": "=",
                        "object": str(v),
                        "source": "neo4j",
                    })
        return facts
