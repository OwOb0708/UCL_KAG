"""Relation extraction prompt for UCL Lab meeting notes.

Output format: JSON list of {subject, predicate, object} objects.
parse_response() converts to 5-tuples for SchemaConstraintExtractor:
  [subject_name, subject_category, predicate, object_name, object_category]
"""

from __future__ import annotations

import ast
import json
from typing import List

from kag.interface.common.prompt import PromptABC


RELATION_SCHEMA: dict[str, tuple[str, str]] = {
    "assignedTo":     ("UCLLab.Person",        "UCLLab.Task"),
    "belongsTo":      ("UCLLab.Task",           "UCLLab.Project"),
    "lead":           ("UCLLab.Person",         "UCLLab.Project"),
    "members":        ("UCLLab.Person",         "UCLLab.Project"),
    "consultant":     ("UCLLab.Person",         "UCLLab.Project"),
    "decidedIn":      ("UCLLab.Task",           "UCLLab.MeetingRecord"),
    "madeIn":         ("UCLLab.Decision",       "UCLLab.MeetingRecord"),
    "updatesProject": ("UCLLab.MeetingRecord",  "UCLLab.Project"),
    "admin":          ("UCLLab.Person",         "UCLLab.ComputeResource"),
    "usedBy":         ("UCLLab.Project",        "UCLLab.ComputeResource"),
    "relatedProject": ("UCLLab.ResearchPlan",   "UCLLab.Project"),
    "executor":       ("UCLLab.Person",         "UCLLab.Decision"),
    "affects":        ("UCLLab.Decision",       "UCLLab.Project"),
    "manager":        ("UCLLab.Person",         "UCLLab.Equipment"),
    "ta":             ("UCLLab.Person",         "UCLLab.Course"),
    "instructor":     ("UCLLab.Person",         "UCLLab.Course"),
    "lead_plan":      ("UCLLab.Person",         "UCLLab.ResearchPlan"),
    "teamMembers":    ("UCLLab.Person",         "UCLLab.Competition"),
    "lead_comp":      ("UCLLab.Person",         "UCLLab.Competition"),
}


class UclLabRelationPrompt(PromptABC):
    """Relation extraction prompt for UCL Lab meeting minutes (Chinese)."""

    template_zh = """
{
    "instruction": "你是關係抽取專家，專門處理台灣大學實驗室會議記錄。請根據已識別的實體列表，從輸入文字中找出實體之間的關係，以 JSON list 格式輸出。每個關係物件包含三個欄位：subject（主體名稱）、predicate（關係類型）、object（客體名稱）。【重要】predicate 必須嚴格使用以下列表中的其中一個值，不可自創。若找不到任何關係，輸出空 list []。",
    "predicates": {
        "lead": "某人是專案/比賽/計畫的主要負責人",
        "members": "某人是專案的參與成員",
        "consultant": "某人是專案的顧問",
        "assignedTo": "某人負責某待辦事項/任務",
        "belongsTo": "某待辦事項/任務屬於某專案",
        "admin": "某人管理某計算資源（GPU/伺服器）",
        "usedBy": "某專案使用某計算資源",
        "manager": "某人負責某設備",
        "ta": "某人擔任某課程助教",
        "instructor": "某人是某課程授課教師",
        "lead_plan": "某人負責某研究計畫/教學計畫",
        "relatedProject": "某研究計畫關聯某專案",
        "teamMembers": "某人是比賽參賽成員",
        "lead_comp": "某人是比賽負責人",
        "decidedIn": "某待辦事項決議於某次會議",
        "madeIn": "某決議來自某次會議",
        "updatesProject": "某次會議涉及某專案",
        "executor": "某人執行某決議",
        "affects": "某決議影響某專案"
    },
    "example": [
        {
            "input": "ToneTone 高分專案（進行中）：負責人王小明，顧問張老師，成員李大華。子任務：完成 API 串接（已完成，負責人王小明）。DGX 伺服器由李大華管理。",
            "entities": ["王小明", "張老師", "李大華", "ToneTone 高分專案", "完成 API 串接", "DGX 伺服器"],
            "output": [
                {"subject": "王小明", "predicate": "lead", "object": "ToneTone 高分專案"},
                {"subject": "張老師", "predicate": "consultant", "object": "ToneTone 高分專案"},
                {"subject": "李大華", "predicate": "members", "object": "ToneTone 高分專案"},
                {"subject": "王小明", "predicate": "assignedTo", "object": "完成 API 串接"},
                {"subject": "完成 API 串接", "predicate": "belongsTo", "object": "ToneTone 高分專案"},
                {"subject": "李大華", "predicate": "admin", "object": "DGX 伺服器"}
            ]
        }
    ],
    "entities": $entity_list,
    "input": "$input"
}
"""

    template_en = """
{
    "instruction": "You are a relation extraction expert for university lab meeting minutes. Given the entity list, extract relations from the input text. Output a JSON list where each item has: subject (entity name), predicate (relation type from the list below), object (entity name). If no relations found, output []. Only use predicates from the provided list.",
    "predicates": {
        "lead": "Person leads a Project/Competition/Plan",
        "members": "Person is a member of a Project",
        "consultant": "Person is advisor/consultant of a Project",
        "assignedTo": "Person is assigned to a Task",
        "belongsTo": "Task belongs to a Project",
        "admin": "Person manages a ComputeResource",
        "usedBy": "Project uses a ComputeResource",
        "manager": "Person manages Equipment",
        "ta": "Person is TA for a Course",
        "instructor": "Person teaches a Course",
        "lead_plan": "Person leads a ResearchPlan",
        "relatedProject": "ResearchPlan is related to a Project",
        "teamMembers": "Person is a team member in a Competition",
        "lead_comp": "Person leads a Competition",
        "decidedIn": "Task was decided in a MeetingRecord",
        "madeIn": "Decision was made in a MeetingRecord",
        "updatesProject": "MeetingRecord discusses a Project",
        "executor": "Person executes a Decision",
        "affects": "Decision affects a Project"
    },
    "entities": $entity_list,
    "input": "$input"
}
"""

    @property
    def template_variables(self) -> List[str]:
        return ["input", "entity_list"]

    def build_prompt(self, variables: dict) -> str:
        entity_list = variables.get("entity_list", [])
        if isinstance(entity_list, list):
            names = [
                e.get("name", "") if isinstance(e, dict) else str(e)
                for e in entity_list
            ]
            variables = dict(variables, entity_list=json.dumps(names, ensure_ascii=False))
        return super().build_prompt(variables)

    def parse_response(self, response: str, **kwargs):
        if not isinstance(response, str):
            response = str(response)

        # Try to parse JSON list from response (LLM sometimes returns Python dict format)
        rsp = response.strip()
        # Extract the list portion if wrapped in other text
        start = rsp.find("[")
        end = rsp.rfind("]") + 1
        if start >= 0 and end > start:
            rsp = rsp[start:end]
        data = None
        for parser in (json.loads, ast.literal_eval):
            try:
                data = parser(rsp)
                break
            except Exception:
                continue
        if data is None:
            return []

        if not isinstance(data, list):
            if isinstance(data, dict) and "output" in data:
                data = data["output"]
            else:
                return []

        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            s_name = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            o_name = str(item.get("object", "")).strip()
            if not s_name or not predicate or not o_name:
                continue

            graph_pred = predicate
            if predicate in ("lead_plan", "lead_comp"):
                graph_pred = "lead"

            schema_entry = RELATION_SCHEMA.get(predicate)
            if schema_entry is None:
                print(f"[REL] dropping unknown predicate '{predicate}': {s_name} → {o_name}")
                continue
            s_category, o_category = schema_entry
            results.append([s_name, s_category, graph_pred, o_name, o_category])

        if results:
            print(f"[REL] extracted {len(results)} relations")
        return results
