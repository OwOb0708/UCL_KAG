"""Custom NER prompt for UCL Lab meeting notes.

Replaces the default OpenIENERPrompt (which uses a movie example and lets the LLM
invent category names) with a UCL-lab-specific prompt that:
  - shows a Chinese meeting-note extraction example
  - instructs the LLM to ONLY use category names from the project schema
  - filters out any entity whose category is not in the schema at parse time
"""

from __future__ import annotations

import json
from string import Template
from typing import List

from kag.interface.common.prompt import PromptABC


class UclLabNERPrompt(PromptABC):
    """Schema-enforcing NER prompt for UCL Lab meeting minutes (Chinese)."""

    template_zh = """
{
    "instruction": "你是命名實體識別的專家，專門處理台灣大學實驗室的會議記錄。請從輸入文字中抽取所有重要實體。每個實體必須包含四個欄位：name（名稱）、category（類別）、type（具體類型）、description（簡短描述）。【重要】category 欄位必須嚴格使用以下 schema 列表中的其中一個值，絕對不可以自創任何新的類別名稱。如果某個實體無法歸類到 schema 中的任何類別，請直接略過，不要輸出。請以 JSON 格式回應，只輸出實體列表（list）。",
    "schema": $schema,
    "category_mapping": "人員→UCLLab.Person, 專案/系統→UCLLab.Project, 待辦/子任務→UCLLab.Task, 會議記錄→UCLLab.MeetingRecord, 決議→UCLLab.Decision, 研究計畫/補助計畫→UCLLab.ResearchPlan, 專利→UCLLab.Patent, GPU/伺服器/雲端→UCLLab.ComputeResource, 設備/硬體/車輛→UCLLab.Equipment, 課程→UCLLab.Course, 廠商/合作單位→UCLLab.ExternalParty, 比賽→UCLLab.Competition",
    "person_rules": "【Person 嚴格規則】(1) 每個節點只能包含「一個」人名，絕對不可把多人放在同一節點。錯誤：'文嬡 楨惟'、'楨惟 子恩'；正確：分別建立'文嬡'和'楨惟'兩個節點。(2) 人名不可包含職稱、任務名稱、系統名稱。錯誤：'通知系統 楨惟'、'辰旭學長'；正確：'楨惟'、'辰旭'。(3) 只取名字部分（去掉姓氏亦可）：劉宇軒→宇軒, 曾子恩→子恩, 翁楨惟→楨惟, 余欣恬→欣恬, 簡婉庭→婉庭, 夏辰旭→辰旭。(4) 指導老師統一用全名：朱彥銘。",
    "example": [
        {
            "input": "114-2 研究生會議記錄（2025/03/14）\n出席：王小明、李大華、張老師\n請假：陳美玲\n一、庶務事項\n(1) 114-2 助教：深度學習課程（課號：CS5001）由王小明擔任助教，李大華擔任顧問助教。\n(2) 公務車（車牌 ABC-1234）下次保養日期為 2025/04/01。\n(3) 合作廠商波力梅提供 SIM 卡設備。\n二、比賽事項\n(1) AIGO 競賽：報名截止 2025/03/20，決賽地點台北，獎金 10 萬元，負責人王小明，參賽成員李大華、陳美玲。\n三、專案事項\n(1) ToneTone 高分專案（進行中）：負責人王小明，顧問張老師。\n    a. 完成 API 串接（已完成）\n    b. UI 優化（進行中，期程 2025/03/28）\n四、計畫事項\n(1) 115 教學實踐計畫（執行中）：負責人張老師，相關專案 ToneTone 高分。\n五、DGX 伺服器（GPU伺服器）由李大華管理，目前狀態正常。\n六、提案討論\n(1) 決議：王小明須於 2025/03/21 前提交期末報告，執行狀態：待執行。",
            "output": [
                {"name": "王小明", "type": "研究生", "category": "UCLLab.Person", "description": "實驗室研究生，擔任多個專案負責人及助教。"},
                {"name": "李大華", "type": "研究生", "category": "UCLLab.Person", "description": "實驗室研究生，管理 DGX 伺服器。"},
                {"name": "張老師", "type": "教師", "category": "UCLLab.Person", "description": "實驗室指導教師。"},
                {"name": "陳美玲", "type": "研究生", "category": "UCLLab.Person", "description": "實驗室研究生，請假。"},
                {"name": "深度學習課程", "type": "課程", "category": "UCLLab.Course", "description": "114-2 學期課程，課號 CS5001。"},
                {"name": "公務車", "type": "車輛", "category": "UCLLab.Equipment", "description": "車牌 ABC-1234，下次保養日期 2025/04/01。"},
                {"name": "波力梅", "type": "廠商", "category": "UCLLab.ExternalParty", "description": "合作廠商，提供 SIM 卡設備。"},
                {"name": "AIGO 競賽", "type": "比賽", "category": "UCLLab.Competition", "description": "報名截止 2025/03/20，決賽在台北，獎金 10 萬元。"},
                {"name": "ToneTone 高分專案", "type": "專案", "category": "UCLLab.Project", "description": "進行中的專案，負責人王小明。"},
                {"name": "完成 API 串接", "type": "待辦事項", "category": "UCLLab.Task", "description": "ToneTone 高分專案子任務，狀態已完成。"},
                {"name": "UI 優化", "type": "待辦事項", "category": "UCLLab.Task", "description": "ToneTone 高分專案子任務，進行中，期程 2025/03/28。"},
                {"name": "115 教學實踐計畫", "type": "教學計畫", "category": "UCLLab.ResearchPlan", "description": "執行中，負責人張老師。"},
                {"name": "DGX 伺服器", "type": "GPU伺服器", "category": "UCLLab.ComputeResource", "description": "由李大華管理，狀態正常。"},
                {"name": "王小明須於 2025/03/21 前提交期末報告", "type": "決議", "category": "UCLLab.Decision", "description": "執行狀態待執行，期限 2025/03/21。"}
            ]
        }
    ],
    "input": "$input"
}
"""

    template_en = """
{
    "instruction": "You are a named entity recognition expert for university research lab meeting minutes. Extract important entities. Each entity MUST have: name, category (MUST be exactly one of the schema values below — never invent new categories), type, description. Skip any entity that does not fit a schema category. Return empty list if nothing matches. Respond with only a JSON list.",
    "schema": $schema,
    "example": [
        {
            "input": "Meeting 2025-03-14. Attendees: Alice (student), Bob (TA), Prof. Chen. Project ToneTone ongoing, led by Alice. Task: complete API integration (done). DGX server managed by Bob, status normal. Decision: Alice must submit final report by 2025-03-21.",
            "output": [
                {"name": "Alice", "type": "Student", "category": "UCLLab.Person", "description": "Lab student, leads ToneTone project."},
                {"name": "Bob", "type": "TA", "category": "UCLLab.Person", "description": "Lab student, manages DGX server."},
                {"name": "Prof. Chen", "type": "Professor", "category": "UCLLab.Person", "description": "Lab advisor."},
                {"name": "ToneTone", "type": "Project", "category": "UCLLab.Project", "description": "Ongoing project led by Alice."},
                {"name": "Complete API integration", "type": "Task", "category": "UCLLab.Task", "description": "Completed task under ToneTone project."},
                {"name": "DGX server", "type": "GPU Server", "category": "UCLLab.ComputeResource", "description": "Managed by Bob, status normal."},
                {"name": "Alice must submit final report by 2025-03-21", "type": "Decision", "category": "UCLLab.Decision", "description": "Deadline 2025-03-21."}
            ]
        }
    ],
    "input": "$input"
}
"""

    # Hardcoded to the 12 types defined in UCLLab.schema.
    # Do NOT load from session.spg_types — that includes all Neo4j labels ever created
    # (including LLM-invented types), causing a snowball effect where more types get
    # accepted each run, making the filter useless.
    _SCHEMA_TYPES = frozenset({
        "UCLLab.Person",
        "UCLLab.Project",
        "UCLLab.Task",
        "UCLLab.MeetingRecord",
        "UCLLab.Decision",
        "UCLLab.ResearchPlan",
        "UCLLab.Patent",
        "UCLLab.ComputeResource",
        "UCLLab.Equipment",
        "UCLLab.Course",
        "UCLLab.ExternalParty",
        "UCLLab.Competition",
    })

    def __init__(self, language: str = "", **kwargs):
        super().__init__(language, **kwargs)
        self._valid_categories = self._SCHEMA_TYPES
        schema_list = sorted(self._valid_categories)
        self.template = Template(self.template).safe_substitute(
            schema=json.dumps(schema_list, ensure_ascii=False)
        )

    @property
    def template_variables(self) -> List[str]:
        return ["input"]

    def parse_response(self, response: str, **kwargs):
        rsp = response
        if isinstance(rsp, str):
            try:
                rsp = json.loads(rsp)
            except json.JSONDecodeError:
                return []
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict) and "named_entities" in rsp:
            rsp = rsp["named_entities"]
        if not isinstance(rsp, list):
            return []

        valid = []
        for entity in rsp:
            if not isinstance(entity, dict):
                continue
            cat = entity.get("category", "")
            if cat not in self._valid_categories:
                print(f"[NER] dropping invalid category '{cat}' for entity '{entity.get('name', '?')}'")
                continue
            # Strip surrounding quotes that some LLM outputs add (e.g. '"子恩"' → '子恩')
            raw_name = entity.get("name", "")
            clean_name = raw_name.strip().strip('"').strip("'").strip()
            if not clean_name:
                continue
            entity["name"] = clean_name
            valid.append(entity)
        return valid
