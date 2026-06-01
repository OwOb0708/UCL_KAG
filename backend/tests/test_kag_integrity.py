"""
KAG structural integrity tests for UCL Lab.

Covers:
  1. RELATION_SCHEMA type validity and coverage vs the UCLLab schema
  2. Template predicate ↔ RELATION_SCHEMA key consistency
  3. UclLabRelationPrompt.parse_response — all predicates, edge cases, normalization
  4. UclLabNERPrompt.parse_response — category filtering, all 12 schema types
  5. build_prompt entity-list JSON serialization

Does NOT require a running OpenSPG server — SDK calls are bypassed via __new__.
Run with: python -m pytest backend/tests/test_kag_integrity.py -v
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

# Allow importing app modules without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompts.ucl_lab_relation import RELATION_SCHEMA, UclLabRelationPrompt
from app.prompts.ucl_lab_ner import UclLabNERPrompt

# ── Ground truth from schema/UCLLab.schema ────────────────────────────────────
SCHEMA_ENTITY_TYPES: frozenset[str] = frozenset({
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

# Entity types that intentionally have zero outgoing edges in RELATION_SCHEMA.
# Updating this set is the correct fix when new relations are added.
KNOWN_UNCOVERED_TYPES: frozenset[str] = frozenset({
    "UCLLab.Patent",
    "UCLLab.ExternalParty",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_relation_prompt() -> UclLabRelationPrompt:
    """Bypass PromptABC.__init__ — only parse_response is under test."""
    return UclLabRelationPrompt.__new__(UclLabRelationPrompt)


def _make_ner_prompt(valid_categories: frozenset[str]) -> UclLabNERPrompt:
    """Bypass SchemaClient call — only parse_response is under test."""
    obj = UclLabNERPrompt.__new__(UclLabNERPrompt)
    obj._valid_categories = valid_categories
    return obj


def _extract_template_predicates() -> set[str]:
    """
    Parse the predicate keys out of UclLabRelationPrompt.template_zh.
    Replaces $-variables with safe placeholders before JSON-parsing the template.
    """
    template = UclLabRelationPrompt.template_zh
    sanitized = re.sub(r"\$entity_list", "null", template)
    sanitized = re.sub(r"\$input", "test_placeholder", sanitized)
    data = json.loads(sanitized)
    return set(data.get("predicates", {}).keys())


# ══════════════════════════════════════════════════════════════════════════════
# 1. RELATION_SCHEMA structural integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestRelationSchemaStructure(unittest.TestCase):
    """RELATION_SCHEMA must be self-consistent with the UCLLab entity type definitions."""

    def test_all_subject_types_are_valid_schema_types(self):
        invalid = {
            pred: s_type
            for pred, (s_type, _) in RELATION_SCHEMA.items()
            if s_type not in SCHEMA_ENTITY_TYPES
        }
        self.assertFalse(invalid, f"Subject types not in schema: {invalid}")

    def test_all_object_types_are_valid_schema_types(self):
        invalid = {
            pred: o_type
            for pred, (_, o_type) in RELATION_SCHEMA.items()
            if o_type not in SCHEMA_ENTITY_TYPES
        }
        self.assertFalse(invalid, f"Object types not in schema: {invalid}")

    def test_template_predicates_all_in_relation_schema(self):
        """Every predicate listed in template_zh must have an entry in RELATION_SCHEMA."""
        template_preds = _extract_template_predicates()
        missing = template_preds - set(RELATION_SCHEMA.keys())
        self.assertFalse(
            missing,
            f"Template predicates missing from RELATION_SCHEMA: {missing}",
        )

    def test_relation_schema_keys_all_in_template(self):
        """Every RELATION_SCHEMA key must be listed in template_zh predicates."""
        template_preds = _extract_template_predicates()
        extra = set(RELATION_SCHEMA.keys()) - template_preds
        self.assertFalse(
            extra,
            f"RELATION_SCHEMA keys not in template predicates: {extra}",
        )

    def test_lead_plan_lead_comp_normalize_to_lead(self):
        """lead_plan and lead_comp are aliases — their subject types must both be Person."""
        for alias in ("lead_plan", "lead_comp"):
            self.assertIn(alias, RELATION_SCHEMA, f"'{alias}' missing from RELATION_SCHEMA")
            s_type, _ = RELATION_SCHEMA[alias]
            self.assertEqual(
                s_type, "UCLLab.Person",
                f"'{alias}' subject must be Person, got '{s_type}'",
            )

    def test_entity_type_coverage_no_unexpected_gaps(self):
        """
        Every schema entity type must appear in at least one RELATION_SCHEMA edge
        (as subject OR object) — except the types listed in KNOWN_UNCOVERED_TYPES.
        """
        covered = set()
        for s_type, o_type in RELATION_SCHEMA.values():
            covered.add(s_type)
            covered.add(o_type)

        uncovered = SCHEMA_ENTITY_TYPES - covered
        unexpected = uncovered - KNOWN_UNCOVERED_TYPES
        self.assertFalse(
            unexpected,
            f"Unexpected entity types with zero RELATION_SCHEMA coverage: {unexpected}\n"
            f"(Known/accepted gaps: {uncovered & KNOWN_UNCOVERED_TYPES})",
        )

    def test_template_is_valid_json_after_variable_substitution(self):
        """template_zh must produce parseable JSON when $-variables are substituted."""
        try:
            _extract_template_predicates()
        except json.JSONDecodeError as exc:
            self.fail(f"template_zh is not valid JSON after substitution: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. UclLabRelationPrompt.parse_response
# ══════════════════════════════════════════════════════════════════════════════

class TestRelationParseResponse(unittest.TestCase):
    """parse_response must convert LLM output to correct [s, s_cat, pred, o, o_cat] tuples."""

    def setUp(self):
        self.prompt = _make_relation_prompt()

    def _parse(self, text: str):
        return self.prompt.parse_response(text)

    # ── basic correctness ──

    def test_valid_json_produces_5_tuple(self):
        rsp = '[{"subject": "王小明", "predicate": "lead", "object": "ToneTone高分專案"}]'
        result = self._parse(rsp)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0],
            ["王小明", "UCLLab.Person", "lead", "ToneTone高分專案", "UCLLab.Project"],
        )

    def test_python_repr_single_quotes_fallback(self):
        """LLM frequently returns Python repr — ast.literal_eval fallback must handle it."""
        rsp = "[{'subject': '王小明', 'predicate': 'lead', 'object': 'ToneTone高分專案'}]"
        result = self._parse(rsp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "lead")

    def test_whitespace_stripped_from_names(self):
        rsp = '[{"subject": "  王小明  ", "predicate": "lead", "object": "  ToneTone  "}]'
        result = self._parse(rsp)
        self.assertEqual(result[0][0], "王小明")
        self.assertEqual(result[0][3], "ToneTone")

    def test_text_wrapped_around_list_extracted(self):
        """Response sometimes has explanation text before/after the JSON list."""
        rsp = "Here are the relations:\n[{\"subject\": \"李大華\", \"predicate\": \"admin\", \"object\": \"DGX伺服器\"}]\nDone."
        result = self._parse(rsp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "admin")

    def test_multiple_relations_all_returned(self):
        data = [
            {"subject": "王小明", "predicate": "assignedTo", "object": "完成API串接"},
            {"subject": "完成API串接", "predicate": "belongsTo", "object": "ToneTone高分專案"},
        ]
        result = self._parse(json.dumps(data))
        self.assertEqual(len(result), 2)

    # ── predicate normalization ──

    def test_lead_plan_normalized_to_lead(self):
        rsp = '[{"subject": "張老師", "predicate": "lead_plan", "object": "115教學計畫"}]'
        result = self._parse(rsp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "lead")
        self.assertEqual(result[0][4], "UCLLab.ResearchPlan")

    def test_lead_comp_normalized_to_lead(self):
        rsp = '[{"subject": "王小明", "predicate": "lead_comp", "object": "AIGO競賽"}]'
        result = self._parse(rsp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "lead")
        self.assertEqual(result[0][4], "UCLLab.Competition")

    # ── filtering / drops ──

    def test_unknown_predicate_dropped(self):
        rsp = '[{"subject": "A", "predicate": "unknownPred", "object": "B"}]'
        self.assertEqual(self._parse(rsp), [])

    def test_invented_predicate_dropped(self):
        rsp = '[{"subject": "A", "predicate": "worksOn", "object": "B"}]'
        self.assertEqual(self._parse(rsp), [])

    def test_missing_subject_skipped(self):
        rsp = '[{"predicate": "lead", "object": "專案A"}]'
        self.assertEqual(self._parse(rsp), [])

    def test_missing_predicate_skipped(self):
        rsp = '[{"subject": "王小明", "object": "專案A"}]'
        self.assertEqual(self._parse(rsp), [])

    def test_missing_object_skipped(self):
        rsp = '[{"subject": "王小明", "predicate": "lead"}]'
        self.assertEqual(self._parse(rsp), [])

    def test_non_dict_items_in_list_skipped(self):
        rsp = '["not a dict", {"subject": "A", "predicate": "lead", "object": "B"}]'
        result = self._parse(rsp)
        self.assertEqual(len(result), 1)

    # ── empty / degenerate inputs ──

    def test_empty_string_returns_empty(self):
        self.assertEqual(self._parse(""), [])

    def test_empty_json_list_returns_empty(self):
        self.assertEqual(self._parse("[]"), [])

    def test_bare_dict_without_output_key_returns_empty(self):
        rsp = '{"subject": "A", "predicate": "lead", "object": "B"}'
        self.assertEqual(self._parse(rsp), [])

    def test_dict_with_output_key_unwrapped(self):
        data = {"output": [{"subject": "A", "predicate": "lead", "object": "B"}]}
        result = self._parse(json.dumps(data))
        self.assertEqual(len(result), 1)

    # ── full predicate coverage ──

    def test_every_relation_schema_predicate_produces_valid_tuple(self):
        """Each predicate in RELATION_SCHEMA must round-trip through parse_response."""
        for pred, (expected_s_type, expected_o_type) in RELATION_SCHEMA.items():
            with self.subTest(predicate=pred):
                rsp = json.dumps([{"subject": "A", "predicate": pred, "object": "B"}])
                result = self._parse(rsp)
                self.assertEqual(len(result), 1, f"Predicate '{pred}' produced no output")
                _, s_type, _, _, o_type = result[0]
                self.assertEqual(s_type, expected_s_type)
                self.assertEqual(o_type, expected_o_type)


# ══════════════════════════════════════════════════════════════════════════════
# 3. UclLabNERPrompt.parse_response
# ══════════════════════════════════════════════════════════════════════════════

class TestNERParseResponse(unittest.TestCase):
    """parse_response must pass valid schema categories and drop everything else."""

    def setUp(self):
        self.prompt = _make_ner_prompt(SCHEMA_ENTITY_TYPES)

    def _parse(self, data) -> list:
        payload = json.dumps(data) if isinstance(data, (list, dict)) else data
        return self.prompt.parse_response(payload)

    def test_all_12_schema_types_pass_filter(self):
        """Entities whose category is any of the 12 schema types must not be dropped."""
        entities = [
            {"name": t.split(".")[1], "category": t, "type": "t", "description": ""}
            for t in sorted(SCHEMA_ENTITY_TYPES)
        ]
        result = self._parse(entities)
        self.assertEqual(
            len(result), len(SCHEMA_ENTITY_TYPES),
            f"Expected {len(SCHEMA_ENTITY_TYPES)} entities, got {len(result)}",
        )

    def test_invalid_category_dropped(self):
        entities = [
            {"name": "A", "category": "UCLLab.Person", "type": "t", "description": ""},
            {"name": "B", "category": "UCLLab.System", "type": "t", "description": ""},
        ]
        result = self._parse(entities)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "A")

    def test_none_category_dropped(self):
        entities = [{"name": "SY", "category": None, "type": "t", "description": ""}]
        self.assertEqual(self._parse(entities), [])

    def test_empty_category_dropped(self):
        entities = [{"name": "X", "category": "", "type": "t", "description": ""}]
        self.assertEqual(self._parse(entities), [])

    def test_missing_category_key_dropped(self):
        entities = [{"name": "X", "type": "t", "description": ""}]
        self.assertEqual(self._parse(entities), [])

    def test_non_json_response_returns_empty(self):
        self.assertEqual(self.prompt.parse_response("not json at all"), [])

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.prompt.parse_response(""), [])

    def test_dict_with_output_key_unwrapped(self):
        data = {"output": [{"name": "A", "category": "UCLLab.Person", "type": "t", "description": ""}]}
        self.assertEqual(len(self._parse(data)), 1)

    def test_dict_with_named_entities_key_unwrapped(self):
        data = {"named_entities": [{"name": "B", "category": "UCLLab.Project", "type": "t", "description": ""}]}
        self.assertEqual(len(self._parse(data)), 1)

    def test_non_dict_items_in_list_skipped(self):
        data = [
            "not a dict",
            {"name": "A", "category": "UCLLab.Person", "type": "t", "description": ""},
        ]
        self.assertEqual(len(self._parse(data)), 1)

    def test_non_list_non_dict_json_returns_empty(self):
        self.assertEqual(self.prompt.parse_response('"just a string"'), [])

    def test_mixed_valid_invalid_categories(self):
        entities = [
            {"name": "Person", "category": "UCLLab.Person", "type": "t", "description": ""},
            {"name": "Task", "category": "UCLLab.Task", "type": "t", "description": ""},
            {"name": "Bad", "category": "UCLLab.Unknown", "type": "t", "description": ""},
            {"name": "Competition", "category": "UCLLab.Competition", "type": "t", "description": ""},
        ]
        result = self._parse(entities)
        self.assertEqual(len(result), 3)
        names = {e["name"] for e in result}
        self.assertNotIn("Bad", names)


# ══════════════════════════════════════════════════════════════════════════════
# 4. build_prompt entity-list JSON serialization
# ══════════════════════════════════════════════════════════════════════════════

class TestEntityListSerialization(unittest.TestCase):
    """
    The entity_list serialization logic in UclLabRelationPrompt.build_prompt
    must produce valid JSON (not Python repr) for the LLM to parse correctly.
    Tested via the same transformation the method applies.
    """

    def _serialize(self, entity_list: list) -> str:
        """Mirror the logic in build_prompt."""
        names = [
            e.get("name", "") if isinstance(e, dict) else str(e)
            for e in entity_list
        ]
        return json.dumps(names, ensure_ascii=False)

    def test_list_of_dicts_extracts_name_field(self):
        entities = [{"name": "王小明", "category": "UCLLab.Person"}, {"name": "ToneTone"}]
        result = json.loads(self._serialize(entities))
        self.assertEqual(result, ["王小明", "ToneTone"])

    def test_list_of_strings_passthrough(self):
        result = json.loads(self._serialize(["王小明", "ToneTone"]))
        self.assertEqual(result, ["王小明", "ToneTone"])

    def test_output_is_double_quoted_json_not_python_repr(self):
        entities = [{"name": "測試"}]
        serialized = self._serialize(entities)
        self.assertNotIn("'", serialized)
        self.assertIn('"', serialized)
        self.assertIsInstance(json.loads(serialized), list)

    def test_chinese_characters_not_ascii_escaped(self):
        entities = [{"name": "王小明"}]
        serialized = self._serialize(entities)
        self.assertIn("王小明", serialized)

    def test_dict_without_name_key_yields_empty_string(self):
        entities = [{"category": "UCLLab.Person"}]
        result = json.loads(self._serialize(entities))
        self.assertEqual(result, [""])

    def test_empty_list_produces_empty_json_array(self):
        result = json.loads(self._serialize([]))
        self.assertEqual(result, [])

    def test_large_entity_list_all_names_extracted(self):
        entities = [{"name": f"Person{i}"} for i in range(50)]
        result = json.loads(self._serialize(entities))
        self.assertEqual(len(result), 50)
        self.assertEqual(result[0], "Person0")
        self.assertEqual(result[-1], "Person49")


if __name__ == "__main__":
    unittest.main(verbosity=2)
