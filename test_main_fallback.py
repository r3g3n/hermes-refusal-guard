"""Simulate main-turn refusal → Hermes fallback_providers path (no fake deep)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Plugin dir on sys.path as package parent
PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLUGIN_DIR.parent))


class MainTurnRefusalTests(unittest.TestCase):
    def setUp(self):
        # Import fresh
        if "refusal-guard" in sys.modules:
            del sys.modules["refusal-guard"]
        # package name has hyphen — load via importlib
        import importlib.util
        init = PLUGIN_DIR / "__init__.py"
        spec = importlib.util.spec_from_file_location("refusal_guard_plugin", init)
        self.mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # Ensure detector sibling import works: load as package
        import types
        pkg = types.ModuleType("refusal_guard_pkg")
        pkg.__path__ = [str(PLUGIN_DIR)]
        sys.modules["refusal_guard_pkg"] = pkg
        # Load detector first
        det_spec = importlib.util.spec_from_file_location(
            "refusal_guard_pkg.detector", PLUGIN_DIR / "detector.py"
        )
        det = importlib.util.module_from_spec(det_spec)
        sys.modules["refusal_guard_pkg.detector"] = det
        assert det_spec.loader is not None
        det_spec.loader.exec_module(det)
        spec = importlib.util.spec_from_file_location(
            "refusal_guard_pkg", init,
            submodule_search_locations=[str(PLUGIN_DIR)],
        )
        # Simpler: exec with package context
        self.mod = types.ModuleType("refusal_guard_pkg")
        self.mod.__path__ = [str(PLUGIN_DIR)]
        sys.modules["refusal_guard_pkg"] = self.mod
        code = init.read_text(encoding="utf-8")
        # Rewrite relative import
        code = code.replace("from .detector import", "from refusal_guard_pkg.detector import")
        exec(compile(code, str(init), "exec"), self.mod.__dict__)

        self.tmp = tempfile.TemporaryDirectory()
        self.omh = Path(self.tmp.name) / ".omh"
        (self.omh / "routing").mkdir(parents=True)
        (self.omh / "runtime").mkdir(parents=True)
        self.mod._DISABLED = False
        self._omh_patch = mock.patch.object(self.mod, "_omh_home", return_value=self.omh)
        self._omh_patch.start()

    def tearDown(self):
        self._omh_patch.stop()
        self.tmp.cleanup()

    def test_suggest_main_fallback_primary_not_in_chain(self):
        chain = [
            {"provider": "litellm", "model": "fallback-a"},
            {"provider": "litellm", "model": "fallback-c"},
        ]
        with mock.patch.object(self.mod, "_load_hermes_fallback_providers", return_value=chain):
            sug = self.mod._suggest_main_fallback("primary-model")
        self.assertEqual(sug["next_model"], "fallback-a")
        self.assertEqual(sug["next_provider"], "litellm")
        self.assertEqual(sug["from_index"], -1)

    def test_transform_llm_output_main_path_no_deep(self):
        chain = [
            {"provider": "litellm", "model": "fallback-a"},
            {"provider": "litellm", "model": "fallback-c"},
        ]
        with mock.patch.object(self.mod, "_load_hermes_fallback_providers", return_value=chain), \
             mock.patch.object(self.mod, "_activate_hermes_fallback", return_value={
                 "activated": False, "reason": "no_live_agent",
             }), \
             mock.patch.object(self.mod, "_call_omh_fallback") as omh_fb:
            out = self.mod.transform_llm_output(
                response_text="Я не могу помочь с этим запросом.",
                model="primary-model",
                session_id="sess-test",
                turn_id="turn-1",
            )
        self.assertIsInstance(out, str)
        self.assertIn("path=main", out)
        self.assertIn("do NOT use category=deep", out)
        self.assertIn("fallback-a", out)
        self.assertIn('{"mode": "main"}', out)
        self.assertNotIn("category='deep'", out)
        self.assertNotIn("NOT_APPLIED (no live OMH route", out)
        omh_fb.assert_not_called()
        signal = json.loads((self.omh / "routing" / "refusal-signal.json").read_text())
        self.assertEqual(signal["path"], "main")
        self.assertEqual(signal["next_model"], "fallback-a")
        self.assertIsNone(signal.get("category"))

    def test_refusal_fallback_tool_main_mode(self):
        chain = [{"provider": "litellm", "model": "fallback-a"}]
        with mock.patch.object(self.mod, "_load_hermes_fallback_providers", return_value=chain), \
             mock.patch.object(self.mod, "_activate_hermes_fallback", return_value={
                 "activated": False, "reason": "no_live_agent",
             }):
            raw = self.mod.refusal_fallback_handler({"mode": "main", "reason": "test"})
        data = json.loads(raw)
        self.assertEqual(data["mode"], "main")
        self.assertEqual(data["next_model"], "fallback-a")
        self.assertIsNone(data["category"])
        self.assertEqual(data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
