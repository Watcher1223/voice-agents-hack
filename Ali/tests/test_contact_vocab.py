"""
Unit tests for config.contact_vocab.

Covers the pieces we can exercise without touching Contacts.app:
  * filter_unusual() — keep/reject matrix against the "unusual" heuristics
  * expand_mis_splits() — the phrases we hope Deepgram also emits
  * cache TTL — fresh vs. stale payloads via monkeypatched time/paths
  * vocab.apply_corrections() — static rules still gate on hints, auto
    rules fire unconditionally
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from config import contact_vocab
from config.contact_vocab import (
    expand_mis_splits,
    filter_unusual,
    get_mis_split_rules,
    get_unusual_first_names,
)


class FilterUnusualTests(unittest.TestCase):
    def test_embedded_capital_kept(self):
        self.assertEqual(
            filter_unusual(["Alspencer", "DeAndre", "McKinley", "LeBron"]),
            ["Alspencer", "DeAndre", "McKinley", "LeBron"],
        )

    def test_non_ascii_kept(self):
        kept = filter_unusual(["Éloïse", "Tōru", "Søren"])
        self.assertEqual(kept, ["Éloïse", "Tōru", "Søren"])

    def test_apostrophe_cap_kept(self):
        self.assertIn("D'Angelo", filter_unusual(["D'Angelo"]))

    def test_long_unfamiliar_kept(self):
        self.assertIn("Krzysztof", filter_unusual(["Krzysztof"]))

    def test_common_names_dropped(self):
        self.assertEqual(filter_unusual(["Alex", "Sam", "Chris", "Mark"]), [])

    def test_short_and_caps_dropped(self):
        self.assertEqual(filter_unusual(["AB", "A", "YCOMB"]), [])

    def test_email_like_tokens_dropped(self):
        # Contacts without a real name surface their email address in the
        # display_name — we must not bias on those.
        self.assertEqual(
            filter_unusual([
                "support@powerflex.com",
                "e.charulak.f@gmail.com",
                "khans107@eq.edu.au",
            ]),
            [],
        )

    def test_dedupe_preserves_order(self):
        kept = filter_unusual(["DeAndre", "deandre", "Alspencer", "DeAndre"])
        self.assertEqual(kept, ["DeAndre", "Alspencer"])

    def test_blank_inputs_ignored(self):
        self.assertEqual(filter_unusual(["", "  ", None]), [])  # type: ignore[list-item]


class ExpandMisSplitsTests(unittest.TestCase):
    def test_alspencer(self):
        out = expand_mis_splits("Alspencer")
        self.assertIn("al spencer", out)
        self.assertIn("all spencer", out)
        self.assertIn("alspencer", out)
        self.assertNotIn("Alspencer", out)  # canonical form excluded

    def test_deandre(self):
        out = expand_mis_splits("DeAndre")
        self.assertIn("de andre", out)
        self.assertIn("deandre", out)

    def test_non_ascii_folded(self):
        out = expand_mis_splits("Éloïse")
        self.assertIn("eloise", out)

    def test_plain_name_empty(self):
        # No boundary, no non-ASCII -> nothing plausible to fold.
        self.assertEqual(expand_mis_splits("Alex"), set())

    def test_multi_word_untouched(self):
        self.assertEqual(expand_mis_splits("Mary Anne"), set())


class MisSplitRuleShapeTests(unittest.TestCase):
    """get_mis_split_rules returns tuples compatible with apply_corrections."""

    def test_empty_hints_means_always_fire(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "contact_vocab.json"
            cache_path.write_text(json.dumps({
                "built_at": time.time(),
                "unusual_first_names": ["Alspencer"],
                "mis_splits": [{"wrong": ["all spencer", "al spencer"], "right": "Alspencer"}],
            }))
            with patch.object(contact_vocab, "_CACHE_PATH", cache_path):
                rules = get_mis_split_rules()
        self.assertEqual(len(rules), 1)
        wrongs, right, hints = rules[0]
        self.assertEqual(right, "Alspencer")
        self.assertSetEqual(hints, set())
        self.assertIn("all spencer", wrongs)


class CacheTTLTests(unittest.TestCase):
    def test_fresh_cache_returned(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "contact_vocab.json"
            cache_path.write_text(json.dumps({
                "built_at": time.time(),
                "unusual_first_names": ["Alspencer", "DeAndre"],
                "mis_splits": [],
            }))
            with patch.object(contact_vocab, "_CACHE_PATH", cache_path):
                names = get_unusual_first_names()
        self.assertEqual(names, ["Alspencer", "DeAndre"])

    def test_stale_cache_ignored(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "contact_vocab.json"
            stale = time.time() - (contact_vocab._CACHE_TTL_SECONDS + 60)
            cache_path.write_text(json.dumps({
                "built_at": stale,
                "unusual_first_names": ["Alspencer"],
                "mis_splits": [],
            }))
            # Short-circuit the rebuild path: no Contacts access in the
            # test harness. Pretend the loader returned nothing.
            with patch.object(contact_vocab, "_CACHE_PATH", cache_path), \
                 patch.object(contact_vocab, "_load_contact_first_names", return_value=[]):
                names = get_unusual_first_names()
        self.assertEqual(names, [])

    def test_disable_env_short_circuits(self):
        with patch.dict("os.environ", {"ALI_CONTACT_VOCAB_DISABLE": "1"}):
            self.assertEqual(get_unusual_first_names(), [])
            self.assertEqual(get_mis_split_rules(), [])


class ApplyCorrectionsIntegrationTests(unittest.TestCase):
    """vocab.apply_corrections should honour both static + auto-derived rules."""

    def test_static_rule_still_needs_hint(self):
        from config import vocab

        # No message/mail/call etc. hint -> "Corinne" should stay put.
        self.assertEqual(
            vocab.apply_corrections("Corinne stopped by earlier"),
            "Corinne stopped by earlier",
        )
        # With a hint, it rewrites.
        self.assertEqual(
            vocab.apply_corrections("email Corinne the brief"),
            "email Korin the brief",
        )

    def test_auto_rule_fires_unconditionally(self):
        from config import vocab

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "contact_vocab.json"
            cache_path.write_text(json.dumps({
                "built_at": time.time(),
                "unusual_first_names": ["Alspencer"],
                "mis_splits": [{"wrong": ["all spencer"], "right": "Alspencer"}],
            }))
            with patch.object(contact_vocab, "_CACHE_PATH", cache_path):
                # No contextual hints, but the auto rule should still fire.
                self.assertEqual(
                    vocab.apply_corrections("tell all spencer the news"),
                    "tell Alspencer the news",
                )


if __name__ == "__main__":
    unittest.main()
