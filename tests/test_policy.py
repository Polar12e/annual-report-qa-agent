import unittest

from src.policy import DataPolicy, PolicyViolation, is_internal_endpoint


class EndpointTests(unittest.TestCase):
    def test_internal_endpoints(self):
        for url in (
            "http://127.0.0.1:11434/v1",
            "http://localhost:8000/v1",
            "http://10.0.0.5/v1",
            "http://192.168.1.10/v1",
            "https://llm.internal/v1",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_internal_endpoint(url))

    def test_external_endpoints(self):
        for url in (None, "", "https://api.openai.com/v1", "https://api.deepseek.com"):
            with self.subTest(url=url):
                self.assertFalse(is_internal_endpoint(url))


class PolicyTests(unittest.TestCase):
    def test_default_blocks_external_endpoint(self):
        with self.assertRaises(PolicyViolation):
            DataPolicy().ensure_endpoint("https://api.openai.com/v1")

    def test_allow_external_opt_in(self):
        DataPolicy(allow_external=True).ensure_endpoint("https://api.openai.com/v1")

    def test_internal_endpoint_is_allowed(self):
        DataPolicy().ensure_endpoint("http://127.0.0.1:11434/v1")

    def test_snippet_is_truncated(self):
        self.assertEqual(DataPolicy(max_snippet_chars=3).snippet("abcdef"), "abc")

    def test_audit_hook_receives_event(self):
        events = []
        DataPolicy(audit_hook=events.append).audit({"tool": "search_report"})
        self.assertEqual(events, [{"tool": "search_report"}])


if __name__ == "__main__":
    unittest.main()
