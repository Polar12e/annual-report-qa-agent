import os
import unittest
from unittest.mock import patch

from src.llm_config import LLMSettings


class LLMSettingsTests(unittest.TestCase):
    def setUp(self):
        # Keep these tests independent of the developer's shell credentials.
        self._env = patch.dict(os.environ, {}, clear=True)
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_deepseek_defaults(self):
        settings = LLMSettings.from_env(provider="deepseek", api_key="sk-test")
        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "deepseek-flash")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")
        self.assertFalse(settings.thinking)
        self.assertEqual(settings.max_retries, 2)

    def test_deepseek_requires_a_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                LLMSettings.from_env(provider="deepseek")

    def test_auto_detects_deepseek_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            settings = LLMSettings.from_env()
        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "deepseek-flash")

    def test_auto_detects_openai_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            settings = LLMSettings.from_env()
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.model, "gpt-4o-mini")
        self.assertEqual(settings.base_url, "https://api.openai.com/v1")

    def test_auto_with_local_base_url_needs_no_key(self):
        settings = LLMSettings.from_env(
            base_url="http://127.0.0.1:11434/v1", model="qwen3:8b"
        )
        self.assertEqual(settings.provider, "local")
        self.assertEqual(settings.api_key, "local")

    def test_custom_external_endpoint_requires_a_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                LLMSettings.from_env(
                    provider="custom",
                    base_url="https://gateway.example.com/v1",
                    model="some-model",
                )

    def test_custom_endpoint_with_explicit_key(self):
        settings = LLMSettings.from_env(
            provider="custom",
            base_url="https://gateway.example.com/v1",
            model="some-model",
            api_key="sk-test",
        )
        self.assertEqual(settings.provider, "custom")
        self.assertEqual(settings.api_key, "sk-test")

    def test_custom_endpoint_with_generic_key(self):
        with patch.dict(os.environ, {"FIN_AGENT_API_KEY": "sk-custom"}, clear=True):
            settings = LLMSettings.from_env(
                provider="custom",
                base_url="https://gateway.example.com/v1",
                model="some-model",
            )
        self.assertEqual(settings.api_key, "sk-custom")

    def test_local_rejects_public_base_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                LLMSettings.from_env(
                    provider="local",
                    base_url="https://api.example.com/v1",
                    model="m",
                )

    def test_explicit_values_override_environment(self):
        with patch.dict(
            os.environ,
            {
                "FIN_AGENT_MODEL": "env-model",
                "FIN_AGENT_TIMEOUT": "99",
                "FIN_AGENT_THINKING": "true",
            },
            clear=True,
        ):
            settings = LLMSettings.from_env(
                provider="deepseek",
                api_key="sk-test",
                model="explicit-model",
                timeout=5,
                thinking=False,
            )
        self.assertEqual(settings.model, "explicit-model")
        self.assertEqual(settings.timeout, 5)
        self.assertFalse(settings.thinking)

    def test_trailing_slash_is_normalized(self):
        settings = LLMSettings.from_env(
            provider="deepseek",
            api_key="sk-test",
            base_url="https://api.deepseek.com/",
        )
        self.assertEqual(settings.base_url, "https://api.deepseek.com")

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            LLMSettings.from_env(provider="magic", api_key="k")
        with self.assertRaises(ValueError):
            LLMSettings.from_env(provider="deepseek", api_key="k", timeout=0)
        with self.assertRaises(ValueError):
            LLMSettings.from_env(provider="deepseek", api_key="k", max_retries=-1)
        with self.assertRaises(ValueError):
            LLMSettings.from_env(provider="deepseek", api_key="k", thinking="maybe")

    def test_repr_and_public_dict_never_leak_the_key(self):
        settings = LLMSettings.from_env(
            provider="deepseek", api_key="sk-very-secret", model="m"
        )
        self.assertNotIn("sk-very-secret", repr(settings))
        self.assertEqual(settings.as_public_dict()["api_key"], "set")
        self.assertNotIn("sk-very-secret", str(settings.as_public_dict()))

    def test_deepseek_detection_uses_base_url(self):
        settings = LLMSettings.from_env(
            provider="custom",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            api_key="sk-test",
        )
        self.assertTrue(settings.is_deepseek)
        self.assertFalse(settings.is_local)


if __name__ == "__main__":
    unittest.main()
