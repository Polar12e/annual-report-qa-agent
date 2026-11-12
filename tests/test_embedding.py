"""Embedding providers: determinism, caching, protocol errors, HTTP handling."""
from __future__ import annotations

import io
import json
import shutil
import unittest
import urllib.error
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.embedding import (
    EmbeddingCache,
    EmbeddingError,
    LsaEmbedder,
    OpenAICompatibleEmbedder,
    corpus_fingerprint,
    embed_documents,
    randomized_svd,
    unit_normalize,
)

FIT = [
    ["存货", "库存", "商品", "存货"],
    ["存货", "库存", "销售", "商品"],
    ["员工", "薪酬", "工资", "员工"],
    ["员工", "人数", "薪酬", "工资"],
    ["营业收入", "销售", "商品", "库存"],
]


@contextmanager
def temp_dir():
    # A project-local directory: tempfile.TemporaryDirectory creates a 0o700
    # DACL that sandboxed Windows processes cannot write into or delete.
    directory = Path(__file__).resolve().parents[1] / "_test_tmp" / "embeddings"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class UnitNormalizeTest(unittest.TestCase):
    def test_rows_are_scaled_to_unit_length(self):
        matrix = unit_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(matrix[0])), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(matrix[1])), 1.0, places=6)

    def test_zero_rows_stay_zero(self):
        matrix = unit_normalize(np.zeros((1, 3), dtype=np.float32))
        self.assertTrue(np.allclose(matrix, 0.0))

    def test_one_dimensional_input_is_promoted_to_a_row(self):
        self.assertEqual(unit_normalize(np.array([3.0, 4.0], dtype=np.float32)).shape, (1, 2))


class RandomizedSvdTest(unittest.TestCase):
    def test_reconstruction_error_is_small_for_low_rank_input(self):
        rng = np.random.default_rng(3)
        low_rank = rng.standard_normal((40, 2)).astype(np.float32) @ rng.standard_normal((2, 25)).astype(np.float32)
        _, basis = randomized_svd(low_rank, components=2, seed=1)
        residual = low_rank - (low_rank @ basis) @ basis.T
        self.assertLess(float(np.abs(residual).max()), 1e-3)

    def test_rejects_non_positive_components(self):
        with self.assertRaises(ValueError):
            randomized_svd(np.zeros((2, 2), dtype=np.float32), components=0)


class LsaEmbedderTest(unittest.TestCase):
    def test_dim_and_unit_norm(self):
        embedder = LsaEmbedder(dim=3).fit(FIT)
        vectors = embedder.encode(["存货 库存", "员工"])
        self.assertEqual(vectors.shape, (2, 3))
        self.assertLess(abs(float(np.linalg.norm(vectors[0])) - 1.0), 1e-5)

    def test_is_deterministic(self):
        first = LsaEmbedder(dim=3, seed=7).fit(FIT).encode(["存货", "员工"])
        second = LsaEmbedder(dim=3, seed=7).fit(FIT).encode(["存货", "员工"])
        self.assertTrue(np.allclose(first, second))

    def test_encode_before_fit_raises(self):
        with self.assertRaises(EmbeddingError):
            LsaEmbedder(dim=3).encode(["存货"])

    def test_vocabulary_smaller_than_dim_raises(self):
        with self.assertRaises(EmbeddingError):
            LsaEmbedder(dim=99).fit(FIT)

    def test_empty_corpus_raises(self):
        with self.assertRaises(EmbeddingError):
            LsaEmbedder(dim=2).fit([])

    def test_roundtrip_through_json_keeps_vectors(self):
        original = LsaEmbedder(dim=3).fit(FIT)
        restored = LsaEmbedder.loads(original.dumps())
        self.assertTrue(np.allclose(original.encode(["存货 库存"]), restored.encode(["存货 库存"])))
        self.assertEqual(restored.name, original.name)

    def test_serialising_unfitted_raises(self):
        with self.assertRaises(EmbeddingError):
            LsaEmbedder(dim=3).dumps()


class EmbeddingCacheTest(unittest.TestCase):
    def test_roundtrip(self):
        with temp_dir() as directory:
            cache = EmbeddingCache(directory)
            vectors = np.eye(3, dtype=np.float32)
            key = cache.key("lsa-3", "fingerprint", 3, 3)
            self.assertIsNone(cache.load(key, 3, 3))
            cache.save(key, vectors)
            self.assertTrue(np.allclose(cache.load(key, 3, 3), vectors))

    def test_shape_mismatch_is_a_miss(self):
        with temp_dir() as directory:
            cache = EmbeddingCache(directory)
            key = cache.key("lsa-3", "fingerprint", 3, 3)
            cache.save(key, np.eye(3, dtype=np.float32))
            self.assertIsNone(cache.load(key, 2, 3))

    def test_corrupt_file_is_a_miss(self):
        with temp_dir() as directory:
            cache = EmbeddingCache(directory)
            key = cache.key("lsa-3", "fingerprint", 3, 3)
            cache.path(key).write_bytes(b"not an npz file")
            self.assertIsNone(cache.load(key, 3, 3))

    def test_unwritable_directory_does_not_break_retrieval(self):
        with temp_dir() as directory:
            blocker = directory / "file"
            blocker.write_text("x", encoding="utf-8")
            cache = EmbeddingCache(blocker / "nested")
            cache.save("key", np.eye(2, dtype=np.float32))  # must not raise
            self.assertIsNone(cache.load("key", 2, 2))


class CountingEmbedder:
    name = "counting"
    dim = 2

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return unit_normalize(np.ones((len(texts), 2), dtype=np.float32))


class EmbedDocumentsTest(unittest.TestCase):
    def test_second_call_is_served_from_cache(self):
        with temp_dir() as directory:
            cache = EmbeddingCache(directory)
            embedder = CountingEmbedder()
            first = embed_documents(["a", "b"], embedder, cache=cache)
            second = embed_documents(["a", "b"], embedder, cache=cache)
            self.assertEqual(embedder.calls, 1)
            self.assertTrue(np.allclose(first, second))

    def test_dimension_mismatch_is_rejected(self):
        class Broken(CountingEmbedder):
            dim = 5

        with temp_dir() as directory:
            with self.assertRaises(EmbeddingError):
                embed_documents(["a"], Broken(), cache=EmbeddingCache(directory))

    def test_lsa_requires_fit_on(self):
        with temp_dir() as directory:
            with self.assertRaises(EmbeddingError):
                embed_documents(["a"], LsaEmbedder(dim=2), cache=EmbeddingCache(directory))

    def test_fingerprint_changes_with_content(self):
        self.assertNotEqual(corpus_fingerprint(["a"]), corpus_fingerprint(["b"]))
        self.assertEqual(corpus_fingerprint(["a"]), corpus_fingerprint(["a"]))


def _response(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class OpenAIEmbedderTest(unittest.TestCase):
    def test_uses_index_to_restore_order_and_normalises(self):
        def transport(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "m")
            self.assertEqual(body["input"], ["a", "b"])
            return _response({"data": [{"index": 1, "embedding": [0.0, 2.0]},
                                       {"index": 0, "embedding": [3.0, 4.0]}]})

        embedder = OpenAICompatibleEmbedder(model="m", dim=2, transport=transport)
        vectors = embedder.encode(["a", "b"])
        self.assertTrue(np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]], atol=1e-6))

    def test_batches_requests(self):
        sizes = []

        def transport(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            sizes.append(len(body["input"]))
            return _response({"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))]})

        embedder = OpenAICompatibleEmbedder(model="m", dim=2, batch_size=2, transport=transport)
        embedder.encode(["a", "b", "c"])
        self.assertEqual(sizes, [2, 1])

    def test_dimension_mismatch_is_reported(self):
        embedder = OpenAICompatibleEmbedder(model="m", dim=3, transport=lambda r, t: _response(
            {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}))
        with self.assertRaises(EmbeddingError):
            embedder.encode(["a"])

    def test_http_error_is_not_retried(self):
        calls = []

        def transport(request, timeout):
            calls.append(request)
            raise urllib.error.HTTPError("url", 401, "unauthorized", {}, io.BytesIO(b"nope"))

        embedder = OpenAICompatibleEmbedder(model="m", dim=2, retries=3, transport=transport)
        with self.assertRaises(EmbeddingError):
            embedder.encode(["a"])
        self.assertEqual(len(calls), 1)

    def test_transport_errors_are_retried_then_reported(self):
        calls = []

        def transport(request, timeout):
            calls.append(request)
            raise urllib.error.URLError("offline")

        embedder = OpenAICompatibleEmbedder(model="m", dim=2, retries=2, transport=transport)
        with self.assertRaises(EmbeddingError):
            embedder.encode(["a"])
        self.assertEqual(len(calls), 3)

    def test_invalid_json_is_reported(self):
        embedder = OpenAICompatibleEmbedder(model="m", dim=2, transport=lambda r, t: b"not json")
        with self.assertRaises(EmbeddingError):
            embedder.encode(["a"])

    def test_response_shape_mismatch_is_reported(self):
        embedder = OpenAICompatibleEmbedder(model="m", dim=2, transport=lambda r, t: _response({"data": []}))
        with self.assertRaises(EmbeddingError):
            embedder.encode(["a"])

    def test_empty_input_skips_the_endpoint(self):
        def transport(request, timeout):
            raise AssertionError("must not be called")

        embedder = OpenAICompatibleEmbedder(model="m", dim=2, transport=transport)
        self.assertEqual(embedder.encode([]).shape, (0, 2))

    def test_name_identifies_the_model(self):
        self.assertEqual(OpenAICompatibleEmbedder(model="bge-m3").name, "api:bge-m3")


if __name__ == "__main__":
    unittest.main()
