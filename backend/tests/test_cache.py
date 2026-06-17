#!/usr/bin/env python3
"""core/cache.py 분리 검증 테스트 — 패널(debate #2-R3b) 요구 4종.
결정론적·ollama·OCR 불필요. 직접 cache API를 호출해 스토리지 계약을 가드한다."""
import os, shutil, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.cache as C


class _CacheTestBase(unittest.TestCase):
    """setUp/tearDown: 임시 DB로 격리."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # infer cache 격리
        self._old_db = C._CACHE_DB
        self._old_conn = C._cache_conn
        self._old_enabled = C._CACHE_ENABLED
        C._CACHE_DB = os.path.join(self.tmp, "infer.db")
        C._cache_conn = None
        C.set_cache_enabled(True)
        # extract cache 격리
        self._old_edb = C._EXTRACT_CACHE_DB
        self._old_econn = C._extract_cache_conn
        C._EXTRACT_CACHE_DB = os.path.join(self.tmp, "extract.db")
        C._extract_cache_conn = None

    def tearDown(self):
        try:
            if C._cache_conn: C._cache_conn.close()
            if C._extract_cache_conn: C._extract_cache_conn.close()
        except Exception:
            pass
        C._CACHE_DB = self._old_db
        C._cache_conn = self._old_conn
        C.set_cache_enabled(self._old_enabled)
        C._EXTRACT_CACHE_DB = self._old_edb
        C._extract_cache_conn = self._old_econn
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestInferCacheKeys(_CacheTestBase):
    """1. 동일 입력 → 동일 키(결정론); 다른 입력 → 다른 키."""
    def test_identical_inputs_same_key(self):
        k1 = C.infer_cache_key("gemma4", b"img", "prompt")
        k2 = C.infer_cache_key("gemma4", b"img", "prompt")
        self.assertEqual(k1, k2)

    def test_different_inputs_different_keys(self):
        base = C.infer_cache_key("m", b"x", "p")
        self.assertNotEqual(base, C.infer_cache_key("m", b"y", "p"))
        self.assertNotEqual(base, C.infer_cache_key("m", b"x", "q"))
        self.assertNotEqual(base, C.infer_cache_key("other", b"x", "p"))


class TestInferCacheHitMiss(_CacheTestBase):
    """2. put→get 왕복; miss는 None."""
    def test_miss_returns_none(self):
        self.assertIsNone(C.infer_cache_get("no_such_key"))

    def test_put_get_roundtrip(self):
        k = C.infer_cache_key("m", b"x", "p")
        C.infer_cache_put(k, {"subject": "수학", "sub_subject": "수학Ⅰ", "confidence": 0.9})
        got = C.infer_cache_get(k)
        self.assertIsNotNone(got)
        self.assertEqual(got["subject"], "수학")
        self.assertAlmostEqual(got["confidence"], 0.9)


class TestExtractCacheEviction(_CacheTestBase):
    """3. byte-cap LRU 축출: MAX_BYTES=1이면 직전 항목이 즉시 비워진다."""
    def test_byte_cap_evicts_oldest(self):
        saved = C._EXTRACT_CACHE_MAX_BYTES
        C._EXTRACT_CACHE_MAX_BYTES = 1
        try:
            # 첫 번째 항목 — 방금 넣은 것은 보존 가드로 살아남음
            C.extract_problems_put("key_a", [{"text": "문제A", "image_bytes": b"A" * 100}])
            with C._extract_cache_lock:
                n = C._extract_cache_connection().execute(
                    "SELECT COUNT(*) FROM extract_cache").fetchone()[0]
            self.assertEqual(n, 1)
            # 두 번째 항목 — 첫 번째가 축출됨
            C.extract_problems_put("key_b", [{"text": "문제B", "image_bytes": b"B" * 100}])
            with C._extract_cache_lock:
                rows = C._extract_cache_connection().execute(
                    "SELECT key FROM extract_cache").fetchall()
            keys = {r[0] for r in rows}
            self.assertEqual(keys, {"key_b"})   # key_a 축출, key_b 유지
        finally:
            C._EXTRACT_CACHE_MAX_BYTES = saved


class TestCacheEnableDisable(_CacheTestBase):
    """4. set_cache_enabled(False) → 모든 ops가 noop; True 복원 후 동작."""
    def test_disabled_put_is_noop(self):
        C.set_cache_enabled(False)
        k = C.infer_cache_key("m", b"x", "p")
        C.infer_cache_put(k, {"subject": "영어"})
        self.assertIsNone(C.infer_cache_get(k))

    def test_disabled_extract_put_is_noop(self):
        C.set_cache_enabled(False)
        C.extract_problems_put("k", [{"text": "q"}])
        self.assertIsNone(C.extract_problems_get("k"))

    def test_reenable_restores_behavior(self):
        C.set_cache_enabled(False)
        C.set_cache_enabled(True)
        k = C.infer_cache_key("m", b"x", "p")
        C.infer_cache_put(k, {"subject": "국어"})
        got = C.infer_cache_get(k)
        self.assertIsNotNone(got)
        self.assertEqual(got["subject"], "국어")


if __name__ == "__main__":
    unittest.main(verbosity=2)
