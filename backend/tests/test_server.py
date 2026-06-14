#!/usr/bin/env python3
"""server 엔드포인트 회귀 테스트 (인프로세스 ASGI — uvicorn·ollama·포트 불필요).

휴먼리뷰 큐 노출(item 8)과 기본 라우팅을 가드한다. ollama 추론은 호출하지 않는다.

Run:  cd backend && python3 -m unittest tests.test_server -v
"""
import base64, io, os, sys, unittest, zipfile

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND)
sys.path.insert(0, os.path.join(_BACKEND, "api"))

try:
    import server
    from fastapi.testclient import TestClient
    _HAVE = True
except Exception:
    _HAVE = False

try:
    import fitz
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False


def _text_pdf_bytes(text):
    """텍스트 레이어가 있는 1페이지 PDF 바이트(≥30자 → transcribe가 OCR 없이 무손실 경로)."""
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    b = doc.tobytes()
    doc.close()
    return b


@unittest.skipUnless(_HAVE, "fastapi/httpx 미설치 — 엔드포인트 테스트 생략")
class TestNeedsReviewEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        self.tid = "test-needs-review"
        server.TASKS[self.tid] = {"status": "completed", "results": [
            {"page": 1, "problem_num": "1", "subject": "수학", "confidence": 0.9,
             "telemetry": {}},                                           # clean → 제외
            {"page": 1, "problem_num": "2", "subject": "미분류", "confidence": 0.0,
             "telemetry": {"needs_review": True, "review_reason": "model_unparseable"}},  # 격리 실패
            {"page": 1, "problem_num": "3", "subject": "과학탐구", "confidence": 0.3,
             "telemetry": {}},                                           # 저신뢰 → 포함
            {"page": 1, "problem_num": "4", "subject": "미분류", "confidence": 0.8,
             "telemetry": {}},                                           # 미분류 → 포함
        ]}

    def tearDown(self):
        server.TASKS.pop(self.tid, None)

    def test_flags_failed_low_conf_and_unclassified(self):
        r = self.client.get(f"/api/needs-review/{self.tid}")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["total"], 4)
        self.assertEqual(body["needs_review"], 3)         # clean 1건만 제외
        nums = sorted(it["problem_num"] for it in body["items"])
        self.assertEqual(nums, ["2", "3", "4"])

    def test_threshold_is_tunable(self):
        # max_confidence=0.0 이면 저신뢰 기준이 사라져 needs_review 플래그+미분류만 남는다
        r = self.client.get(f"/api/needs-review/{self.tid}?max_confidence=0.0")
        nums = sorted(it["problem_num"] for it in r.json()["items"])
        self.assertEqual(nums, ["2", "4"])                # #3(0.3 저신뢰)은 빠짐

    def test_unknown_task_404(self):
        self.assertEqual(self.client.get("/api/needs-review/nope").status_code, 404)


@unittest.skipUnless(_HAVE and _HAVE_FITZ, "fastapi/fitz 미설치")
class TestForceClassifyEscape(unittest.TestCase):
    """해설서 감지 오탐 시 탈출구: force_classify=true면 해설 분기를 건너뛰고 일반 분류
    경로로 간다(ollama 불통이라 격리 폴백으로 완료 — 분기 선택만 검증)."""
    def test_force_classify_skips_solution_branch(self):
        import time
        client = TestClient(server.app)
        pdf = _text_pdf_bytes("1. 다음 중 옳은 것은? 충분히 긴 본문 텍스트입니다. 보기와 선지.")
        captured = {}
        orig = server.save_solution_captures
        server.save_solution_captures = lambda problems, name, base_dir=None: captured.setdefault("called", True) or "/tmp/x"
        tid = None
        try:
            r = client.post("/api/classify",
                            files={"file": ("수학_정답지.pdf", pdf, "application/pdf")},
                            data={"force_classify": "true", "model": "없는모델", "concurrency": "1"})
            self.assertEqual(r.status_code, 200)
            tid = r.json()["task_id"]
            for _ in range(100):                     # 백그라운드 태스크 종료 대기
                st = client.get(f"/api/status/{tid}").json()
                if st["status"] in ("completed", "failed"):
                    break
                time.sleep(0.1)
            self.assertNotIn("called", captured)     # 해설 분기 미진입(분류 경로로 감)
        finally:
            server.save_solution_captures = orig
            if tid:
                server.TASKS.pop(tid, None); server.CAPTURES.pop(tid, None)


@unittest.skipUnless(_HAVE, "fastapi/httpx 미설치")
class TestBasicRouting(unittest.TestCase):
    def test_health_ok(self):
        c = TestClient(server.app)
        self.assertEqual(c.get("/health").json()["status"], "ok")

    def test_status_unknown_404(self):
        c = TestClient(server.app)
        self.assertEqual(c.get("/api/status/nope").status_code, 404)


@unittest.skipUnless(_HAVE and _HAVE_FITZ, "fastapi/httpx/fitz 미설치 — transcribe 테스트 생략")
class TestTranscribeEndpoint(unittest.TestCase):
    """POST /api/transcribe: 텍스트레이어 PDF → 평문 텍스트 + 유효한 .hwpx(base64). OCR/ollama 무관."""
    def test_transcribe_text_pdf(self):
        client = TestClient(server.app)
        pdf = _text_pdf_bytes("The quick brown fox jumps over the lazy dog repeatedly.")
        r = client.post("/api/transcribe", files={"file": ("exam.pdf", pdf, "application/pdf")})
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertGreaterEqual(j["pages"], 1)
        self.assertGreaterEqual(j["paragraphs"], 1)
        self.assertIn("quick brown fox", j["text"])
        self.assertEqual(j["hwpx_filename"], "exam.hwpx")
        self.assertIsInstance(j["pages_text"], list)                      # 페이지별 문단 구조
        self.assertTrue(all(isinstance(p, list) for p in j["pages_text"]))
        # hwpx_base64가 유효한 hwpx(zip, mimetype 첫 엔트리·무압축)인지
        raw = base64.b64decode(j["hwpx_base64"])
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(raw)))
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            self.assertEqual(z.infolist()[0].filename, "mimetype")
            self.assertEqual(z.infolist()[0].compress_type, zipfile.ZIP_STORED)
            self.assertEqual(z.read("mimetype").decode(), "application/hwp+zip")

    def test_bad_pdf_returns_400(self):
        # 손상/비-PDF 업로드 → 500 아닌 400(사용자 입력 오류)
        client = TestClient(server.app)
        r = client.post("/api/transcribe",
                        files={"file": ("bad.pdf", b"not a pdf at all", "application/pdf")})
        self.assertEqual(r.status_code, 400)


@unittest.skipUnless(_HAVE, "fastapi/httpx 미설치")
class TestReviewLogEndpoint(unittest.TestCase):
    """GET /api/review-log: list_recent 결과를 {total, items}로 노출(실제 DB는 건드리지 않게 패치)."""
    def test_review_log_shape(self):
        client = TestClient(server.app)
        orig = server.review_log.list_recent
        server.review_log.list_recent = lambda limit=100, include_resolved=False, source_pdf=None: [
            {"problem_num": "2", "subject": "미분류", "confidence": 0.0, "reason": "미분류"}]
        try:
            r = client.get("/api/review-log")
            self.assertEqual(r.status_code, 200)
            j = r.json()
            self.assertEqual(j["total"], 1)
            self.assertEqual(j["items"][0]["reason"], "미분류")
        finally:
            server.review_log.list_recent = orig

    def test_resolve_endpoint(self):
        client = TestClient(server.app)
        captured = {}
        orig = server.review_log.resolve

        def fake_resolve(source_pdf, page, problem_num, gold_subject=""):
            captured.update(source_pdf=source_pdf, page=page,
                            problem_num=problem_num, gold_subject=gold_subject)
            return 1
        server.review_log.resolve = fake_resolve
        try:
            r = client.post("/api/review-log/resolve",
                            data={"source_pdf": "exam.pdf", "page": "1",
                                  "problem_num": "2", "gold_subject": "수학"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["resolved"], 1)
            self.assertEqual(captured["source_pdf"], "exam.pdf")
            self.assertEqual(captured["gold_subject"], "수학")
        finally:
            server.review_log.resolve = orig

    def test_stats_endpoint(self):
        client = TestClient(server.app)
        orig = server.review_log.stats
        server.review_log.stats = lambda: {"total": 3, "pending": 2, "resolved": 1,
                                           "by_reason": {"미분류": 2}}
        try:
            r = client.get("/api/review-log/stats")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["pending"], 2)
            self.assertEqual(r.json()["by_reason"]["미분류"], 2)
        finally:
            server.review_log.stats = orig

    def test_clear_resolved_endpoint(self):
        client = TestClient(server.app)
        orig = server.review_log.clear_resolved
        server.review_log.clear_resolved = lambda: 5
        try:
            r = client.post("/api/review-log/clear-resolved")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["cleared"], 5)
        finally:
            server.review_log.clear_resolved = orig


class _FakeModel:
    def __init__(self, name): self.model = name


class _FakeResp:
    def __init__(self, names): self.models = [_FakeModel(n) for n in names]


def _fake_ollama(names=None, raises=False):
    class _FakeClient:
        def __init__(self, host=None): pass
        async def list(self):
            if raises:
                raise RuntimeError("ollama down")
            return _FakeResp(names or [])
    return _FakeClient


@unittest.skipUnless(_HAVE, "fastapi/httpx 미설치")
class TestListModels(unittest.TestCase):
    """GET /api/models — 설치된 비전 모델 노출(ollama는 몽키패치로 격리)."""
    def setUp(self):
        self.client = TestClient(server.app)
        self.orig = server.ollama.AsyncClient

    def tearDown(self):
        server.ollama.AsyncClient = self.orig

    def test_whitelist_narrows_when_it_can(self):
        # 화이트리스트가 맞으면 비전 변종만 좁혀 노출(텍스트 전용 llama3는 제외).
        server.ollama.AsyncClient = _fake_ollama(["gemma3:4b", "llama3:8b", "llava:13b"])
        r = self.client.get("/api/models")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(r.json()), ["gemma3:4b", "llava:13b"])

    def test_unknown_vision_models_not_dropped(self):
        # 핵심 회귀: 화이트리스트 밖이지만 설치된 비전 모델(moondream/cogvlm)을 떨구지 않는다.
        server.ollama.AsyncClient = _fake_ollama(["moondream", "cogvlm:19b"])
        r = self.client.get("/api/models")
        self.assertEqual(sorted(r.json()), ["cogvlm:19b", "moondream"])

    def test_ollama_down_returns_empty_not_fake(self):
        # ollama 불통 → 미설치일 수 있는 하드코딩 목록이 아니라 빈 목록.
        server.ollama.AsyncClient = _fake_ollama(raises=True)
        r = self.client.get("/api/models")
        self.assertEqual(r.json(), [])


@unittest.skipUnless(_HAVE, "fastapi/httpx 미설치")
class TestCaptureEviction(unittest.TestCase):
    """_evict_old_tasks: CAPTURES 총 바이트가 상한을 넘으면 오래된 완료 태스크의 캡처부터 비운다."""
    def test_evicts_captures_over_byte_budget(self):
        orig_budget = server.MAX_CAPTURE_BYTES
        server.MAX_CAPTURE_BYTES = 1000
        tids = ("ev_old", "ev_mid", "ev_new")
        try:
            for tid in tids:
                server.TASKS[tid] = {"status": "completed", "results": [{"page": 1}]}
                server.CAPTURES[tid] = [{"image_b64": "A" * 600}]  # 각 600B, 합 1800 > 1000
            server._evict_old_tasks()
            self.assertLessEqual(server._captures_bytes(), 1000)
            self.assertNotIn("ev_old", server.CAPTURES)   # 가장 오래된 캡처 제거
            self.assertIn("ev_new", server.CAPTURES)        # 최신 캡처 유지
            self.assertIn("ev_old", server.TASKS)           # 결과(TASKS)는 보존
        finally:
            server.MAX_CAPTURE_BYTES = orig_budget
            for tid in tids:
                server.TASKS.pop(tid, None)
                server.CAPTURES.pop(tid, None)


@unittest.skipUnless(_HAVE, "fastapi/httpx 미설치 — 엔드포인트 테스트 생략")
class TestCalibrationTrainEndpoint(unittest.TestCase):
    """POST /api/calibration/train — 트레이너 호출·결과 노출(실제 DB/sklearn은 몽키패치로 격리)."""
    def setUp(self):
        self.client = TestClient(server.app)
        self.orig = server.calibration_trainer.train_from_review_log

    def tearDown(self):
        server.calibration_trainer.train_from_review_log = self.orig

    def test_trained_returns_summary(self):
        server.calibration_trainer.train_from_review_log = \
            lambda db_path, out_path, min_samples: {"source": "gold", "n_samples": 12, "train_accuracy": 0.83}
        r = self.client.post("/api/calibration/train")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["trained"])
        self.assertEqual(body["n_samples"], 12)
        self.assertEqual(body["source"], "gold")

    def test_insufficient_data_reports_not_trained(self):
        server.calibration_trainer.train_from_review_log = lambda db_path, out_path, min_samples: None
        r = self.client.post("/api/calibration/train")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["trained"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
