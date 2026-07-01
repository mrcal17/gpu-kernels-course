"""Local web UI for the self-study drills. Pure stdlib, no installs.

    python drill/server.py            # starts on http://127.0.0.1:8177 and
                                      # opens the browser
    python drill/server.py --port 9000
    python drill/server.py --no-browser

Serves drill/app.html at / and a small JSON API under /api/. Localhost only:
binds 127.0.0.1 and rejects anything that is not a loopback client.

API surface (all JSON):
    GET  /api/overview        stats + exercises + predictions payload
    POST /api/quiz/start      {n, family|null, new_only} -> {session_id, questions}
                              (questions carry NO answers -- see /api/quiz/reveal)
    POST /api/quiz/reveal     {id} -> {a, why}
    POST /api/quiz/grade      {id, correct} -> {box, due, seen, correct}
                              (state saved atomically per answer, same as CLI)
    POST /api/redo            {ex} -> {backup_path, runner_cmd}
    POST /api/redo/restore    {ex} -> {restored_from}
    POST /api/predict/run     {ex, predicted, reasoning} -> {status, measured,
                              unit, pct_error, verdict, raw_output}
                              (runs the real benchmark synchronously -- takes
                              seconds; logged to predictions.csv unless [TODO])
"""
import argparse
import json
import sys
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drill as core  # noqa: E402

APP_HTML = Path(__file__).resolve().parent / "app.html"
DEFAULT_PORT = 8177

# In-memory quiz sessions: {session_id: [question ids]}. Purely informational
# (grading is per-question and persisted immediately), so losing these on
# restart costs nothing.
_sessions = {}
_sessions_lock = threading.Lock()

# Replaced by a fake when --mock-benchmark is passed (testing only).
_run_benchmark = core.run_benchmark


def _mock_run_benchmark(prefix, line_callback=None):
    folder = core.find_exercise(prefix)
    metric = core.read_metric(folder)
    unit = core.metric_unit(metric)
    val = 123.4 if unit == "GB/s" else 12.3
    output = (f"[mock] pretending to run harness.runner for {folder.name}\n"
              f"[PERF]   1.000 ms   {val} {unit}\n")
    return {"exercise": folder.name, "status": "ok", "output": output,
            "measured": val, "unit": unit}


class DrillHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------

    def _reject_non_local(self):
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            self._send_json(403, {"error": "localhost only"})
            return True
        return False

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Request body is not valid JSON")
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        return data

    def log_message(self, fmt, *args):  # quieter default log
        sys.stderr.write("[server] %s\n" % (fmt % args))

    # ---- GET ------------------------------------------------------------

    def do_GET(self):
        if self._reject_non_local():
            return
        path = self.path.split("?", 1)[0]
        try:
            if path == "/" or path == "/index.html":
                self._serve_app()
            elif path == "/api/overview":
                self._send_json(200, self._overview())
            else:
                self._send_json(404, {"error": f"Unknown path: {path}"})
        except core.ExerciseNotFound as e:
            self._send_json(404, {"error": str(e)})
        except core.DrillError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:  # keep the server alive
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _serve_app(self):
        if not APP_HTML.exists():
            self._send_json(500, {"error": "drill/app.html is missing"})
            return
        body = APP_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _overview(self):
        bank = core.load_bank()
        state = core.load_state()
        rows = core.load_predictions()
        quiz = core.quiz_stats(bank, state)
        quiz.update(core.due_counts(bank, state))
        return {
            "quiz": quiz,
            "families": sorted({q["family"] for q in bank}),
            "weak_families": core.WEAK_FAMILIES,
            "predictions": {"rows": rows, **core.prediction_summary(rows)},
            "exercises": core.list_exercises_with_status(),
            "using_sample_bank": core.using_sample_bank(),
        }

    # ---- POST -----------------------------------------------------------

    def do_POST(self):
        if self._reject_non_local():
            return
        path = self.path.split("?", 1)[0]
        routes = {
            "/api/quiz/start": self._quiz_start,
            "/api/quiz/reveal": self._quiz_reveal,
            "/api/quiz/grade": self._quiz_grade,
            "/api/redo": self._redo,
            "/api/redo/restore": self._redo_restore,
            "/api/predict/run": self._predict_run,
        }
        handler = routes.get(path)
        try:
            if handler is None:
                self._send_json(404, {"error": f"Unknown path: {path}"})
                return
            handler(self._read_json())
        except core.ExerciseNotFound as e:
            self._send_json(404, {"error": str(e)})
        except core.DrillError as e:
            self._send_json(400, {"error": str(e)})
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:  # keep the server alive
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _quiz_start(self, body):
        n = body.get("n", 12)
        family = body.get("family") or None
        new_only = bool(body.get("new_only"))
        if not isinstance(n, int) or n < 1:
            raise ValueError("'n' must be a positive integer")
        bank = core.load_bank()
        state = core.load_state()
        session = core.select_due_questions(bank, state, n, family, new_only)
        sid = uuid.uuid4().hex
        with _sessions_lock:
            _sessions[sid] = [q["id"] for q in session]
        # Answers deliberately withheld -- the learner commits via /reveal.
        questions = [{"id": q["id"], "family": q["family"], "type": q["type"],
                      "q": q["q"], "source": q["source"]} for q in session]
        self._send_json(200, {"session_id": sid, "questions": questions})

    def _find_question(self, qid):
        for q in core.load_bank():
            if q["id"] == qid:
                return q
        return None

    def _quiz_reveal(self, body):
        qid = body.get("id")
        q = self._find_question(qid)
        if q is None:
            self._send_json(404, {"error": f"Unknown question id: {qid}"})
            return
        self._send_json(200, {"a": q["a"], "why": q["why"]})

    def _quiz_grade(self, body):
        qid = body.get("id")
        if "correct" not in body:
            raise ValueError("'correct' (true/false) is required")
        if self._find_question(qid) is None:
            self._send_json(404, {"error": f"Unknown question id: {qid}"})
            return
        state = core.load_state()
        entry = core.grade_question(state, qid, bool(body["correct"]))
        self._send_json(200, entry)

    def _redo(self, body):
        ex = body.get("ex")
        if not ex:
            raise ValueError("'ex' is required")
        info = core.redo_backup(ex)  # verifies git show before touching anything
        self._send_json(200, {"exercise": info["exercise"],
                              "backup_path": info["backup"],
                              "runner_cmd": info["runner_cmd"]})

    def _redo_restore(self, body):
        ex = body.get("ex")
        if not ex:
            raise ValueError("'ex' is required")
        info = core.redo_restore(ex)
        self._send_json(200, {"exercise": info["exercise"],
                              "restored_from": info["restored_from"]})

    def _predict_run(self, body):
        ex = body.get("ex")
        if not ex:
            raise ValueError("'ex' is required")
        try:
            predicted = float(body.get("predicted"))
        except (TypeError, ValueError):
            raise ValueError("'predicted' must be a number")
        reasoning = str(body.get("reasoning") or "").strip()

        folder = core.find_exercise(ex)
        metric = core.read_metric(folder)
        if metric == "none":
            raise core.DrillError(
                f"{folder.name} has METRIC = 'none' -- nothing to predict.")
        unit = core.metric_unit(metric)

        # Real benchmark, run synchronously; these take seconds.
        result = _run_benchmark(ex)

        if result["status"] == "todo":
            self._send_json(200, {"status": "todo",
                                  "raw_output": result["output"],
                                  "message": "Exercise is unimplemented -- "
                                             "nothing logged."})
            return
        if result["status"] == "no_perf":
            self._send_json(200, {"status": "no_perf",
                                  "raw_output": result["output"],
                                  "message": "No [PERF] line in the output "
                                             "(failed run?) -- nothing logged."})
            return

        measured, meas_unit = result["measured"], result["unit"]
        if meas_unit != unit:
            unit = meas_unit  # trust the runner, same as the CLI
        pct_error = abs(predicted - measured) / measured * 100
        core.log_prediction(folder.name, unit, predicted, measured,
                            pct_error, reasoning)
        self._send_json(200, {"status": "ok",
                              "exercise": folder.name,
                              "predicted": predicted,
                              "measured": measured,
                              "unit": unit,
                              "pct_error": round(pct_error, 1),
                              "verdict": core.verdict(pct_error),
                              "raw_output": result["output"]})


def main():
    global _run_benchmark
    ap = argparse.ArgumentParser(description="Web UI for the drill tool")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true",
                    help="don't open a browser tab on startup")
    ap.add_argument("--mock-benchmark", action="store_true",
                    help=argparse.SUPPRESS)  # testing only: fake predict runs
    args = ap.parse_args()

    if args.mock_benchmark:
        _run_benchmark = _mock_run_benchmark
        print("[server] MOCK benchmark mode -- predict runs are FAKE")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DrillHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[server] drill web UI on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        # Server socket is already listening (bound in the constructor).
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
