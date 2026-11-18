"""The Ollama backend against a mock Ollama server (no model download needed)."""
import json, shutil, sys, tempfile, threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow import classify as clf, triage
from paperflow.config import Config
from paperflow.embed import Tfidf
from test_triage import build_db, write_examples

seen = []
class MockOllama(BaseHTTPRequestHandler):
    supports_think = False          # like a non-thinking model: rejects the "think" field
    def log_message(self, *a): pass
    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self.reply(200, {"models": [{"name": "qwen3:14b", "model": "qwen3:14b"}]})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append(req)
        if "think" in req and not self.supports_think:
            return self.reply(400, {"error": f'"{req["model"]}" does not support thinking'})
        user = req["messages"][1]["content"].lower()
        label = "mandatory" if ("nlr" in user or "effector" in user or "resistosome" in user) else \
                "interest" if "protein" in user else "skip"
        content = "<think>hmm</think>" + json.dumps({"label": label, "projects": [], "reason": "mock"})
        self.reply(200, {"message": {"role": "assistant", "content": content},
                         "prompt_eval_count": 900, "eval_count": 30, "done": True})

def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    # Missing model / unreachable server give actionable messages.
    for args, expect in (((url,), "ollama pull llama-nope"), (("http://127.0.0.1:9",), "Is it running")):
        try:
            clf.OllamaBackend("llama-nope" if expect.startswith("ollama") else "qwen3:14b", *args)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert expect in str(e), e
            print("ok:", str(e).splitlines()[0])

    b = clf.OllamaBackend("qwen3:14b", url, num_ctx=8192)
    text, usage = b.complete([{"type": "text", "text": "sys"}], "Title: An NLR paper")
    v = clf._parse_verdict(text)
    assert v["label"] == "mandatory" and usage == {"input": 900, "output": 30, "truncated": False}
    assert b.think_param is False                      # learned that this model rejects "think"
    req = seen[-1]
    assert req["format"]["properties"]["label"]["enum"] == ["mandatory", "interest", "skip"]
    assert req["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 400} and req["stream"] is False
    assert req["messages"][0]["role"] == "system"
    print("ok: structured output, num_ctx, think fallback, <think> stripped")

    try:
        clf.OllamaBackend("qwen3:14b", url, num_ctx=1024).check(prompt_chars=12000)
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "context window" in str(e); print("ok: oversize prompt caught before sending")

    # Full CLI run with provider=ollama.
    tmp = Path(tempfile.mkdtemp())
    store, papers = build_db(tmp / "db.sqlite")
    write_examples(tmp / "examples")
    triage.sync_examples(store, tmp / "examples")
    triage.run_filter(store, Tfidf(), progress=False)
    store.close()
    profile = tmp / "profile.md"; profile.write_text("# Research profile\n")
    cfg = Config(database=tmp / "db.sqlite", cache=tmp / "c.sqlite", profile=profile, examples_dir=tmp / "examples",
                 llm_provider="ollama", llm_model="qwen3:14b", ollama_url=url, llm_workers=1)
    from paperflow.triage_cli import cmd_classify
    cmd_classify(cfg, Namespace(dry_run=False, redo=False, limit=None, model=None))
    from paperflow.store import Store
    store = Store(tmp / "db.sqlite")
    rows = store.db.execute("SELECT label, detail FROM assessments WHERE stage='llm'").fetchall()
    assert rows and json.loads(rows[0][1])["model"] == "ollama:qwen3:14b"
    final = triage.final_labels(store)
    nlr_mand = sum(final[p]["label"] == "mandatory" for p in papers["nlr"])
    print(f"ok: CLI classify via Ollama: {len(rows)} verdicts, {nlr_mand}/40 NLR papers mandatory")
    store.close(); shutil.rmtree(tmp); srv.shutdown()
    print("OLLAMA CHECKS PASSED")

if __name__ == "__main__":
    main()
