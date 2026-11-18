"""Ollama's model process crashing (out of memory): retried once it restarts; if it stays
down, `brief` stops cleanly with advice instead of failing every paper."""
import json, shutil, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperflow import classify
from paperflow.briefing import chunk_chars_for, MAX_PIECE_CHARS
from paperflow.config import Config
from paperflow.store import Store
import test_brief as tb

class Ollama(BaseHTTPRequestHandler):
    mode = "ok"            # ok | crash_once | down
    chats = 0
    def log_message(self, *a): pass
    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply(200, {"models": [{"name": "qwen3:14b"}]})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "messages" not in req:
            return self.reply(200, {})                              # unload request
        Ollama.chats += 1
        if Ollama.mode == "down" or (Ollama.mode == "crash_once" and Ollama.chats == 1):
            return self.reply(500, {"error": "model runner has unexpectedly stopped, this may be due to "
                                             "resource limitations or an internal error"})
        if req.get("format"):
            content = json.dumps({"question": "Q?", "systems": ["Pik"], "approach": [], "results": ["r"],
                                  "limitations": [], "key_figures": []})
        else:
            content = "A short write-up about the paper."
        self.reply(200, {"message": {"content": content}, "prompt_eval_count": 10, "eval_count": 5,
                         "done_reason": "stop"})

def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    classify.OllamaBackend.retry_wait = 0

    b = classify.OllamaBackend("qwen3:14b", url, num_ctx=16384)
    Ollama.mode, Ollama.chats = "crash_once", 0
    text, _ = b.complete([{"type": "text", "text": "s"}], "u", schema=None)
    assert text and Ollama.chats == 2
    print("ok: a crash is retried once Ollama restarts the model")
    assert chunk_chars_for(b) == MAX_PIECE_CHARS and chunk_chars_for(classify.OllamaBackend("qwen3:14b", url, 4096)) < MAX_PIECE_CHARS
    print("ok: pieces capped at", MAX_PIECE_CHARS, "characters")

    tmp = Path(tempfile.mkdtemp())
    tb.make_db(tmp); (tmp / "profile.md").write_text(tb.PROFILE)
    cfg = Config(database=tmp / "data" / "db.sqlite", cache=tmp / "c.sqlite", profile=tmp / "profile.md",
                 briefings_dir=tmp / "b", pronunciations=tmp / "p.txt", ollama_url=url, ollama_num_ctx=16384)
    Ollama.mode, Ollama.chats = "down", 0
    out = tb.run(cfg, classify.OllamaBackend("qwen3:14b", url, 16384), {}, no_audio=True)
    print("   " + out.strip().splitlines()[-2][:150])
    assert "Stopped:" in out and "ollama_num_ctx" in out and "Every paper failed" not in out
    assert Ollama.chats == 3                                 # gave up after the retries, not 6 papers × 3
    assert not (tmp / "b").exists() or not any((tmp / "b").iterdir())

    Ollama.mode = "ok"                                       # back up: the same command just continues
    out = tb.run(cfg, classify.OllamaBackend("qwen3:14b", url, 16384), {}, no_audio=True)
    assert "Briefing 'mandatory-" in out and "6 mandatory" in out, out
    srv.shutdown(); shutil.rmtree(tmp)
    print("ok: stops cleanly when Ollama stays down; rerun continues once it's back")
    print("CRASH CHECKS PASSED")

if __name__ == "__main__":
    main()
