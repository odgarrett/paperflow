"""Text to speech with Kokoro (https://github.com/hexgrad/kokoro), a small, high-quality local
voice model. Install with `pip install -e ".[tts]"`, plus the system packages espeak-ng (for
words outside its dictionary) and ffmpeg (for MP3s):  sudo apt install espeak-ng ffmpeg

A script is a list of segments: ("text", paragraph), ("pause", seconds) or ("chapter", title).
Each paragraph is voiced separately and written straight to a WAV file, then the whole briefing
is encoded as one MP3 (or M4B) with a chapter per paper (WAV plus a chapter list without ffmpeg).
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
VOICES_HINT = ("American: af_heart, af_bella, af_nicole, am_michael, am_fenrir; "
               "British: bf_emma, bf_isabella, bm_george, bm_fable")


class Kokoro:
    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        try:
            from kokoro import KPipeline
        except ImportError:
            raise SystemExit('Audio needs Kokoro: pip install -e ".[tts]" (and: sudo apt install espeak-ng ffmpeg). '
                             "Or run with --no-audio to make the EPUB and scripts only.")
        lang = voice[:1] if voice[:1] in ("a", "b") else "a"   # a = American, b = British English
        self.pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
        self.voice, self.speed = voice, speed

    def say(self, text: str) -> np.ndarray:
        pieces = []
        for result in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=None):
            audio = result.audio
            if audio is not None:
                pieces.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def render_to_wav(tts, script: list[tuple[str, object]], wav_path: Path, progress=None) -> list[tuple[str, float]]:
    """Voice a script into a WAV file, writing as it goes (hours of audio never sit in memory).

    Script segments: ("text", paragraph), ("pause", seconds), ("chapter", title).
    Returns the chapters as (title, start in seconds)."""
    import soundfile as sf
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    chapters, written = [], 0
    with sf.SoundFile(str(wav_path), "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as f:
        for kind, value in script:
            if kind == "chapter":
                chapters.append((str(value), written / SAMPLE_RATE))
                if progress:
                    progress()
            elif kind == "pause":
                n = int(SAMPLE_RATE * float(value))
                f.write(np.zeros(n, dtype=np.float32))
                written += n
            elif str(value).strip():
                audio = tts.say(str(value)).astype(np.float32)
                f.write(audio)
                written += len(audio)
    return chapters


def encode(wav_path: Path, out_base: Path, fmt: str, title: str, album: str,
           chapters: list[tuple[str, float]]) -> Path:
    """Encode to MP3 or M4B with a chapter per entry; keeps the WAV if ffmpeg isn't installed."""
    import soundfile as sf
    duration = sf.info(str(wav_path)).duration
    if not shutil.which("ffmpeg"):
        out = out_base.with_suffix(".wav")
        wav_path.replace(out)
        lines = [f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}  {name}" for name, t in chapters]
        out.with_suffix(".chapters.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def esc(v: str) -> str:
        return re.sub(r"([=;#\\\n])", r"\\\1", v)

    meta = [";FFMETADATA1", f"title={esc(title)}", f"album={esc(album)}", "artist=paperflow"]
    for i, (name, start) in enumerate(chapters):
        end = chapters[i + 1][1] if i + 1 < len(chapters) else duration
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}", f"END={int(end * 1000)}",
                 f"title={esc(name)}"]
    meta_path = wav_path.with_suffix(".ffmeta")
    meta_path.write_text("\n".join(meta) + "\n", encoding="utf-8")
    if fmt == "m4b":
        out, codec = out_base.with_suffix(".m4b"), ["-c:a", "aac", "-b:a", "64k", "-f", "mp4"]
    else:
        out, codec = out_base.with_suffix(".mp3"), ["-codec:a", "libmp3lame", "-q:a", "4", "-id3v2_version", "3"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-i", str(meta_path),
                    "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1", *codec, str(out)], check=True)
    meta_path.unlink(missing_ok=True)
    wav_path.unlink(missing_ok=True)
    return out


def script_to_text(script: list[tuple[str, object]]) -> str:
    """The script as plain text (for checking it, or feeding another TTS tool)."""
    out = []
    for kind, value in script:
        if kind == "chapter":
            out.append(f"[{value}]\n\n")
        elif kind == "text":
            out.append(str(value) + "\n\n")
        elif float(value) >= 0.9:
            out.append("\n")
    return re.sub(r"\n{4,}", "\n\n\n", "".join(out)).strip() + "\n"
