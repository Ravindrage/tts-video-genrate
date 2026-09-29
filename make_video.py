#!/usr/bin/env python3
"""
Local sales-video pipeline: images -> Piper voice -> FFmpeg Ken Burns -> captions.

Usage:
  python make_video.py                    # use images in images/<scene_id>.png
  python make_video.py --generate         # generate missing images with diffusers
  python make_video.py --redo 02_problem  # force-rebuild specific scene(s) (images + voice)
  python make_video.py --whisper          # word-accurate captions via faster-whisper
  python make_video.py --no-captions      # skip burned-in subtitles

Requires: ffmpeg + ffprobe on PATH, Piper (pip install piper-tts) and a voice model.
Optional: diffusers/torch (--generate), faster-whisper (--whisper).
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
IMG_DIR, AUDIO_DIR, CLIP_DIR, OUT_DIR = (ROOT / d for d in ("images", "audio", "clips", "out"))

# Ken Burns motions. N = frame count, on = current frame index.
MOTIONS = [
    ("1+0.15*on/{N}",    "iw/2-(iw/zoom/2)",              "ih/2-(ih/zoom/2)"),   # zoom in
    ("1.15-0.15*on/{N}", "iw/2-(iw/zoom/2)",              "ih/2-(ih/zoom/2)"),   # zoom out
    ("1.12",             "(iw-iw/zoom)*on/{N}",           "ih/2-(ih/zoom/2)"),   # pan right
    ("1.12",             "(iw-iw/zoom)*(1-on/{N})",       "ih/2-(ih/zoom/2)"),   # pan left
]


def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r


def duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path])
    return float(r.stdout.strip())


def sha(text):
    return hashlib.sha1(text.encode()).hexdigest()[:12]


# ---------- images ----------
_pipe = None

def generate_image(scene, cfg, out_path):
    global _pipe
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        sys.exit("--generate needs: pip install diffusers transformers accelerate torch")
    icfg = cfg["images"]
    if _pipe is None:
        print(f"  loading {icfg['generate_model']} ...")
        _pipe = AutoPipelineForText2Image.from_pretrained(
            icfg["generate_model"], torch_dtype=torch.float16
        ).to("cuda")
    prompt = f"{scene['image_prompt']}, {icfg['style_suffix']}"
    img = _pipe(prompt=prompt, width=icfg["gen_width"], height=icfg["gen_height"],
                num_inference_steps=icfg["steps"], guidance_scale=icfg["guidance"]).images[0]
    img.save(out_path)


def ensure_image(scene, cfg, generate, redo):
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = IMG_DIR / f"{scene['id']}.{ext}"
        if p.exists() and not (redo and generate):
            return p
    if not generate:
        sys.exit(f"Missing image images/{scene['id']}.png\n"
                 f"  prompt: {scene['image_prompt']}\n"
                 f"Add it, or run with --generate.")
    out = IMG_DIR / f"{scene['id']}.png"
    print(f"[img] generating {out.name}")
    generate_image(scene, cfg, out)
    return out


# ---------- voice ----------
def ensure_audio(scene, cfg, redo):
    wav = AUDIO_DIR / f"{scene['id']}.wav"
    stamp = AUDIO_DIR / f"{scene['id']}.hash"
    h = sha(scene["narration"] + cfg["voice"]["model"])
    if wav.exists() and stamp.exists() and stamp.read_text() == h and not redo:
        return wav
    print(f"[tts] {scene['id']}")
    v = cfg["voice"]
    cmd = [c for c in str(v["piper_cmd"]).split()] + ["--model", str(ROOT / v["model"]),
                                                     "--output_file", str(wav)]
    r = subprocess.run(cmd, input=scene["narration"], text=True, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"Piper failed: {r.stderr[-1500:]}")
    stamp.write_text(h)
    return wav


# ---------- clip ----------
def build_clip(idx, scene, img, wav, cfg):
    W, H, fps = cfg["video"]["width"], cfg["video"]["height"], cfg["video"]["fps"]
    dur = duration(wav) + cfg["video"]["tail_pad"]
    N = max(2, math.ceil(dur * fps))
    z, x, y = (m.format(N=N) for m in MOTIONS[idx % len(MOTIONS)])
    vf = (f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,crop={W*2}:{H*2},"
          f"zoompan=z='{z}':x='{x}':y='{y}':d={N}:s={W}x{H}:fps={fps},"
          f"format=yuv420p")
    out = CLIP_DIR / f"{scene['id']}.mp4"
    print(f"[clip] {scene['id']} ({dur:.1f}s)")
    run(["ffmpeg", "-y", "-i", img, "-i", wav,
         "-filter_complex",
         f"[0:v]{vf}[v];[1:a]apad,aresample=44100,aformat=channel_layouts=stereo[a]",
         "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", fps,
         "-c:a", "aac", "-b:a", "192k", out])
    return out, dur


# ---------- captions ----------
def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def chunk_words(words, max_words):
    """Group words into short caption chunks, breaking early at punctuation."""
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?,;:]$", w):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def script_captions(scenes, speech_durs, starts, max_words):
    """Caption timing from script text, spread by character count within each scene's speech."""
    entries = []
    for scene, sdur, start in zip(scenes, speech_durs, starts):
        chunks = chunk_words(scene["narration"].split(), max_words)
        weights = [len(" ".join(c)) for c in chunks]
        total, t = sum(weights), start
        for c, wgt in zip(chunks, weights):
            d = sdur * wgt / total
            entries.append((t, t + d, " ".join(c)))
            t += d
    return entries


def whisper_captions(audio_path, max_words):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("--whisper needs: pip install faster-whisper")
    model = WhisperModel("small", compute_type="int8")
    segs, _ = model.transcribe(str(audio_path), word_timestamps=True)
    words = [w for s in segs for w in s.words]
    entries, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?,;:]$", w.word.strip()):
            entries.append((cur[0].start, cur[-1].end, "".join(x.word for x in cur).strip()))
            cur = []
    if cur:
        entries.append((cur[0].start, cur[-1].end, "".join(x.word for x in cur).strip()))
    return entries


def write_srt(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(entries, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")


# ---------- assembly ----------
def concat(clips, out):
    lst = OUT_DIR / "concat.txt"
    lst.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])


def add_music(video, music, vol, out):
    run(["ffmpeg", "-y", "-i", video, "-stream_loop", "-1", "-i", ROOT / music,
         "-filter_complex",
         f"[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out])


def burn_captions(video, srt, out, font_size):
    # Run inside OUT_DIR with relative names to dodge path-escaping problems in the filter.
    style = (f"FontName=Arial,FontSize={font_size},Bold=1,Alignment=2,MarginV=90,"
             f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0")
    run(["ffmpeg", "-y", "-i", Path(video).name,
         "-vf", f"subtitles={Path(srt).name}:force_style='{style}'",
         "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-c:a", "copy", Path(out).name],
        cwd=OUT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "scenes.json"))
    ap.add_argument("--generate", action="store_true", help="generate missing images with diffusers")
    ap.add_argument("--redo", default="", help="comma-separated scene ids to rebuild from scratch")
    ap.add_argument("--whisper", action="store_true", help="word-level captions with faster-whisper")
    ap.add_argument("--no-captions", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    for d in (IMG_DIR, AUDIO_DIR, CLIP_DIR, OUT_DIR):
        d.mkdir(exist_ok=True)
    redo_ids = {s.strip() for s in args.redo.split(",") if s.strip()}

    scenes = cfg["scenes"]
    clips, speech_durs, starts, t = [], [], [], 0.0
    for i, scene in enumerate(scenes):
        redo = scene["id"] in redo_ids
        img = ensure_image(scene, cfg, args.generate, redo)
        wav = ensure_audio(scene, cfg, redo)
        clip, dur = build_clip(i, scene, img, wav, cfg)
        clips.append(clip)
        speech_durs.append(duration(wav))
        starts.append(t)
        t += dur

    base = OUT_DIR / "video_nocaptions.mp4"
    concat(clips, base)

    if cfg["video"].get("music"):
        with_music = OUT_DIR / "video_music.mp4"
        add_music(base, cfg["video"]["music"], cfg["video"]["music_volume"], with_music)
        base = with_music

    final = OUT_DIR / "final.mp4"
    if args.no_captions:
        base.replace(final)
    else:
        maxw = cfg["captions"]["max_words"]
        if args.whisper:
            print("[captions] whisper")
            entries = whisper_captions(base, maxw)
        else:
            entries = script_captions(scenes, speech_durs, starts, maxw)
        srt = OUT_DIR / "captions.srt"
        write_srt(entries, srt)
        print("[captions] burning in")
        burn_captions(base, srt, final, cfg["captions"]["font_size"])

    print(f"\nDone: {final}  ({duration(final):.1f}s)")


if __name__ == "__main__":
    main()
