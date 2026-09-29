# Local LinkedIn Sales Video Generator

Builds a 20–30 second sales video entirely offline: still images animated with a
Ken Burns effect, narrated with Piper TTS, assembled with FFmpeg, with burned-in
captions. No paid APIs, no cloud video generation.

Pipeline: **image → animation → voice → FFmpeg**

---

## 1. Requirements

- Python 3.10, 3.11, or 3.12
- FFmpeg (with `libass` support, for burned-in captions)
- Piper TTS (`pip install piper-tts`)
- A Piper voice model (`.onnx` + `.onnx.json`)
- *(Optional)* An NVIDIA GPU, if you want the script to generate scene images
  itself with Stable Diffusion, instead of supplying your own
- *(Optional)* `faster-whisper`, if you want word-accurate caption timing
  instead of the default script-based timing

---

## 2. Install (Windows)

1. **Python** — install from [python.org/downloads](https://python.org/downloads).
   Check **"Add python.exe to PATH"** during setup.

   > If `python --version` fails with a Microsoft Store message, disable the
   > stub aliases: **Settings → Apps → Advanced app settings → App execution
   > aliases** → turn off `python.exe` and `python3.exe`, then **restart your
   > PC**. Until then, call Python with its full path, e.g.:
   > ```
   > "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" make_video.py
   > ```

2. **FFmpeg** — download the "ffmpeg-release-essentials" build from
   [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/), extract to
   e.g. `C:\ffmpeg`, and add `C:\ffmpeg\bin` to your PATH (search
   "Environment Variables" in Windows search → edit `Path` under User
   variables). Confirm with:
   ```
   ffmpeg -version
   ```

3. **Piper TTS**:
   ```
   pip install piper-tts
   ```
   Confirm the `piper` command works, or that its `Scripts` folder
   (e.g. `...\Python312\Scripts`) is on PATH:
   ```
   where piper
   ```

4. **A Piper voice** — download from the
   [Piper voices page on Hugging Face](https://huggingface.co/rhasspy/piper-voices).
   Grab both the `.onnx` file and the matching `.onnx.json`, and put them in
   a `models\` folder next to `make_video.py`.

### macOS / Linux
Same steps, using your platform's package manager for FFmpeg
(`brew install ffmpeg` / `apt install ffmpeg`) and `pip install piper-tts`
for Piper. No Store-alias issue on these platforms.

---

## 3. Project folder layout

```
tts-video-genrate\
├── make_video.py
├── scenes.json
├── models\
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
├── images\              <- your scene images go here
│   ├── 01_hook.png
│   ├── 02_problem.png
│   ├── 03_solution.png
│   ├── 04_proof.png
│   └── 05_cta.png
├── audio\               <- generated automatically
├── clips\               <- generated automatically
└── out\                 <- final.mp4 ends up here
```

---

## 4. Configure your scenes

Edit `scenes.json`. Each scene needs an `id` (used as the image/audio
filename), a `narration` line, and an `image_prompt` (used only if you
generate images with `--generate`; otherwise it's just a reminder of what
picture to supply).

Video, voice, image-generation, and caption settings are all in the same
file — resolution, frame rate, font size, model paths, etc.

---

## 5. Add images

**Option A — supply your own (no GPU needed):**
Create any images you like with Canva, Bing Image Creator, ChatGPT/DALL·E,
Midjourney, your own photos, etc. Save them into `images\` with filenames
matching each scene's `id`:
```
images\01_hook.png
images\02_problem.png
...
```

**Option B — generate locally (needs an NVIDIA GPU):**
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
python make_video.py --generate
```
The first run downloads several GB of model weights.

---

## 6. Run it

```
python make_video.py
```

The finished video is written to `out\final.mp4`, with captions burned in
from `out\captions.srt`.

### Useful flags

| Flag | What it does |
|---|---|
| `--generate` | Generate any missing images locally with Stable Diffusion (needs NVIDIA GPU) |
| `--redo 02_problem` | Force-rebuild one scene's image + voice (comma-separate for several) |
| `--whisper` | Use `faster-whisper` for word-accurate caption timing (needs `pip install faster-whisper`) |
| `--no-captions` | Skip burned-in subtitles |
| `--config myfile.json` | Use a different scene config |

Voice is automatically re-rendered whenever you edit a scene's narration
text — no need to pass `--redo` just for a text change.

---

## 7. Optional: background music

Set `"music": "music.mp3"` under `"video"` in `scenes.json` (path relative to
the project folder). It loops under the narration at the volume set by
`"music_volume"` (default 0.10 = 10%).

---

## 8. Troubleshooting

- **"Python was not found" / Store popup** — see the alias fix in Section 2.
- **"Missing image images/xx.png"** — add the image or run with `--generate`.
- **Piper fails / `piper` not found** — confirm `pip show piper-tts`
  succeeded and that its `Scripts` folder is on PATH.
- **Captions don't appear** — your FFmpeg build needs `libass`; the
  gyan.dev "essentials" build includes it by default.
- **`--generate` fails or is very slow** — it needs a CUDA-capable NVIDIA
  GPU; on CPU-only or AMD/Intel machines, supply your own images instead.