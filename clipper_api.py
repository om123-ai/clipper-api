
#!/usr/bin/env python3
"""
clipper_service.py

Flask service that:
- accepts a YouTube URL and starts a background job to:
  1. download auto English subtitles (if available)
  2. ask OpenAI to pick up to 3 highlight segments (JSON)
  3. download video to a local file (mp4)
  4. cut clips with ffmpeg (ensures seeking works)
  5. expose job status and clip URLs

Requirements:
- yt-dlp on PATH
- ffmpeg on PATH
- Python packages: flask, openai (or OpenAI Python SDK), but this script assumes
  "from openai import OpenAI" (as in your original). If you use openai==0.x you
  might need to adapt the client usage.
- Set OPENAI_API_KEY in environment (or modify client initialization below)
"""

import os
import re
import json
import uuid
import shlex
import math
import tempfile
import traceback
import subprocess
from threading import Thread, Lock
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI  # keep same import pattern you used

# --- Configuration ---
app = Flask(__name__, static_folder="static")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

JOBS = {}
JOBS_LOCK = Lock()

# Output directories
BASE_OUTPUT = os.path.abspath("static")
CLIPS_DIR = os.path.join(BASE_OUTPUT, "clips")
VIDEOS_DIR = os.path.join(BASE_OUTPUT, "videos")
SUBS_DIR = os.path.join(BASE_OUTPUT, "subs")
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SUBS_DIR, exist_ok=True)


# --- Helpers ---
def sanitize_filename(name: str) -> str:
    """Removes characters that are invalid for file names."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def time_to_seconds(t):
    """
    Accepts time strings like "HH:MM:SS", "MM:SS", "SS", or numbers (int/float) and returns seconds (float).
    """
    if isinstance(t, (int, float)):
        return float(t)
    t = str(t).strip()
    # already a number (like "12.5")
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return float(t)
    parts = t.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 1:
        return parts[0]
    raise ValueError(f"Unrecognized time format: {t}")


def seconds_to_hhmmss(s: float):
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{sec:06.3f}"  # includes milliseconds


def run_cmd(cmd, cwd=None, check=True, capture_output=True, text=True):
    """Run subprocess command and return CompletedProcess. Raises on non-zero if check=True."""
    if isinstance(cmd, (list, tuple)):
        pass
    else:
        # allow string command
        cmd = shlex.split(cmd)
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=text)


def extract_video_id(url):
    """
    Try a few common ways to extract YouTube id; fallback to sanitized uuid if not found.
    """
    # common patterns
    m = re.search(r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    # as fallback, use uuid portion
    return str(uuid.uuid4())[:12]


def safe_update_job(job_id, **kwargs):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)


# --- Main job ---
def run_clipping_job(job_id, video_url):
    safe_update_job(job_id, status="processing", message="Starting job")
    try:
        video_id = extract_video_id(video_url)
        base_name = sanitize_filename(video_id)
        # 1) Download subtitles (auto English) using yt-dlp
        safe_update_job(job_id, message="Downloading subtitles (if available)...")
        # write subs to SUBS_DIR with output template
        sub_output_template = os.path.join(SUBS_DIR, "%(id)s.%(ext)s")
        try:
            # --write-auto-sub for auto-generated; --write-sub if uploaded subtitles exist
            run_cmd([
                "yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt/srv3/srt",
                "-o", sub_output_template,
                video_url
            ])
        except subprocess.CalledProcessError as e:
            # subtitles may not exist; continue but warn in job
            safe_update_job(job_id, message="No auto-subtitles found or yt-dlp error while fetching subtitles (proceeding).")
        # locate subtitle file
        found_sub = None
        for ext in ("en.vtt", "en.srt", "vtt", "srt"):
            candidate = os.path.join(SUBS_DIR, f"{video_id}.{ext}")
            if os.path.exists(candidate):
                found_sub = candidate
                break
        transcript_text = ""
        if found_sub:
            safe_update_job(job_id, message=f"Reading subtitles: {os.path.basename(found_sub)}")
            # Read raw subtitle content (simple)
            with open(found_sub, "r", encoding="utf-8", errors="ignore") as f:
                transcript_text = f.read()
        else:
            # Try to fetch plain transcript via yt-dlp's --get-description or fallback to using yt-dlp --dump-json
            safe_update_job(job_id, message="No subtitles found; attempting to extract description as fallback.")
            try:
                cp = run_cmd(["yt-dlp", "--dump-json", video_url])
                info = json.loads(cp.stdout.splitlines()[-1]) if cp.stdout else {}
                transcript_text = info.get("description", "") or ""
            except Exception:
                transcript_text = ""

        if not transcript_text.strip():
            safe_update_job(job_id, message="No transcript/subtitles found; AI will have limited input.")

        # 2) Ask OpenAI for highlight segments
        safe_update_job(job_id, message="Asking AI to identify highlight segments...")
        # Prompt: request JSON array of objects with start_time (e.g. "00:01:23" or seconds), end_time, reason
        ai_prompt = (
            "You are a video editor assistant. Based on the transcript/text below, identify up to 3 highlight moments.\n"
            "Return ONLY a JSON ARRAY of objects. Each object must contain:\n"
            ' - "start_time": start (format HH:MM:SS or seconds),\n'
            ' - "end_time": end (format HH:MM:SS or seconds),\n'
            ' - "reason": short explanation (max 140 chars).\n'
            "Clips should be between 20 and 75 seconds. If transcript is empty, you may return an empty array.\n\n"
            "Transcript/Text:\n\n"
            + transcript_text
        )
        # Safe API call; we attempt multiple response shapes
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful video editing assistant that responds in JSON format."},
                    {"role": "user", "content": ai_prompt}
                ],
                # don't rely on response_format (may or may not be supported)
                max_tokens=800
            )
            # extract content robustly
            raw_content = None
            if isinstance(response, dict):
                raw_content = response.get("choices", [{}])[0].get("message", {}).get("content")
            else:
                # object-like
                try:
                    raw_content = response.choices[0].message.content
                except Exception:
                    raw_content = str(response)
            if not raw_content:
                raise ValueError("Empty AI response")
        except Exception as e:
            # record and fail gracefully
            safe_update_job(job_id, status="error", message=f"OpenAI request failed: {e}")
            return

        # Try to pull out JSON from the response (in case there is additional text)
        json_text = None
        try:
            # First try if the whole content is JSON
            json_text = raw_content.strip()
            # locate first '[' and last ']' to extract array
            first = json_text.find("[")
            last = json_text.rfind("]")
            if first != -1 and last != -1:
                json_text = json_text[first:last+1]
            ai_segments = json.loads(json_text)
            if not isinstance(ai_segments, list):
                raise ValueError("AI did not return a JSON array")
        except Exception:
            # fallback: try to parse safely line by line or set empty
            try:
                # naive attempt: find a JSON-looking substring that is an array
                m = re.search(r"(\[.*\])", raw_content, flags=re.S)
                if m:
                    ai_segments = json.loads(m.group(1))
                else:
                    ai_segments = []
            except Exception:
                ai_segments = []

        if not ai_segments:
            safe_update_job(job_id, message="AI returned no segments (empty array). Marking job complete with no clips.", status="complete", clips=[])
            return

        # 3) Download the full video robustly (merged mp4) so ffmpeg can seek reliably
        safe_update_job(job_id, message="Downloading video (full) for reliable clipping...")
        video_output_template = os.path.join(VIDEOS_DIR, f"{base_name}.%(ext)s")
        try:
            # Best video+audio merged, force mp4 container
            run_cmd([
                "yt-dlp",
                "-f", "bestvideo+bestaudio/best",
                "--merge-output-format", "mp4",
                "-o", video_output_template,
                video_url
            ])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"yt-dlp failed to download video: {e.stderr or e}")

        # find downloaded file
        downloaded_file = None
        for ext in ("mp4", "mkv", "webm", "mpv"):
            candidate = os.path.join(VIDEOS_DIR, f"{base_name}.{ext}")
            if os.path.exists(candidate):
                downloaded_file = candidate
                break
        if not downloaded_file:
            # maybe yt-dlp used different name (id only) - try find most recent file in VIDEOS_DIR
            files = [os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR)]
            files = [f for f in files if os.path.isfile(f)]
            if files:
                downloaded_file = max(files, key=os.path.getmtime)
        if not downloaded_file:
            raise RuntimeError("Could not find downloaded video file after yt-dlp run.")

        # 4) Create clips
        final_clips = []
        for idx, seg in enumerate(ai_segments):
            safe_update_job(job_id, message=f"Processing AI segment {idx+1}/{len(ai_segments)}...")
            try:
                start_raw = seg.get("start_time") or seg.get("start") or seg.get("from")
                end_raw = seg.get("end_time") or seg.get("end") or seg.get("to")
                reason = (seg.get("reason") or seg.get("label") or "AI Highlight").strip()
                if start_raw is None or end_raw is None:
                    # skip invalid
                    continue
                start_sec = time_to_seconds(start_raw)
                end_sec = time_to_seconds(end_raw)
                if end_sec <= start_sec:
                    # skip invalid
                    continue
                duration = end_sec - start_sec

                # Enforce clip length bounds: between 20 and 75 seconds
                MIN_CLIP = 20.0
                MAX_CLIP = 75.0
                if duration < MIN_CLIP:
                    # try to expand end (if possible)
                    end_sec = start_sec + MIN_CLIP
                    duration = end_sec - start_sec
                if duration > MAX_CLIP:
                    end_sec = start_sec + MAX_CLIP
                    duration = MAX_CLIP

                # ensure not beyond file duration: try to probe duration with ffprobe
                try:
                    probe = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "default=noprint_wrappers=1:nokey=1", downloaded_file])
                    total_dur = float(probe.stdout.strip())
                    if start_sec >= total_dur:
                        # skip
                        continue
                    if end_sec > total_dur:
                        end_sec = total_dur
                        duration = end_sec - start_sec
                        if duration < MIN_CLIP:
                            # too short after trim -> skip
                            continue
                except Exception:
                    # if ffprobe fails, proceed anyway
                    pass

                start_ts = seconds_to_hhmmss(start_sec)
                # Use -ss before -i and -t <duration> to ensure accurate clip from local file
                clip_filename = f"{job_id}_clip_{idx+1}.mp4"
                clip_path = os.path.join(CLIPS_DIR, clip_filename)
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_sec:.3f}",  # seek in seconds
                    "-i", downloaded_file,
                    "-t", f"{duration:.3f}",
                    "-c", "copy",
                    clip_path
                ]
                run_cmd(ffmpeg_cmd)
                public_url = f"/clips/{clip_filename}"
                final_clips.append({"reason": reason, "url": public_url, "start_time": start_sec, "end_time": start_sec + duration})
            except Exception as e:
                # don't fail whole job for a single clip; log and continue
                safe_update_job(job_id, message=f"Error processing segment {idx+1}: {e}")

        # finalize
        safe_update_job(job_id, status="complete", message="Clipping complete", clips=final_clips)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error in job {job_id}: {e}\n{tb}")
        safe_update_job(job_id, status="error", message=str(e))
from flask import Flask, jsonify

app = Flask(__name__)

# --- ADD THIS CODE ---
@app.route("/api/health")
def health_check():
  """
  
  # Health Check Endpoint for Render
@app.route("/api/health")
def health_check():
    """Render.com health check endpoint"""
    return jsonify({"status": "healthy"}), 200


# --- API Endpoints ---
@app.route('/api/create-clips', methods=['POST'])
def create_clips_endpoint():
    payload = request.get_json(force=True, silent=True) or {}
    video_url = payload.get("url") or request.args.get("url")
    if not video_url:
        return jsonify({"error": "URL is required"}), 400
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {'status': 'pending', 'message': 'Job queued', 'clips': []}
    thread = Thread(target=run_clipping_job, args=(job_id, video_url), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id}), 202


@app.route('/api/check-status/<job_id>', methods=['GET'])
def check_status_endpoint(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# Serve static clips (simple)
@app.route('/clips/<path:filename>', methods=['GET'])
def serve_clip(filename):
    # security: only serve from CLIPS_DIR
    return send_from_directory(CLIPS_DIR, filename, as_attachment=False)


if __name__ == '__main__':
    # use 0.0.0.0 if you want external access
    app.run(debug=True, port=5000)

