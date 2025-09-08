# streamlit_app.py
# --------------------------------------------------
# Streamlit UI wrapper for live_cam.AudioLoop
# Features:
# - Start / Pause / Resume / Stop buttons
# - Mode selector: camera / screen / none
# - Per-user token-bucket rate limiting (configurable)
# - Live logs panel
# - Safe background thread that runs asyncio loop
# --------------------------------------------------

import os
import time
import threading
import queue
import asyncio
from dataclasses import dataclass

import streamlit as st

# Import your backend
from live_cam import AudioLoop, DEFAULT_MODE

# ------------- Config (adjust as needed) ----------------

DEFAULT_HOURLY_TOKENS = 600     # total media packets/hour (audio chunks + frames)
DEFAULT_FPS_BUDGET = 60         # optional budget for images/hour (subset of tokens)

# ------------- Utilities -------------------------------

@dataclass
class TokenBucket:
    capacity: int
    refill_rate_per_sec: float   # tokens per second
    tokens: float
    last_refill: float

    def try_consume(self, n: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        # Refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate_per_sec)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


def make_token_bucket(capacity: int, per_seconds: int) -> TokenBucket:
    return TokenBucket(
        capacity=capacity,
        refill_rate_per_sec=capacity / per_seconds,
        tokens=capacity,
        last_refill=time.time(),
    )


# ------------- Session State Init ----------------------

def ensure_state():
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "mode" not in st.session_state:
        st.session_state.mode = DEFAULT_MODE
    if "running" not in st.session_state:
        st.session_state.running = False
    if "paused" not in st.session_state:
        st.session_state.paused = False
    if "log_queue" not in st.session_state:
        st.session_state.log_queue = queue.Queue()
    if "pause_event" not in st.session_state:
        st.session_state.pause_event = threading.Event()
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if "loop_thread" not in st.session_state:
        st.session_state.loop_thread = None
    if "loop_instance" not in st.session_state:
        st.session_state.loop_instance = None
    if "media_bucket" not in st.session_state:
        # Default: 600 packets/hour ~ (10 packets/minute) conservative for cloud
        st.session_state.media_bucket = make_token_bucket(DEFAULT_HOURLY_TOKENS, per_seconds=3600)
    if "image_bucket" not in st.session_state:
        st.session_state.image_bucket = make_token_bucket(DEFAULT_FPS_BUDGET, per_seconds=3600)


# ------------- Logging hook ----------------------------

def ui_logger(msg: str):
    # Push logs from background thread into a queue; the UI will render them
    try:
        st.session_state.log_queue.put_nowait(msg)
    except Exception:
        pass


# ------------- Send guard (rate limiter) ---------------

def send_guard(media: dict) -> bool:
    """
    Decide whether to send this media packet to the model.
    media: {"data": ..., "mime_type": "..."}
    Returns False to drop (rate limit).
    """
    mime = media.get("mime_type", "")
    # 1 token per packet
    if not st.session_state.media_bucket.try_consume(1):
        ui_logger("⛔ Rate limit reached: media budget exhausted.")
        return False

    # For image frames, also check image bucket
    if mime.startswith("image/"):
        if not st.session_state.image_bucket.try_consume(1):
            ui_logger("⛔ Rate limit reached: image frame budget exhausted.")
            return False

    return True


# ------------- Background runner -----------------------

def run_asyncio_loop(mode: str):
    """
    Runs the AudioLoop inside asyncio in a dedicated thread.
    Streamlit re-runs the script per interaction, so we keep references in session_state.
    """
    loop_inst = AudioLoop(
        video_mode=mode,
        pause_event=st.session_state.pause_event,
        stop_event=st.session_state.stop_event,
        send_guard=send_guard,
        logger=ui_logger,
    )
    st.session_state.loop_instance = loop_inst
    try:
        asyncio.run(loop_inst.run(enable_cli_input=False))
    except Exception as e:
        ui_logger(f"[runner] crashed: {e}")
    finally:
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.stop_event.set()
        st.session_state.loop_thread = None
        ui_logger("✅ Session stopped.")


# ------------- UI --------------------------------------

def main():
    ensure_state()
    st.set_page_config(page_title="Gemini Live Assistance", page_icon="🎥", layout="wide")

    st.title("🎥 Gemini Live Assistance")
    st.caption("Stream mic + (camera/screen) to Gemini 2.0 Flash Live and get real-time audio replies.")

    with st.sidebar:
        st.header("User & Limits")
        st.session_state.username = st.text_input("User ID / Name", st.session_state.username, placeholder="e.g., krishna")
        st.session_state.mode = st.selectbox("Video Mode", ["camera", "screen", "none"], index=["camera", "screen", "none"].index(st.session_state.mode))

        st.subheader("Rate Limits (per hour)")
        total_tokens = st.number_input("Total media packets", min_value=60, max_value=5000, value=DEFAULT_HOURLY_TOKENS, step=60)
        image_tokens = st.number_input("Image frames", min_value=0, max_value=1000, value=DEFAULT_FPS_BUDGET, step=10)

        cols = st.columns(2)
        if cols[0].button("Apply Limits"):
            st.session_state.media_bucket = make_token_bucket(int(total_tokens), 3600)
            st.session_state.image_bucket = make_token_bucket(int(image_tokens), 3600)
            st.toast("Limits applied.", icon="✅")

        st.divider()
        st.info("Set your `GEMINI_API_KEY` in Streamlit Cloud → App → **Settings → Secrets**.", icon="🔑")

    # Controls
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    if c1.button("▶️ Start", disabled=st.session_state.running):
        if not st.session_state.username.strip():
            st.warning("Please enter a User ID / Name in the sidebar.")
        else:
            # Reset flags and tokens on start
            st.session_state.stop_event.clear()
            st.session_state.pause_event.clear()
            st.session_state.paused = False
            st.session_state.running = True
            # Re-seed token buckets (fresh hour)
            st.session_state.media_bucket = make_token_bucket(
                int(st.session_state.media_bucket.capacity), 3600
            )
            st.session_state.image_bucket = make_token_bucket(
                int(st.session_state.image_bucket.capacity), 3600
            )
            # Launch background thread
            t = threading.Thread(target=run_asyncio_loop, args=(st.session_state.mode,), daemon=True)
            st.session_state.loop_thread = t
            t.start()
            st.success("Started session.")

    if c2.button("⏸️ Pause", disabled=(not st.session_state.running) or st.session_state.paused):
        st.session_state.pause_event.set()
        st.session_state.paused = True
        st.info("Paused. Video/screen frames are halted; audio capture may continue.")

    if c3.button("⏯️ Resume", disabled=(not st.session_state.running) or (not st.session_state.paused)):
        st.session_state.pause_event.clear()
        st.session_state.paused = False
        st.success("Resumed.")

    if c4.button("🛑 Stop", type="primary", disabled=not st.session_state.running):
        st.session_state.stop_event.set()
        st.session_state.running = False
        st.session_state.paused = False
        st.info("Stopping… (this may take a moment to wind down).")

    # Status panel
    st.subheader("Status")
    status_cols = st.columns(3)
    status_cols[0].metric("Running", "Yes" if st.session_state.running else "No")
    status_cols[1].metric("Paused", "Yes" if st.session_state.paused else "No")
    status_cols[2].metric("Mode", st.session_state.mode)

    # Buckets panel
    with st.expander("Rate Limit Usage", expanded=True):
        mb = st.session_state.media_bucket
        ib = st.session_state.image_bucket
        st.write(f"**Media tokens**: {mb.tokens:.0f}/{mb.capacity}  |  refill ~ {mb.refill_rate_per_sec:.2f}/sec")
        st.write(f"**Image frame tokens**: {ib.tokens:.0f}/{ib.capacity}  |  refill ~ {ib.refill_rate_per_sec:.2f}/sec")
        st.progress(min(1.0, mb.tokens / max(1, mb.capacity)))
        st.progress(min(1.0, ib.tokens / max(1, ib.capacity)))

    # Live logs
    st.subheader("Live Logs")
    log_box = st.empty()
    logs = []

    # Drain queued logs and render
    while True:
        try:
            msg = st.session_state.log_queue.get_nowait()
            logs.append(str(msg).rstrip())
        except queue.Empty:
            break
    if logs:
        log_box.code("\n".join(logs[-200:]), language="text")
    else:
        log_box.info("No logs yet. Start the session to see activity.")

    st.caption("Tip: If deployment environment cannot access microphone/speakers (e.g., some cloud hosts), "
               "you can set mode to **screen** or **none** to test connectivity and text logs.")


if __name__ == "__main__":
    main()
