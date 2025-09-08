# streamlit_app_fixed.py
import time
import threading
import queue
import asyncio
from dataclasses import dataclass

import streamlit as st

from live_cam import AudioLoop, DEFAULT_MODE

# ----- Config -----
DEFAULT_HOURLY_TOKENS = 600
DEFAULT_FPS_BUDGET = 60

@dataclass
class TokenBucket:
    capacity: int
    refill_rate_per_sec: float
    tokens: float
    last_refill: float

    def try_consume(self, n: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
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

# ----- Session state init -----
def ensure_state():
    if "username" not in st.session_state: st.session_state.username = ""
    if "mode" not in st.session_state: st.session_state.mode = DEFAULT_MODE
    if "running" not in st.session_state: st.session_state.running = False
    if "paused" not in st.session_state: st.session_state.paused = False
    if "log_queue" not in st.session_state: st.session_state.log_queue = queue.Queue()
    if "pause_event" not in st.session_state: st.session_state.pause_event = threading.Event()
    if "stop_event" not in st.session_state: st.session_state.stop_event = threading.Event()
    if "loop_thread" not in st.session_state: st.session_state.loop_thread = None
    if "loop_instance" not in st.session_state: st.session_state.loop_instance = None
    if "media_bucket" not in st.session_state: st.session_state.media_bucket = make_token_bucket(DEFAULT_HOURLY_TOKENS, 3600)
    if "image_bucket" not in st.session_state: st.session_state.image_bucket = make_token_bucket(DEFAULT_FPS_BUDGET, 3600)

# ----- Background runner (thread-safe: no st.* inside thread) -----
def run_asyncio_loop(mode: str, pause_event: threading.Event, stop_event: threading.Event,
                     media_bucket: TokenBucket, image_bucket: TokenBucket, log_queue: queue.Queue):
    """
    This function runs inside a background thread. It must NOT call st.* or touch Streamlit internals.
    We'll create local closures for logger and send_guard that use only thread-safe objects.
    """
    def logger(msg: str):
        try:
            log_queue.put_nowait(str(msg))
        except Exception:
            print("[logger] failed to put log:", msg)

    def send_guard(media: dict) -> bool:
        mime = media.get("mime_type", "")
        if not media_bucket.try_consume(1):
            logger("⛔ Rate limit reached: media budget exhausted.")
            return False
        if mime.startswith("image/"):
            if not image_bucket.try_consume(1):
                logger("⛔ Rate limit reached: image frame budget exhausted.")
                return False
        return True

    # Create and run the AudioLoop with thread-safe objects
    loop_inst = AudioLoop(
        video_mode=mode,
        pause_event=pause_event,
        stop_event=stop_event,
        send_guard=send_guard,
        logger=logger,
    )

    try:
        logger(f"[runner] Starting AudioLoop in mode='{mode}'")
        asyncio.run(loop_inst.run(enable_cli_input=False))
    except Exception as e:
        logger(f"[runner] crashed: {e}")
        import traceback as tb
        tb.print_exc()
    finally:
        # final cleanup logs (these are thread-safe writes to the queue)
        logger("[runner] Session ended.")
        try:
            stop_event.set()
        except Exception:
            pass

# ----- Streamlit UI -----
def main():
    ensure_state()
    st.set_page_config(page_title="Gemini Live Assistance", page_icon="🎥", layout="wide")
    st.title("🎥 Gemini Live Assistance (fixed)")
    st.caption("Stream mic + (camera/screen) to Gemini 2.0 Flash Live and get real-time audio replies.")

    with st.sidebar:
        st.header("User & Limits")
        st.session_state.username = st.text_input("User ID / Name", st.session_state.username, placeholder="e.g., krishna")
        st.session_state.mode = st.selectbox("Video Mode", ["camera", "screen", "none"], index=["camera","screen","none"].index(st.session_state.mode))
        st.subheader("Rate Limits (per hour)")
        total_tokens = st.number_input("Total media packets", min_value=60, max_value=5000, value=DEFAULT_HOURLY_TOKENS, step=60)
        image_tokens = st.number_input("Image frames", min_value=0, max_value=1000, value=DEFAULT_FPS_BUDGET, step=10)
        if st.button("Apply Limits"):
            st.session_state.media_bucket = make_token_bucket(int(total_tokens), 3600)
            st.session_state.image_bucket = make_token_bucket(int(image_tokens), 3600)
            st.success("Limits applied.")

        st.divider()
        st.info("Set GEMINI_API_KEY in your environment (or Streamlit Secrets for cloud).")

    # Controls
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    if c1.button("▶️ Start", disabled=st.session_state.running):
        if not st.session_state.username.strip():
            st.warning("Please enter a User ID / Name in the sidebar.")
        else:
            # Reset events and buckets
            st.session_state.stop_event.clear()
            st.session_state.pause_event.clear()
            st.session_state.paused = False
            st.session_state.running = True

            # Re-seed token buckets
            st.session_state.media_bucket = make_token_bucket(int(st.session_state.media_bucket.capacity), 3600)
            st.session_state.image_bucket = make_token_bucket(int(st.session_state.image_bucket.capacity), 3600)

            # Start the background thread but pass thread-safe objects (no st.* inside thread)
            t = threading.Thread(
                target=run_asyncio_loop,
                args=(
                    st.session_state.mode,
                    st.session_state.pause_event,
                    st.session_state.stop_event,
                    st.session_state.media_bucket,
                    st.session_state.image_bucket,
                    st.session_state.log_queue,
                ),
                daemon=True
            )
            st.session_state.loop_thread = t
            t.start()
            st.success("Started session (background thread launched).")

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
        st.info("Stopping… (this may take a moment).")

    # Status
    st.subheader("Status")
    status_cols = st.columns(3)
    status_cols[0].metric("Running", "Yes" if st.session_state.running else "No")
    status_cols[1].metric("Paused", "Yes" if st.session_state.paused else "No")
    status_cols[2].metric("Mode", st.session_state.mode)

    with st.expander("Rate Limit Usage", expanded=True):
        mb = st.session_state.media_bucket
        ib = st.session_state.image_bucket
        st.write(f"**Media tokens**: {mb.tokens:.0f}/{mb.capacity}  |  refill ~ {mb.refill_rate_per_sec:.2f}/sec")
        st.write(f"**Image frame tokens**: {ib.tokens:.0f}/{ib.capacity}  |  refill ~ {ib.refill_rate_per_sec:.2f}/sec")
        st.progress(min(1.0, mb.tokens / max(1, mb.capacity)))
        st.progress(min(1.0, ib.tokens / max(1, ib.capacity)))

    # Logs
    st.subheader("Live Logs")
    log_box = st.empty()
    logs = []
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

    st.caption("If microphone/speaker unavailable (cloud), set mode to 'screen' or 'none' to test connectivity.")

if __name__ == "__main__":
    main()
