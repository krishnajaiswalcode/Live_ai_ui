# live_cam.py
# -----------------------------------------
# Real-time Gemini Live client with camera/screen streaming and audio I/O.
# Upgrades:
# - Pause / Resume / Stop controls via threading Events (safe from Streamlit)
# - Optional send_guard(media_dict) -> bool to rate-limit outgoing media
# - Optional logger(text: str) to forward model text and status to a UI
# - Safer task orchestration and graceful shutdown
# -----------------------------------------

import asyncio
import base64
import io
import traceback
import argparse
import os
from typing import Callable, Optional
from threading import Event

import cv2
import pyaudio
import PIL.Image
import mss

from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

# ---- Audio / stream constants ----
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY"))

tools = [
    types.Tool(google_search=types.GoogleSearch()),
]

# Only AUDIO response modality during experimental preview.
CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")
        )
    ),
    tools=types.ToolListUnion(tools),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    """
    Controls the Gemini Live session, audio capture/playback, and optional camera/screen streaming.

    External control (from UI like Streamlit) is done via:
      - pause_event: Event() -> when set, pause sending video/screen frames (mic still captured)
      - stop_event: Event() -> when set, stop the entire session ASAP

    Optional hooks:
      - send_guard: Callable[[dict], bool]  where dict has {"data": ..., "mime_type": "..."}.
                    Return False to drop the media (for rate limiting).
      - logger: Callable[[str], None] for status and text logs.
    """

    def __init__(
        self,
        video_mode: str = DEFAULT_MODE,
        pause_event: Optional[Event] = None,
        stop_event: Optional[Event] = None,
        send_guard: Optional[Callable[[dict], bool]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.video_mode = video_mode
        self.pause_event = pause_event or Event()
        self.stop_event = stop_event or Event()
        self.send_guard = send_guard
        self.logger = logger

        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)

        self.session = None
        self.audio_stream = None

    # ------------- helpers -------------

    def log(self, msg: str):
        if self.logger:
            try:
                self.logger(msg)
            except Exception:
                pass
        else:
            print(msg)

    # ------------- text input (optional) -------------

    async def send_text(self):
        # Optional manual CLI input; in UI you can ignore this task.
        while not self.stop_event.is_set():
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                self.log("\nUser requested exit.")
                self.stop_event.set()
                break
            if self.session is not None:
                await self.session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=text or ".")])
                )
            else:
                self.log("Session not initialized; unable to send text.")

    # ------------- camera --------------

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail((1024, 1024))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        image_bytes = image_io.read()
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        try:
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    await asyncio.sleep(0.1)
                    continue
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    self.log("Camera frame read failed (None).")
                    break
                await asyncio.sleep(1.0)  # throttle to ~1 fps
                await self.out_queue.put(frame)
        finally:
            cap.release()

    # ------------- screen --------------

    def _grab_screen_jpeg(self):
        sct = mss.mss()
        monitor = sct.monitors[0]  # full virtual screen
        shot = sct.grab(monitor)

        # Convert raw BGRA -> RGB PIL image safely
        img = PIL.Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
        img.thumbnail((1280, 720))  # reasonable bound for bandwidth

        buf = io.BytesIO()
        img.save(buf, format="jpeg")
        buf.seek(0)
        return {"mime_type": "image/jpeg", "data": base64.b64encode(buf.read()).decode()}

    async def get_screen(self):
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                await asyncio.sleep(0.1)
                continue
            frame = await asyncio.to_thread(self._grab_screen_jpeg)
            await asyncio.sleep(1.0)  # throttle ~1 fps
            await self.out_queue.put(frame)

    # ------------- realtime send -------------

    async def send_realtime(self):
        while not self.stop_event.is_set():
            msg = await self.out_queue.get()
            if self.stop_event.is_set():
                break
            if self.send_guard and not self.send_guard(msg):
                # Rate limiter said "drop this media"
                continue
            if self.session is not None:
                try:
                    await self.session.send_realtime_input(
                        media=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                    )
                except Exception as e:
                    self.log(f"[send_realtime] error: {e}")
            else:
                self.log("Session not initialized; unable to send realtime media.")

    # ------------- mic capture -------------

    async def listen_audio(self):
        try:
            mic_info = pya.get_default_input_device_info()
            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=int(mic_info["index"]),
                frames_per_buffer=CHUNK_SIZE,
            )
            kwargs = {"exception_on_overflow": False} if __debug__ else {}
            while not self.stop_event.is_set():
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                packet = {"data": data, "mime_type": "audio/pcm"}
                if self.send_guard and not self.send_guard(packet):
                    continue
                await self.out_queue.put(packet)
        except Exception as e:
            self.log(f"[listen_audio] error: {e}")
        finally:
            try:
                if self.audio_stream is not None:
                    self.audio_stream.close()
            except Exception:
                pass

    # ------------- receive & play model audio -------------

    async def receive_audio(self):
        """Reads from the websocket and writes PCM chunks to the output queue for playback."""
        while not self.stop_event.is_set():
            if self.session is None:
                await asyncio.sleep(0.1)
                continue

            try:
                turn = self.session.receive()
                async for response in turn:
                    if self.stop_event.is_set():
                        break
                    if data := response.data:
                        # streaming audio bytes
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        # occasional text (logs/debug)
                        self.log(text)
                # If interrupted, flush any buffered audio
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
            except Exception as e:
                self.log(f"[receive_audio] error: {e}")
                await asyncio.sleep(0.1)

    async def play_audio(self):
        try:
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            while not self.stop_event.is_set():
                bytestream = await self.audio_in_queue.get()
                if self.stop_event.is_set():
                    break
                await asyncio.to_thread(stream.write, bytestream)
        except Exception as e:
            self.log(f"[play_audio] error: {e}")

    # ------------- main orchestration -------------

    async def run(self, enable_cli_input: bool = False):
        """Run the full pipeline until stop_event is set or user quits (CLI q)."""
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Optional CLI text task (off by default for UIs)
                if enable_cli_input:
                    tg.create_task(self.send_text())

                # Core tasks
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())
                # elif "none": no video task

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Wait for stop_event
                while not self.stop_event.is_set():
                    await asyncio.sleep(0.05)

                # Raising CancelledError to tear down the TaskGroup gracefully
                raise asyncio.CancelledError("Stop requested")
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            try:
                if self.audio_stream:
                    self.audio_stream.close()
            except Exception:
                pass
            traceback.print_exception(EG)
        finally:
            try:
                if self.audio_stream:
                    self.audio_stream.close()
            except Exception:
                pass


# ------------- CLI entrypoint -------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="video/screen source to stream",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Enable CLI input (type messages and 'q' to quit).",
    )
    args = parser.parse_args()

    pause_event = Event()
    stop_event = Event()

    loop = AudioLoop(
        video_mode=args.mode,
        pause_event=pause_event,
        stop_event=stop_event,
    )
    asyncio.run(loop.run(enable_cli_input=args.cli))
