import time
import queue
import threading
import asyncio
import base64
import json
import urllib.request
import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000/api/v1"
SAMPLE_RATE = 16000 
DEFAULT_MODEL = "Whisper-Tiny" 

class LemonFlow:
    """
    live speech to text via lemonade realtime api
    streams audio and fires callback on every text delta

    :param self:
    :param model: speech to text model name
    :param sample_rate: audio sample rate
    :param base_url: lemonade api base url
    """

    def __init__(self, model: str=DEFAULT_MODEL, sample_rate: int=SAMPLE_RATE, base_url=BASE_URL):
        self.model = model
        self.sample_rate = sample_rate
        self.base_url = base_url
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None

    def _get_ws_port(self) -> str:
        """
        fetches dynamic websocket port from lemonade health endpoint
        """
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data.get("websocket_port")
        except Exception as e:
            print(f"error getting port: {e}")
            return None

    def _ensure_model_loaded(self) -> None:
        """
        ensures model is loaded via rest api
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/load",
                data=json.dumps({"model_name": self.model}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60):
                pass
        except Exception:
            pass

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """
        sounddevice callback to fill queue
        """
        if status:
            print(f"status: {status}")
        self.audio_queue.put(indata.copy())

    async def _realtime_worker(self, callback):
        """
        async worker using openai sdk to stream audio and events
        """
        self._ensure_model_loaded()
        ws_port = self._get_ws_port()
        
        if not ws_port:
            print("server not healthy or port not found")
            return

        client = AsyncOpenAI(
            api_key="unused",
            base_url=self.base_url,
            websocket_base_url=f"ws://localhost:{ws_port}"
        )

        print("connecting...")
        
        try:
            async with client.beta.realtime.connect(model=self.model) as conn:
                # wait for session created event
                await asyncio.wait_for(conn.recv(), timeout=10)
                print("listening...")

                async def send_audio():
                    while self.is_running:
                        try:
                            # get audio from queue
                            indata = self.audio_queue.get_nowait()
                            
                            # convert to pcm16
                            audio_int16 = (indata * 32767).astype(np.int16)
                            pcm_bytes = audio_int16.tobytes()
                            
                            # send to buffer
                            await conn.input_audio_buffer.append(
                                audio=base64.b64encode(pcm_bytes).decode("utf-8")
                            )
                            await asyncio.sleep(0) 
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                        except Exception as e:
                            print(f"send error: {e}")
                            break

                async def receive_transcripts():
                    async for event in conn:
                        if not self.is_running:
                            break
                        print(event)
                        # handle partial text updates immediately
                        if event.type == "response.audio_transcript.delta":
                            if hasattr(event, 'delta') and event.delta:
                                callback(event.delta)
                                
                        # compatibility for older lemonade versions
                        elif event.type == "conversation.item.input_audio_transcription.delta":
                            if hasattr(event, 'delta') and event.delta:
                                callback(event.delta)

                await asyncio.gather(send_audio(), receive_transcripts())

        except Exception as e:
            print(f"connection error: {e}")

    def _process_loop(self, callback) -> None:
        """
        runs the async loop in a separate thread
        """
        try:
            asyncio.run(self._realtime_worker(callback))
        except Exception as e:
            print(f"loop error: {e}")

    def start_listening(self, callback, model: str|None=None) -> None:
        """
        starts audio stream and background processing
        """
        if model:
            self.model = model
        
        if self.is_running:
            print("already running")
            return
        self.is_running = True
        
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            callback=self._audio_callback,
            blocksize=4096
        )
        self.stream.start()

        self.processing_thread = threading.Thread(
            target=self._process_loop, 
            args=(callback,),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_listening(self) -> None:
        """
        stops stream and thread
        """
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if self.processing_thread:
            self.processing_thread.join()
        print("\nstopped listening")

def main():
    lemon = LemonFlow()
    
    def on_text_received(text: str) -> None:
        # print chunks as they arrive without newlines
        pass # print(text, end="", flush=True)
        
    try:
        lemon.start_listening(callback=on_text_received)
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        lemon.stop_listening()

if __name__ == "__main__":
    main()