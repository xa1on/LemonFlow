import logging
import asyncio
import base64
import json
import urllib.request
import numpy as np
import pyaudio
import threading
import collections
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000/api/v1"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
DEFAULT_MODEL = "Whisper-Base"
BUFFER_SECONDS = 4.0

logger = logging.getLogger("recorder")
logging.basicConfig(level=logging.INFO)

def downsample_pcm16(pcm16_bytes: bytes, native_rate: int, target_rate: int) -> bytes:
    """
    downsample pcm16 from native rate to target rate

    :param pcm16_bytes: raw pcm16 data
    :param native_rate: original rate
    :param target_rate: new rate
    :return: downsampled result
    """
    if native_rate == target_rate:
        return pcm16_bytes
    
    audio_data = np.frombuffer(pcm16_bytes, dtype=np.int16)
    num_samples = len(audio_data)
    new_num_samples = int(num_samples * target_rate / native_rate)
    
    resampled_audio = np.interp(
        np.linspace(0, num_samples, new_num_samples, endpoint=False),
        np.arange(num_samples),
        audio_data
    ).astype(np.int16)
    
    return resampled_audio.tobytes()

class Recorder:
    """
    live speech to text via lemonade realtime api
    streams audio and fires callback on every text delta

    :param self:
    :param model: speech to text model name
    :param sample_rate: audio sample rate
    :param base_url: lemonade api base url
    """

    def __init__(self, model: str=DEFAULT_MODEL, sample_rate: int=SAMPLE_RATE, chunk_size: int=CHUNK_SIZE, base_url=BASE_URL):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.base_url = base_url
        self.is_listening = False
        
        # buffering system
        # ensure buffer can hold at least BUFFER_SECONDS even with high native sample rates
        # (native_rate/sample_rate) more chunks are needed because resampled chunks are smaller
        # we use a conservative multiplier of 4 to cover up to 64khz native rate
        self._buffer = collections.deque(maxlen=int(SAMPLE_RATE * BUFFER_SECONDS / chunk_size * 4) + 10)
        self._stream_active = True
        self._audio_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self._audio_thread.start()
        
        self._load_model()

    def _capture_audio(self) -> None:
        """
        continuously capture audio into a circular buffer
        """
        pa = pyaudio.PyAudio()
        try:
            device_info = pa.get_default_input_device_info()
            native_rate = int(device_info['defaultSampleRate'])
            
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=native_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("audio hardware initialized and buffering")
            
            while self._stream_active:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                data = downsample_pcm16(data, native_rate, self.sample_rate)
                self._buffer.append(data)
                
        except Exception as e:
            logger.error(f"audio capture error: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            pa.terminate()

    def _load_model(self) -> None:
        """
        load whisper.cpp model
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/load",
                data=json.dumps({"model_name": self.model}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as _:
                logger.info(f"model loaded: {self.model}")
        except Exception as e:
            logger.error(f"error loading model: {e}")
        
    def _get_ws_port(self) -> int | None:
        """
        get websocket port

        :return: websocket port
        """
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as resp:
                health = json.loads(resp.read().decode())
                ws_port = health.get("websocket_port")
                if not ws_port:
                    logger.error("server did not provide health")
                    return None
                logger.info(f"found websocket port: {ws_port}")
                return ws_port
        except Exception as e:
            logger.error(f"error fetching websocket: {e}")
            return None

    async def _realtime_connection(self, client: AsyncOpenAI, delta_callback: callable, transcript_callback: callable) -> None:
        """
        initialize async realitime_conection

        :param client: openai client
        :param delta_callback: callable to call when delta event recieved (first param will be the delta text)
        :param transcript_callback: callable to call when transcript completed event recieved (first param will be transcript text)
        """
        logger.info("connecting...")

        async with client.beta.realtime.connect(model=self.model) as conn:
            try:
                event = await asyncio.wait_for(conn.recv(), timeout=15)
                session_id = getattr(event.session, "id", "unknown") if hasattr(event, "session") else "unknown"
                logger.info(f"connected to session: {session_id}")
            except Exception as e:
                logger.error(f"initial connection failed: {e}")
                return

            if not self.is_listening:
                logger.info("stop signal received during connection phase")
                return

            # ensure transcription is enabled for the session
            try:
                await conn.session.update(session={
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {"type": "server_vad"}
                })
                logger.info("session configured for continuous transcription")
            except Exception as e:
                logger.warning(f"could not update session config (might be a simplified server): {e}")

            transcription_complete = asyncio.Event()
            last_event_type = "conversation.item.input_audio_transcription.completed"

            logger.info("streaming buffered audio...")

            async def send_audio():
                nonlocal last_event_type
                try:
                    # send buffered data first
                    while len(self._buffer) > 0:
                        data = self._buffer.popleft()
                        await conn.input_audio_buffer.append(
                            audio=base64.b64encode(data).decode()
                        )
                        # yield control to avoid blocking during burst
                        await asyncio.sleep(0)

                    # small pause after burst to let server process the initial segment
                    await asyncio.sleep(0.1)

                    # continue streaming live data
                    while self.is_listening:
                        if len(self._buffer) > 0:
                            data = self._buffer.popleft()
                            await conn.input_audio_buffer.append(
                                audio=base64.b64encode(data).decode()
                            )
                        else:
                            await asyncio.sleep(0.02)
                    
                    # flush remaining chunks
                    while len(self._buffer) > 0:
                        data = self._buffer.popleft()
                        await conn.input_audio_buffer.append(
                            audio=base64.b64encode(data).decode()
                        )
                except Exception as e:
                    logger.error(f"send_audio error: {e}")
                finally:
                    if last_event_type != "conversation.item.input_audio_transcription.completed":
                        logger.info("committing final audio...")
                        try:
                            await conn.input_audio_buffer.commit()
                        except:
                            pass
                    self.is_listening = False
            
            async def receive_transcriptions():
                nonlocal last_event_type
                try:
                    async for event in conn:
                        last_event_type = event.type
                        if event.type == "conversation.item.input_audio_transcription.delta":
                            delta_callback(getattr(event, "delta", "").replace('\n', ' ').strip())
                        
                        elif event.type == "conversation.item.input_audio_transcription.completed":
                            transcript_callback(getattr(event, "transcript", "").replace('\n', ' ').strip())
                            if not self.is_listening:
                                transcription_complete.set()
                        
                        elif event.type == "error":
                            error = getattr(event, "error", None)
                            msg = getattr(error, "message", "Unknown") if error else "Unknown"
                            logger.error(f"server error: {msg}")

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"receive_transcriptions error: {e}")

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_transcriptions())

            await send_task
            logger.info("received stop signal...")
            send_task.cancel()

            if last_event_type != "conversation.item.input_audio_transcription.completed":
                try:
                    await asyncio.wait_for(transcription_complete.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.warning("final transcript never finished.")
            
            receive_task.cancel()
            logger.info("ended stream")

    def start_listening(self, delta_callback: callable, transcript_callback: callable) -> None:
        """
        starts audio stream

        :param delta_callback: callable to call when delta event recieved (first param will be the delta text)
        :param transcript_callback: callable to call when transcript completed event recieved (first param will be transcript text)
        """
        if self.is_listening:
            logger.warning("already listening.")
            return
        
        ws_port = self._get_ws_port()
        if ws_port is None:
            logger.error("could not determine port")
            return

        client = AsyncOpenAI(
            api_key="lemonflow",
            base_url=self.base_url,
            websocket_base_url=f"ws://localhost:{ws_port}"
        )
        self.is_listening = True
        logger.info("started listening.")
        try:
            asyncio.run(self._realtime_connection(client, delta_callback, transcript_callback))
        except KeyboardInterrupt:
            self.stop_listening()

    def stop_listening(self) -> None:
        """
        stops stream
        """
        if not self.is_listening:
            logger.warning("not currently listening.")
            return
        logger.info("stopping...")
        self.is_listening = False

def main():
    """
    example code
    """
    recorder = Recorder()
    
    def on_text_received(txt: str) -> None:
        print(txt)
    
    def on_transcript_received(txt: str) -> None:
        if txt:
            print(f"\n\n---\n{txt}\n---\n")

    recorder.start_listening(delta_callback=on_text_received, transcript_callback=on_transcript_received)

if __name__ == "__main__":
    main()
