import os
import time
import struct
import logging
import queue
import threading
import asyncio
import base64
import json
import urllib.request
import numpy as np
import pyaudio
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000/api/v1"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
DEFAULT_MODEL = "Whisper-Tiny"

logger = logging.getLogger("lemonflow")
logging.basicConfig(level=logging.INFO)

def downsample_pcm16(pcm16_bytes, native_rate, target_rate):
    """
    downsample pcm16 from native rate to target rate

    :param pcm16_bytes: raw pcm16 data
    :param native_rate: original rate
    :param target_rate: new rate
    """
    if native_rate == target_rate:
        return pcm16_bytes
    n_samples = len(pcm16_bytes) // 2
    samples = struct.unpack(f'<{n_samples}h', pcm16_bytes)
    ratio = native_rate / target_rate
    output_length = int(n_samples / ratio)
    output = bytearray(output_length * 2)
    for i in range(output_length):
        src_idx = i * ratio
        idx_floor = int(src_idx)
        frac = src_idx - idx_floor
        struct.pack_into('<h', output, i * 2, max(-32768, min(32767, int(samples[idx_floor] * (1 - frac) + samples[min(idx_floor + 1, n_samples - 1)] * frac))))
    return bytes(output)

class LemonFlow:
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
        self._load_model()

    def _load_model(self) -> None:
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
            return
        
    def _get_ws_port(self) -> None | int:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as resp:
                health = json.loads(resp.read().decode())
                ws_port = health.get("websocket_port")
                if not ws_port:
                    logger.error("server did not provide health")
                    return
                logger.info(f"found websocket port: {ws_port}")
                return ws_port
        except Exception as e:
            logger.error(f"error fetching websocket: {e}")
            return

    async def _realtime_connection(self, client: AsyncOpenAI, delta_callback: callable, transcript_callback: callable):
        logger.info("connecting...")

        async with client.beta.realtime.connect(model=self.model) as conn:
            event = await asyncio.wait_for(conn.recv(), timeout=15)
            logger.info(f"session {event.session.id}")
            self.is_listening = True

            transcription_complete = asyncio.Event()

            pa = pyaudio.PyAudio()
            device_info = pa.get_default_input_device_info()
            native_rate = int(device_info['defaultSampleRate'])
            
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=native_rate,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            logger.info("recording...")

            async def send_audio():
                try:
                    while self.is_listening:
                        data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        data = downsample_pcm16(data, native_rate, self.sample_rate)
                        
                        await conn.input_audio_buffer.append(
                            audio=base64.b64encode(data).decode()
                        )
                        await asyncio.sleep(0.01)
                    
                    logger.info("committing final audio...")
                    await conn.input_audio_buffer.commit()

                except asyncio.CancelledError:
                    logger.info("committing final audio...")
                    await conn.input_audio_buffer.commit()
                    self.is_listening = False
                    pass
            
            async def receive_transcriptions():
                try:
                    async for event in conn:
                        if event.type == "conversation.item.input_audio_transcription.delta":
                            print(event)
                            delta_callback(getattr(event, "delta", "").replace('\n', ' ').strip())
                        
                        elif event.type == "conversation.item.input_audio_transcription.completed":
                            transcript_callback(getattr(event, "transcript", "").replace('\n', ' ').strip())
                            if not self.is_listening:
                                transcription_complete.set()
                        
                        elif event.type == "error":
                            error = getattr(event, "error", None)
                            msg = getattr(error, "message", "Unknown") if error else "Unknown"
                            logger.error(f"\nerror: {msg}")

                except asyncio.CancelledError:
                    pass

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_transcriptions())

            await send_task
            logger.info("received stop signal...")
            send_task.cancel()
            try:
                await asyncio.wait_for(transcription_complete.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("final transcript never finished.")

            receive_task.cancel()
            


            if stream.is_active():
                stream.stop_stream()
            stream.close()
            pa.terminate()
            logger.info("ended stream")

            


    def start_listening(self, delta_callback: callable, transcript_callback: callable) -> None:
        """
        starts audio stream
        """
        if self.is_listening:
            logger.warning("already listening.")
            return
        ws_port = self._get_ws_port()
        client = AsyncOpenAI(
            api_key="lemonflow",
            base_url=self.base_url,
            websocket_base_url=f"ws://localhost:{ws_port}"
        )
        self.is_listening = True
        logger.info("started listening.")
        asyncio.run(self._realtime_connection(client, delta_callback, transcript_callback))

    
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
    lemon = LemonFlow()
    
    def on_text_received(text: str) -> None:
        print(text)
    
    def on_transcript_received(text: str) -> None:
        print(f"\n---\n{text}\n---\n\n")

    try:
        lemon.start_listening(delta_callback=on_text_received, transcript_callback=on_transcript_received)
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        lemon.stop_listening()

if __name__ == "__main__":
    main()