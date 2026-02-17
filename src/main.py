import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import io
from io import BytesIO

from openai import OpenAI

SAMPLE_RATE = 44100
BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "Whisper-Tiny"

class LemonFlow:

    def __init__(self, sample_rate: int=SAMPLE_RATE, base_url=BASE_URL):
        self.sample_rate = sample_rate
        self.client = OpenAI(
            base_url=base_url,
            api_key="lemonflow"
        )

    def save_audio(self, inp_audio) -> BytesIO:
        byte_io = io.BytesIO()
        wav.write(byte_io, self.sample_rate, inp_audio)
        byte_io.seek(0)
        byte_io.name = "chunk.wav"
        return byte_io
    
    def transcribe_io(self, byte_io: BytesIO, model: str=DEFAULT_MODEL) -> str:
        print("transcribing")
        transcript = self.client.audio.transcriptions.create(
            model=model,
            file=byte_io
        )
        return transcript.text

    def transcribe(self, inp_audio, model: str=DEFAULT_MODEL) -> str:
        return self.transcribe_io(self.save_audio(inp_audio), model)
    
    def start_recording(self, length: int=10):
        print("recording started")
        recording = sd.rec(int(length * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("recording ended")
        print(self.transcribe(recording))

def main():
    lemon_flow = LemonFlow()
    lemon_flow.start_recording()

if __name__ == "__main__":
    main()