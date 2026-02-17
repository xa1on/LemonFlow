import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import io

from openai import OpenAI

SAMPLE_RATE = 44100


def main():
    print("loading model")
    client = OpenAI(
        base_url="http://localhost:8000/api/v1",
        api_key="lemonade"
    )
    print("model loaded")
    print("recording...")
    recording = sd.rec(int(10 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("recording ended.")

    byte_io = io.BytesIO()
    wav.write(byte_io, SAMPLE_RATE, recording)
    byte_io.seek(0)
    byte_io.name = "chunk.wav"
    print("transcribing...")
    transcript = client.audio.transcriptions.create(
        model="Whisper-Tiny",
        file=byte_io
    )
    print(transcript.text.strip())

if __name__ == "__main__":
    main()