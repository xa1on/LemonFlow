"""
LemonFlow

live speech to text recording and transcription via lemonade and whisper.cpp

Authors: Chenghao Li
Org: OCF (i guess)
"""


import io
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from io import BytesIO
from openai import OpenAI

BASE_URL = "http://localhost:8000/api/v1"
SAMPLE_RATE = 16000 # recording sample rate (samples/sec) default: 16000 (whisper.cpp)

DEFAULT_MODEL = "Whisper-Base" # stt model
VAD_START_THRESHOLD = 0.02 # threshold required to start recording
VAD_SILENCE_THRESHOLD = 0.005 # threshold required to continue a recording (dont stop recording prematurely)
SILENCE_DURATION = 1 # length in seconds of silence before stopping recording 
MIN_RECORDING_LEN = 0.3 # minimum length of recording in seconds
TEMPERATURE = 0.5 # transcription model temp


class LemonFlow:
    """
    live speech to text recording and transcription via lemonade and whisper.cpp

    :param self:
    :param model: speech to text model name
    :param temperature: speech to text model
    :param sample_rate: audio sample rate
    :param base_url: openai api base url to connect to lemonade
    """

    def __init__(self, model: str=DEFAULT_MODEL, temperature: float=TEMPERATURE, sample_rate: int=SAMPLE_RATE, base_url=BASE_URL):
        self.model = model
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.client = OpenAI(base_url=base_url, api_key="lemonflow")
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None

    def _save_audio(self, inp_audio) -> BytesIO:
        """
        writes audio into memory to be transcribed
        
        :param self:
        :param inp_audio: input audio
        :return: saved audio in memory
        """
        byte_io = io.BytesIO()
        audio_int16 = (inp_audio * 32767).astype(np.int16)
        wav.write(byte_io, self.sample_rate, audio_int16)
        
        byte_io.seek(0)
        byte_io.name = f"chunk{str(time.time())}.wav"
        return byte_io
    
    def _transcribe_io(self, byte_io: BytesIO) -> str:
        """
        trascribe audio saved in memory via openai api
        
        :param self:
        :param byte_io: stored audio
        :return: resulting transcription
        """
        try:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=byte_io,
                temperature=self.temperature
            )
            return transcript.text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
    
    def _transcribe_worker(self, audio_data, callback) -> None:
        """
        transcribes audio data, then runs callback with result
        
        :param self:
        :param audio_data: raw audio data
        :param callback: function to call with resulting transcription
        """
        text = self._transcribe_io(self._save_audio(audio_data), self.model)
        if text:
            callback(text)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """
        stores input data from audio stream into queue for processing
        
        :param self:
        :param indata: input data from stream
        :param frames:
        :param time_info:
        :param status: audio status
        """
        if status:
            print(f"Audio Status: {status}")
        self.audio_queue.put(indata.copy())
    
    def _process_loop(self, callback) -> None:
        """
        main processing loop for audio listening and recording -> transcription
        
        :param self:
        :param callback: callback function for resulting transcriptions
        """
        print(f"Listening...")
        recording_buffer = []
        is_recording = False
        silence_start_time = None

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback):
            while self.is_running:
                try:
                    indata = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                volume = np.linalg.norm(indata) / np.sqrt(len(indata))
                
                # use start threshold when not recording and silence threshold when recording
                if (not is_recording and volume > VAD_START_THRESHOLD) or (is_recording and volume > VAD_SILENCE_THRESHOLD):
                    if not is_recording:
                        print("Voice detected")
                        is_recording = True
                    silence_start_time = None
                    recording_buffer.append(indata)
                
                elif is_recording:
                    recording_buffer.append(indata) 
                    
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    
                    elif (time.time() - silence_start_time) > SILENCE_DURATION:
                        if len(recording_buffer) > 0:
                            full_audio = np.concatenate(recording_buffer, axis=0)
                            duration = len(full_audio) / self.sample_rate
                            
                            if duration >= MIN_RECORDING_LEN:
                                print("Transcribing", end="\r", flush=True)
                                t = threading.Thread(
                                    target=self._transcribe_worker, 
                                    args=(full_audio, callback),
                                    daemon=True
                                )
                                t.start()
                        
                        # clear buffers
                        recording_buffer = []
                        is_recording = False
                        silence_start_time = None

    def start_listening(self, callback, model: str|None, temperature: float|None) -> None:
        """
        start listening via microphone
        
        :param self:
        :param callback: callback function for resulting transcriptions
        :param model: speech to text model name
        :param temperature: speech to text model temperature
        """
        if model != None:
            self.model = model
        if temperature != None:
            self.temperature = temperature
        
        if self.is_running:
            print("Already running.")
            return
        self.is_running = True
        
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        self.processing_thread = threading.Thread(
            target=self._process_loop, 
            args=(callback, model),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_listening(self) -> None:
        """
        ends listening and transcription
        
        :param self:
        """
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        print("Stopped listening.")

def main():
    # example code of streaming transcription
    lemon = LemonFlow()
    
    def on_text_received(text: str) -> None:
        print(f"\nOutput: {text}")
        print("Listening...", end="\r", flush=True)
        
    try:
        lemon.start_listening(callback=on_text_received)
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        lemon.stop_listening()

if __name__ == "__main__":
    main()