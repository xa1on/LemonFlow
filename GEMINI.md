# LemonFlow (lemonade-vtt)

Live speech-to-text recording and transcription powered by the [lemonade-sdk](https://github.com/lemonade-sdk/lemonade) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Project Overview
LemonFlow is a real-time transcription tool that captures audio from the local microphone and streams it to a local Lemonade API server. It utilizes the OpenAI Realtime API protocol to provide live text updates (deltas) and completed transcripts as speech is processed.

### Key Technologies
- **Python 3.10+**: Core logic and audio processing.
- **OpenAI Python SDK**: Used for its `AsyncOpenAI` Realtime API client to communicate with the local server.
- **PyAudio**: For cross-platform audio capture.
- **NumPy**: For audio data manipulation and downsampling.
- **Lemonade API**: Local backend providing the STT engine (assumed to be running on `localhost:8000`).

### Architecture
- **Audio Capture**: Captures raw PCM16 audio at the system's native sample rate.
- **Downsampling**: Converts audio to 16kHz (mono) to match the expected input for Whisper models.
- **Streaming**: Uses WebSockets to stream audio chunks and receive transcription events in real-time.
- **Event Handling**: Separates "delta" events (immediate partial text) from "completed" events (finalized sentences/segments).

## Building and Running

### Prerequisites
1. **Python**: Ensure Python 3.10 or higher is installed.
2. **System Dependencies**: `pyaudio` requires PortAudio.
   - **Linux**: `sudo apt-get install libasound-dev portaudio19-dev`
   - **macOS**: `brew install portaudio`
   - **Windows**: Typically included in the wheel, but may require visual studio build tools.
3. **Backend**: A running [Lemonade API](https://github.com/lemonade-sdk/lemonade) server on `http://localhost:8000`.

### Setup
```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Unix/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python src/main.py
```
- Press `Ctrl+C` to stop listening and gracefully close the connection.

## Development Conventions
- **Asynchronous Code**: The project relies heavily on `asyncio` for non-blocking audio streaming and event reception.
- **Logging**: Uses the standard `logging` library for status updates and error reporting.
- **Error Handling**: Gracefully handles model loading failures and connection timeouts.
- **TODOs**:
    - [ ] String results into a unified script.
    - [ ] Local LLM formatting.
    - [ ] Autocorrect punctuation and grammar.
    - [ ] Global hotkeys and keyboard typing integration.

# DO NOT WRITE COMMENTS WITH PUNCTUATION OR UPPERCASE CHARACTERS

## Write docstrings in the following format:

Example:

```python
def downsample_pcm16(pcm16_bytes: bytes, native_rate: int, target_rate: int) -> bytes:
    """
    downsample pcm16 from native rate to target rate

    :param pcm16_bytes: raw pcm16 data
    :param native_rate: original rate
    :param target_rate: new rate
    """
```
