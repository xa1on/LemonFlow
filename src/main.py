import logging
from .recorder import Recorder
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000/api/v1"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
DEFAULT_MODEL = "Whisper-Tiny"

logger = logging.getLogger("lemonflow")
logging.basicConfig(level=logging.INFO)

class LemonFlow:
    """
    live speech to text via lemonade realtime api
    streams audio and fires callback on every text delta

    
    """

    def __init__(self):
        self.recorder = Recorder()

def main():
    lemon = LemonFlow()

if __name__ == "__main__":
    main()