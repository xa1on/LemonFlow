import os
import logging
from .recorder import Recorder

logger = logging.getLogger("lemonflow")
logging.basicConfig(level=logging.INFO)

class LemonFlowScribe:
    """
    live speech to text via lemonade realtime api w/ formatting n stuff
    """

    def __init__(self):
        self.recorder = Recorder()

    def start_listening(self, output_file_path: str):
        previous_text_len: int = 0
        def helper_write(inp_bytes, n: int):
            with open(output_file_path, "rb+") as f:
                f.seek(-n, os.SEEK_END)
                f.write(inp_bytes)
                f.truncate()

        def transcript_write(txt: str):
            logger.info("finished section")
            nonlocal previous_text_len
            inp_bytes = (txt + '\n\n').encode('utf-8')
            helper_write(inp_bytes, previous_text_len)
            previous_text_len = 0
        
        def delta_write(txt: str):
            nonlocal previous_text_len
            inp_bytes = txt.encode('utf-8')
            helper_write(inp_bytes, previous_text_len)
            previous_text_len = len(inp_bytes)
        
        # create file if not exist & clear if already exists
        with open(output_file_path, "w"):
            pass

        self.recorder.start_listening(delta_write, transcript_write)
    
    def stop_listening(self):
        self.recorder.stop_listening()

def main():
    lemon = LemonFlowScribe()
    lemon.start_listening("log.md")

if __name__ == "__main__":
    main()