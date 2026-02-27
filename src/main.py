import os
import time
import logging
import threading

from .recorder import Recorder

from pynput import keyboard

logger = logging.getLogger("lemonflow")
logging.basicConfig(level=logging.INFO)

SCRIBE_KEYBIND = "<ctrl>+<alt>+s"

class LemonFlowScribe:
    """
    live speech to text via lemonade realtime api with formatting and stuff
    """

    def __init__(self):
        self.recorder = Recorder()

    def start_listening(self, output_file_path: str):
        """
        start listening and writing to file

        :param output_file_path: path to output file
        """
        previous_text_len: int = 0
        def helper_write(inp_bytes, n: int):
            with open(output_file_path, "rb+") as f:
                f.seek(-n, os.SEEK_END)
                f.write(inp_bytes)
                f.truncate()

        def transcript_write(txt: str):
            logger.info("finished section")
            nonlocal previous_text_len
            helper_write((txt + '\n\n').encode('utf-8'), previous_text_len)
            previous_text_len = 0
        
        def delta_write(txt: str):
            nonlocal previous_text_len
            inp_bytes = txt.encode('utf-8')
            helper_write(inp_bytes, previous_text_len)
            previous_text_len = len(inp_bytes)
        
        # create file if not exist and clear if already exists
        with open(output_file_path, "w"):
            pass

        self.recorder.start_listening(delta_write, transcript_write)
    
    def stop_listening(self):
        """
        stop listening
        """
        self.recorder.stop_listening()

def main():
    """
    main entry point for lemonflow scribe
    """
    lemon = LemonFlowScribe()
    def on_press_callback():
        lemon.start_listening("log.md")
    def on_release_callback():
        lemon.stop_listening()

    combination = set(keyboard.HotKey.parse(SCRIBE_KEYBIND))
    current_keys = set()

    def on_p(key):
        try:
            k = listener.canonical(key)
        except Exception:
            return
        current_keys.add(k)
        if combination.issubset(current_keys):
            if not lemon.recorder.is_listening:
                threading.Thread(target=on_press_callback, daemon=True).start()

    def on_r(key):
        try:
            k = listener.canonical(key)
        except Exception:
            return
        if k in current_keys:
            current_keys.remove(k)
        
        if not combination.issubset(current_keys):
            if lemon.recorder.is_listening:
                on_release_callback()

    listener = keyboard.Listener(on_press=on_p, on_release=on_r)
    listener.start()
    logger.info(f"scribe active. press {SCRIBE_KEYBIND} to record. press ctrl+c to exit.")
    
    try:
        while listener.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("shutting down...")
        lemon.stop_listening()
        listener.stop()

if __name__ == "__main__":
    main()
