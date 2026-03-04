import os
import time
import logging
import threading
import asyncio

from .recorder import Recorder
from .formatter import Formatter

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
        self.formatter = Formatter()
        
        self.unformatted_history: list[str] = []
        self.formatted_text: str = ""
        self.formatted_count: int = 0
        
        self.current_delta: str = ""
        self.base_text: str = ""
        self.delta_bytes_len: int = 0
        
        self.file_lock: asyncio.Lock | None = None
        self.format_tasks: set[asyncio.Task] = set()

    async def _rewrite_entire_file(self, output_file_path: str):
        """
        rebuild the entire document text and overwrite the file
        """
        parts = []
        if self.formatted_text:
            parts.append(self.formatted_text)
        
        for t in self.unformatted_history[self.formatted_count:]:
            if t:
                parts.append(t)
                
        self.base_text = "\n\n".join(parts)
        
        volatile_text = ""
        if self.current_delta:
            if self.base_text:
                volatile_text = "\n\n" + self.current_delta
            else:
                volatile_text = self.current_delta
                
        full_text = self.base_text + volatile_text
        content_bytes = full_text.encode('utf-8')
        
        with open(output_file_path, "wb") as f:
            f.write(content_bytes)
            
        self.delta_bytes_len = len(volatile_text.encode('utf-8'))

    async def start_listening(self, output_file_path: str):
        """
        start listening and writing to file

        :param output_file_path: path to output file
        """
        # lazy init lock to ensure it's in the right loop
        if self.file_lock is None:
            self.file_lock = asyncio.Lock()

        # reset state for new recording
        self.unformatted_history = []
        self.formatted_text = ""
        self.formatted_count = 0
        self.current_delta = ""
        self.base_text = ""
        self.delta_bytes_len = 0
        
        # create file if not exist and clear if already exists
        with open(output_file_path, "w", encoding="utf-8"):
            pass

        async def format_history_task(history_to_format: list[str], count: int):
            try:
                logger.info(f"formatting {count} transcripts in background...")
                formatted = await self.formatter.format(history_to_format)
                
                async with self.file_lock:
                    if count > self.formatted_count:
                        self.formatted_text = formatted
                        self.formatted_count = count
                        await self._rewrite_entire_file(output_file_path)
            except Exception as e:
                logger.error(f"background formatting task failed: {e}")

        async def transcript_write(txt: str):
            logger.info(f"finished section: {txt[:50]}...")
            
            async with self.file_lock:
                self.unformatted_history.append(txt)
                self.current_delta = ""
                await self._rewrite_entire_file(output_file_path)
            
            count = len(self.unformatted_history)
            history_copy = list(self.unformatted_history)
            task = asyncio.create_task(format_history_task(history_copy, count))
            self.format_tasks.add(task)
            task.add_done_callback(self.format_tasks.discard)
        
        async def delta_write(txt: str):
            async with self.file_lock:
                self.current_delta = txt
                
                volatile_text = ""
                if txt:
                    if self.base_text:
                        volatile_text = "\n\n" + txt
                    else:
                        volatile_text = txt
                        
                volatile_bytes = volatile_text.encode('utf-8')
                
                with open(output_file_path, "rb+") as f:
                    f.seek(-self.delta_bytes_len, os.SEEK_END)
                    f.write(volatile_bytes)
                    f.truncate()
                
                self.delta_bytes_len = len(volatile_bytes)

        await self.recorder.start_listening(delta_write, transcript_write)
        
        # after recording stops, wait for any pending formatting tasks
        if self.format_tasks:
            logger.info(f"waiting for {len(self.format_tasks)} remaining formatting tasks...")
            await asyncio.gather(*self.format_tasks, return_exceptions=True)
            logger.info("all formatting tasks finished")
    
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
    loop = asyncio.new_event_loop()
    
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=run_loop, daemon=True).start()

    def on_press_callback():
        asyncio.run_coroutine_threadsafe(
            lemon.start_listening("log.md"), 
            loop
        )

    def on_release_callback():
        lemon.stop_listening()

    combination = set(keyboard.HotKey.parse(SCRIBE_KEYBIND)) # setting up keybind
    current_keys = set()

    def on_p(key):
        try:
            k = listener.canonical(key)
        except Exception:
            return
        current_keys.add(k)
        if combination.issubset(current_keys):
            if not lemon.recorder.is_listening:
                on_press_callback()

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
        if lemon.recorder.is_listening:
            lemon.stop_listening()
        listener.stop()
        loop.stop()

if __name__ == "__main__":
    main()
