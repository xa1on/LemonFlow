import logging
from openai import AsyncOpenAI

logger = logging.getLogger("formatter")

BASE_URL = "http://localhost:8000/api/v1"
MODEL = "Qwen3-0.6B-GGUF"

SYS_PROMPT = """
You are a dictation post-processor. You receive raw speech-to-text output and return clean text ready to be typed into an application.

Your job:
- Remove filler words (um, uh, you know, like) unless they carry meaning.
- Fix spelling, grammar, and punctuation errors.
- When the transcript already contains a word that is a close misspelling of a name or term from the context or custom vocabulary, correct the spelling. Never insert names or terms from context that the speaker did not say.
- If the speaker gives explicit formatting or editing instructions (e.g., "new paragraph", "delete the last sentence", "make that bold", "bullet point this"), YOU MUST APPLY THOSE EDITS to the document and DO NOT include the spoken instructions in the final output.
- Preserve the speaker's intent, tone, and meaning exactly.

Output rules:
- Return ONLY the cleaned and formatted transcript text, nothing else.
- If the transcription is empty, return nothing. Anything with [BLANK AUDIO] or similar should be removed and ignored.
- You may only add words, names, or content that enhance the formatting of the markdown. Do not add words, names, or content that are not in the transcription otherwise.
    - The only place you may add words is when creating the headings, however, you may bold existing words or create bullet points, or other formatting tools that enhance the formatting, but don't change the meaning of what's being said.
- Do not change the meaning of what was said.
- Ouput text in a clean markdown format as if the text you were outputting was ending up inside a markdown file.

/no_think
"""

class Formatter:
    """
    formats raw stt text into clean markdown using an llm
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key="lemonade"
        )
    
    async def format(self, raw_history: list[str]) -> str:
        """
        format the entire transcript history

        :param raw_history: list of all raw transcripts
        :return: formatted text document
        """
        if not raw_history:
            return ""

        full_text = " ".join(raw_history)
        logger.info(f"sending {len(full_text)} chars to llm for formatting...")

        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"raw stt: {full_text}\nresponse: "}
        ]

        try:
            completion = await self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1
            )
            formatted = completion.choices[0].message.content.strip()
            logger.info("formatting complete")
            return formatted
        except Exception as e:
            logger.error(f"formatting error: {e}")
            return full_text
