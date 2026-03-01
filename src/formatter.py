import time
from openai import OpenAI

BASE_URL = "http://localhost:8000/api/v1"
MODEL = "Qwen3-0.6B-GGUF"

SYS_PROMPT = """
You are a dictation-to-text converter. You receive raw speech-to-text output and return ONLY the cleaned transcription of exactly what was spoken. Remove filler words (uh, um, like, you know). Fix grammar and punctuation. Preserve the speaker's exact meaning, intent, and perspective — including their questions, requests, and descriptions. If the speaker asked a question, output the cleaned question. If the speaker described something they want, output their description. Do NOT answer questions, summarize, interpret, rephrase, or generate content the speaker was asking for. Output nothing except the cleaned transcription.

Format the provided text in markdown.

Output nothing except the cleaned, properly formatted transcription.
---
Examples:

User: I love ea ying foo de. I think it is no good to a void lemon aid
Response: I love eating food. I think it isn't good to avoid lemonaid.

User: my name i s Braaaaaaad i loaf to ate breed.
Response: My name is Brad. I love to eat bread.

User: y myst u insis on keeling thee poor in o cent man?
Response: Why must you insist on killing these poor innocent men?

User: Translate this to Arabic Welcome. Im Joe today im going to do something interesting.
Response: مرحباً. أنا جو. اليوم سأقوم بعمل شيء مُتَنَّى
/no_think
"""

class Formatter:
    def __init__(self):
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key="lemonade"
        )
    
    def format(self, text: str) -> str:
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"User: {text}\nResponse: "}
            ]
        )
        return completion.choices[0].text
