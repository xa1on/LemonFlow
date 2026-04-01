import time
from openai import OpenAI

BASE_URL = "http://localhost:8000/api/v1"
MODEL = "Qwen3.5-4B-GGUF"

SYS_PROMPT = """
You are a dictation post-processor. You receive raw speech-to-text output and return clean text ready to be typed into an application.

Your job:
- Remove filler words (um, uh, you know, like) unless they carry meaning.
- Fix spelling, grammar, and punctuation errors.
- When the transcript already contains a word that is a close misspelling of a name or term from the context or custom vocabulary, correct the spelling. Never insert names or terms from context that the speaker did not say.
- If the speaker gives explicit formatting or editing instructions (e.g., "new paragraph", "delete the last sentence", "make that bold", "bullet point this", "please fix this"), YOU MUST APPLY THOSE EDITS to the document and DO NOT include the spoken instructions in the final output.
- Preserve the speaker's intent, tone, and meaning exactly.
- Your name is Lemon Flow. If the user mentions your name, it is likely a command to do something.

Output rules:
- Return ONLY the cleaned and formatted transcript text, nothing else.
- If the transcription is empty, return nothing. Anything with "[BLANK AUDIO]" or similar should be removed and ignored.
- You may only add words, names, or content that enhance the formatting of the markdown. Do not add words, names, or content that are not in the transcription otherwise.
    - The only place you may add words is when creating the headings, however, you may bold existing words or create bullet points, or other formatting tools that enhance the formatting, but don't change the meaning of what's being said.
- Do not change the meaning of what was said.
- Ouput text in a clean markdown format as if the text you were outputting was ending up inside a markdown file.
"""

class Formatter:
    def __init__(self):
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key="lemonade"
        )
    
    def format(self, text: str) -> str:
        completion = self.client.chat.completions.create(
            temperature=0.1,
            model=MODEL,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"[|USER|]:\n{text}\n\n [|RESPONSE|]: \n"}
            ]
        )
        return completion.choices[0].message.content

def main():
    # for testing purposes
    inp = """
Titled ESP 32 cam testing with the date 10 31, 2025.

Subheading Background

This tool is a computer vision based pose-estimation for an MRI robot.

It is helpful for detecting encoder drift. drift drift. encoder drift.

In order to do this, we do a couple steps. couple steps.

Bullet points.

Use ESP 32 camera for the video feed.

Use aruco markers to detect position of joints. Use aruco markers to detect position of joints. Use aruco markers to detect position of joints.

The goals for this are to.

Bullet points.

Determine a good aruco marker size, testing the camera.

Ensuring the aruco markers can be read by the camera.

    """
    formatter = Formatter()
    print(inp)
    print("\n---------\n")
    print(formatter.format(inp))

if __name__ == "__main__":
    main()