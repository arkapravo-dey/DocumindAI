import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
key = os.getenv("GROQ_API_KEY", "").strip()
if not key:
    raise SystemExit("GROQ_API_KEY is missing. Add it to .env first.")

client = Groq(api_key=key, timeout=30.0, max_retries=1)
response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    messages=[{"role": "user", "content": "Reply with: Groq connection successful."}],
    max_tokens=30,
    temperature=0,
)
print(response.choices[0].message.content)
