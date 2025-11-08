# Pranav — AI Voice Interview Bot (Gradio + OpenAI)

> Demo URL: (replace this with your Hugging Face Space URL after deployment)  
> https://huggingface.co/spaces/<your-username>/pranav-voicebot-demo

## What this project does
A simple voice-enabled interview bot that:
1. Records a short voice question from the browser (microphone or upload).  
2. Transcribes audio using OpenAI Whisper.  
3. Generates a persona-driven response using ChatGPT.  
4. Plays the reply audio (gTTS) and shows the transcript + text reply.

This demo is designed for the 100x Stage 1 assessment (voice answers to interview questions).

## How it works (quick)
- Frontend: Gradio UI handles recording and file upload.
- Backend: `app.py` transcribes with `openai.Audio.transcribe(...)`, then calls `openai.ChatCompletion.create(...)` with a server-side persona prompt.
- TTS: gTTS converts the assistant text to an MP3 for playback.
- Secrets: the OpenAI API key must be set in environment variables (locally via `.env` or PowerShell/CMD). On Hugging Face Spaces set `OPENAI_API_KEY` in Space secrets.

## Run locally (recommended)
1. Create & activate a Python venv:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate
   # macOS / Linux
   source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Create a .env file (local only — do NOT commit this) and add your OpenAI key:

OPENAI_API_KEY=sk-...


Run the app:

python app.py


Open http://localhost:7860 in your browser and test the mic.

Privacy note

For the demo, audio is processed server-side to produce a transcript and response. Do not commit or publish your API key.

In deployment (Hugging Face Spaces), set OPENAI_API_KEY via the Space Secrets setting so the key is never visible to users.

Troubleshooting

If you see OPENAI_API_KEY not found — set the env var (PowerShell: $env:OPENAI_API_KEY="sk-..." or use .env + python-dotenv).

If Gradio audio component errors, upgrade Gradio: pip install --upgrade "gradio>=3.40".

If soundfile install fails, try removing it temporarily and rely on Gradio-provided file paths.

Submission

Deploy the repo to Hugging Face Spaces (Gradio) and add OPENAI_API_KEY in Secrets.

Paste the public Space URL in your submission email to bhumika@100x.inc. Use subject:
GEN AI: GEN AI ROUND 1 ASSESSMENT (LINKEDIN) — Pranav

Contact

Pranav — [your-email@example.com
]
(Replace contact details above before submitting)


---

## 2 — Create files & commit (copy-paste commands)

From your project root (where `app.py` already exists):

```bash
# create files quickly from terminal (Linux/macOS) - Windows PowerShell adjust or create via editor
echo "gradio>=3.40
openai>=0.27.0
gtts>=2.3.0
soundfile>=0.12.1
python-dotenv>=0.21.0" > requirements.txt

echo ".venv/
__pycache__/
*.pyc
.env
*.env
*.mp3
*.wav
*.tmp
*.m4a
recorded_*
.vscode/
.idea/
.DS_Store
Thumbs.db" > .gitignore

# create README.md using a text editor, or paste the content into README.md

# init git / commit
git init
git add .
git commit -m "Initial commit: AI Voice Interview Bot (app + requirements + README + .gitignore)"


If you already initialized git earlier, just:

git add requirements.txt .gitignore README.md
git commit -m "Add requirements, README, and .gitignore"


Then push to GitHub (replace URL):

git remote add origin https://github.com/<your-username>/ai-voicebot.git
git branch -M main
git push -u origin main