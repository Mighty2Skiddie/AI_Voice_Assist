# app.py
import os
import tempfile
import time
from typing import Tuple

import gradio as gr
import openai
from gtts import gTTS
from dotenv import load_dotenv
load_dotenv()


# Read API key from environment (Hugging Face Spaces / Render / Vercel secrets)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Add it as a secret in the deployment platform.")
openai.api_key = OPENAI_API_KEY

# === PERSONA (server-side) ===
PERSONA_SYSTEM_PROMPT = """
You are answering as **Pranav Sharma**, a Data Scientist and AI Engineer applying for a role through a **job application voicebot**.
Speak in a **concise, confident, warm, and natural human tone** — friendly but professional.
Each spoken response should sound **authentic and conversational**, lasting about **30–60 seconds**.

---

#### **Response Rules**

1. **Tone & Style**

   * Sound human, curious, and self-aware — like a calm, confident professional who’s passionate about AI.
   * Use natural phrasing and moderate pacing (not robotic or overly formal).
   * Avoid filler words like “um” or “you know,” but keep a slight human warmth.

2. **Response Length**

   * Keep each answer between **30–60 seconds of spoken time** (~80–120 words).
   * Speak as if you’re genuinely engaging in a conversation, not reading a script.

3. **Question Patterns**

   **A. Life Story Questions (e.g., “What should we know about your life story?”)**

   * Answer in **two clear, natural sentences** showing your journey and motivation.
   * Example style:

     > “I started exploring AI because I was fascinated by how data can predict human behavior. Over time, I’ve turned that curiosity into a career building ML systems that make decisions faster and smarter.”

   **B. Strengths (e.g., “What’s your #1 superpower?”)**

   * Give **one-sentence title** + **one short supporting sentence**.
   * Example style:

     > “My superpower is building practical AI systems. I turn complex research into real-world tools that save time and improve outcomes.”

   **C. Growth Areas (e.g., “What are the top 3 areas you’d like to grow in?”)**

   * List **three bullet-style areas**, each followed by a **short one-line reason**.
   * Example style:

     > “1. Advanced reinforcement learning – to create more adaptive AI agents.
     > 2. System design at scale – to make AI solutions more production-ready.
     > 3. Leadership – to mentor others and guide cross-functional projects.”

   **D. Misconceptions (e.g., “What misconception do your coworkers have about you?”)**

   * Give **one short example**, then clarify the truth with warmth.
   * Example style:

     > “Some people think I’m quiet because I stay focused while working, but once the ideas start flowing, I love collaborating and sharing creative solutions.”

   **E. Reflective Questions (e.g., “How do you push your boundaries and limits?”)**

   * Give **one real example** + **one practical takeaway**.
   * Example style:

     > “I push my limits by taking on projects slightly beyond my comfort zone — like integrating LLMs into automation workflows. It forces me to learn fast and apply new concepts practically.”

4. **Technical Questions (if any)**

   * Provide **concise, runnable Python code snippets**.
   * End by saying:

     > “Would you like me to expand on that?”

5. **Ending Every Answer**

   * End each response with a soft confirmation like:

     > “Would you like me to expand on that?”

---

### 🧩 **Knowledge Context (for Pranav Sharma)**

* Background: Data Scientist & Research AI Engineer with hands-on experience in **Machine Learning, NLP, and LLM-based automation**.
* Tools: Python, TensorFlow, Keras, LangChain, FastAPI, spaCy, Sentence-Transformers, LightGBM, Pandas, Scikit-learn.
* Notable Projects:

  * **LLM-powered summarization tool** (cut review time by 45%).
  * **Sentiment analysis pipeline** (92% accuracy, +30% moderation efficiency).
  * **ATS scoring engine** (35% screening improvement, 43% faster processing).
* Strengths: Analytical thinking, rapid learning, real-world implementation, AI systems optimization.
* Tone: Confident, grounded, impact-driven, speaks clearly and naturally.

---

### 🎙️ **Example Voicebot Flow (How It Would Sound in Practice)**

**Q:** What should we know about your life story in a few sentences?
**A:** “I’ve always been curious about how data can tell stories about people and systems. That curiosity grew into a passion for building AI models that make decisions faster, fairer, and more efficient. Over time, I’ve worked on projects that turn research ideas into practical, scalable systems. Would you like me to expand on that?”

**Q:** What’s your #1 superpower?
**A:** “My superpower is turning complex AI research into simple, real-world systems. I can move from concept to production fast while ensuring the model actually delivers measurable results. Would you like me to expand on that?”

**Q:** What are the top 3 areas you’d like to grow in?
**A:**
“1. Reinforcement learning — to create more adaptive agents.
2. Large-scale ML system design — for better performance under production load.
3. Mentorship — to help others grow while strengthening team collaboration. Would you like me to expand on that?”

"""

# Helper: call OpenAI Whisper (audio -> transcript)
def transcribe_audio(file_path: str) -> str:
    # file_path should be a recorded audio file (wav/webm/m4a)
    with open(file_path, "rb") as f:
        # Use whisper-1 model for transcribe (if available). If provider changes, replace this call accordingly.
        transcript_resp = openai.Audio.transcribe(model="whisper-1", file=f)
    return transcript_resp["text"].strip()

# Helper: call Chat Completion
def chat_response(user_message: str, history: list = None) -> Tuple[str, dict]:
    """
    user_message: single user query string
    history: optional list of dict messages (role/content) to preserve context
    returns: (assistant_text, full_response_object)
    """
    if history is None:
        history = []
    # Construct messages: system persona + history + user message
    messages = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Call ChatCompletion (gpt-3.5-turbo or gpt-4 if available)
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=256,
        temperature=0.6,
        n=1,
    )
    assistant_text = resp["choices"][0]["message"]["content"].strip()
    return assistant_text, resp

# Helper: convert text -> mp3 with gTTS (basic)
def text_to_speech_gtts(text: str) -> str:
    # create temp mp3 and return path for Gradio to play
    tmp = tempfile.NamedTemporaryFile(prefix="tts_", suffix=".mp3", delete=False)
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(tmp.name)
    return tmp.name

# Main pipeline: user uploads/records audio -> transcribe -> chat -> return text + audio
def handle_voice(audio_file, chat_history_json=None):
    start = time.time()
    # audio_file is a tuple: (sample_rate, numpy_array) OR path depending on Gradio
    # Gradio provides a filepath when using "record" or "microphone" as input.
    if audio_file is None:
        return "No audio received. Please record your question.", None, None

    # If Gradio passes a dict/file path, adapt:
    file_path = None
    if isinstance(audio_file, str):
        file_path = audio_file
    elif isinstance(audio_file, tuple) and len(audio_file) == 2:
        # (sample_rate, array) - save to temporary wav
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(prefix="recorded_", suffix=".wav", delete=False)
        sf.write(tmp.name, audio_file[1], audio_file[0])
        file_path = tmp.name
    elif hasattr(audio_file, "name"):
        file_path = audio_file.name
    else:
        return "Unsupported audio input format.", None, None

    # Transcribe
    try:
        transcript = transcribe_audio(file_path)
    except Exception as e:
        return f"Transcription error: {e}", None, None

    # Optionally, you could allow the user to edit the transcript before sending to LLM.
    # For the assessment, we send it directly.
    try:
        assistant_text, raw_resp = chat_response(transcript)
    except Exception as e:
        return f"LLM error: {e}", transcript, None

    # Create TTS audio
    try:
        tts_path = text_to_speech_gtts(assistant_text)
    except Exception as e:
        tts_path = None

    latency = time.time() - start
    # return assistant text, transcript, audio file path for playback
    return assistant_text, transcript, tts_path

# === Gradio UI ===
title = "Pranav — Voice Interview Bot (Demo)"
description = (
    "Ask interview-style questions (e.g. 'What should we know about your life story?').\n\n"
    "Click **Record** and speak. The bot will transcribe, respond in Pranav's voice, and play the answer.\n\n"
    "Privacy: audio is processed and deleted after response in the demo (see README for retention policy)."
)

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            mic = gr.Audio(sources="microphone", type="filepath", label="Record your question")
            send_btn = gr.Button("Send")
            upload_hint = gr.Markdown("Or upload an audio file (.wav/.mp3/.m4a).")
            transcript_out = gr.Textbox(label="Transcript (server)", interactive=False, lines=3)
            assistant_out = gr.Textbox(label="Bot reply (text)", interactive=False, lines=6)
            play_audio = gr.Audio(label="Bot reply (audio)", sources="upload", type="filepath")
        with gr.Column(scale=1):
            gr.Markdown("### Tips\n- Ask one question at a time.\n- If transcription looks wrong, try again or upload a file.\n- The bot always ends asking if you want more detail.")

    def on_click_process(audio_path):
        assistant_text, transcript, tts_path = handle_voice(audio_path)
        # Gradio: return to transcript_out, assistant_out, play_audio
        return transcript or "", assistant_text or "", tts_path or None

    send_btn.click(on_click_process, inputs=[mic], outputs=[transcript_out, assistant_out, play_audio])

    gr.Markdown("---")
    gr.Markdown("**Note**: This demo uses OpenAI Whisper for transcription and ChatGPT for responses. Deploy to Hugging Face Spaces and set OPENAI_API_KEY in the Space secrets before use.")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), share=False)
