import csv
import os
import sys
import time
from datetime import datetime
import pandas as pd
import sounddevice as sd
import soundfile as sf
from dotenv import find_dotenv, load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from google import genai

# Load environment variables
load_dotenv(find_dotenv())

# Initialize API clients
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

LOG_FILE = "pipeline_execution_logs.csv"


def log_event(step: str, status: str, details: str):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Step", "Status", "Details"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, status, details])


def record_audio(output_filename="user_input.wav", duration=5, sample_rate=44100) -> str:
    print(f"\n🎙️ Recording for {duration} seconds... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    sf.write(output_filename, audio_data, sample_rate)
    return output_filename


def transcribe_audio(audio_filename="user_input.wav") -> str:
    print("⏳ Transcribing audio with ElevenLabs Scribe...")
    with open(audio_filename, "rb") as audio_file:
        transcription = elevenlabs_client.speech_to_text.convert(
            file=audio_file, 
            model_id="scribe_v2", 
            tag_audio_events=False
        )
    text = transcription.text.strip()
    print(f"🗣️ Transcribed Text: \"{text}\"")
    return text


def query_llm_with_csv(transcribed_text: str, csv_filename="boiler_inspection_data.csv") -> str:
    print("🤖 Querying Gemini LLM with CSV context...")
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Dataset '{csv_filename}' not found.")

    df = pd.read_csv(csv_filename)
    csv_context = df.to_csv(index=False)

    system_instruction = (
        "You are an industrial boiler inspection assistant. "
        "Answer the user's question concisely in 1-2 natural sentences based strictly on the provided CSV dataset."
    )
    full_prompt = f"CSV DATASET:\n{csv_context}\n\nUSER QUESTION: {transcribed_text}"

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=full_prompt, 
        config={"system_instruction": system_instruction}
    )
    answer = response.text.strip()
    print(f"💡 Gemini Answer: \"{answer}\"")
    return answer


def text_to_speech_and_play(text: str, voice_id="pNInz6obpgDQGcFmaJgB"):
    print("🔊 Playing response...")
    audio = elevenlabs_client.text_to_speech.convert(
        text=text, 
        voice_id=voice_id, 
        model_id="eleven_flash_v2_5", 
        output_format="mp3_44100_128"
    )
    play(audio)


def run_continuous_pipeline(csv_dataset="boiler_inspection_data.csv", record_seconds=5):
    print("==================================================")
    print("  CONTINUOUS VOICE ASSISTANT INITIALIZED          ")
    print("  Press Ctrl + C at any time to exit the program. ")
    print("==================================================")

    session_count = 0

    while True:
        try:
            session_count += 1
            print(f"\n--- Cycle #{session_count} ---")
            
            # Optional press-enter trigger (or remove input prompt for automatic looping)
            input("Press [ENTER] when ready to ask a question...")

            # Step 1: Record
            audio_file = record_audio(duration=record_seconds)
            log_event("Step 1: Record Audio", "SUCCESS", f"File saved: {audio_file}")

            # Step 2: Transcribe
            user_query = transcribe_audio(audio_file)
            log_event("Step 2: Speech-to-Text", "SUCCESS", f"Transcript: {user_query}")

            if not user_query:
                print("⚠️ No speech detected. Retrying...")
                continue

            # Step 3: Query Gemini
            llm_response = query_llm_with_csv(user_query, csv_filename=csv_dataset)
            log_event("Step 3: Gemini Query", "SUCCESS", f"Response: {llm_response}")

            # Step 4: TTS & Playback
            text_to_speech_and_play(llm_response)
            log_event("Step 4: TTS Playback", "SUCCESS", "Completed.")

            time.sleep(1)  # Brief pause before next cycle

        except KeyboardInterrupt:
            print("\n\n🛑 Continuous loop stopped by user. Goodbye!")
            sys.exit(0)

        except Exception as e:
            print(f"❌ Error during cycle #{session_count}: {e}")
            log_event("Pipeline Loop", "ERROR", str(e))
            print("Restarting cycle in 3 seconds...")
            time.sleep(3)


if __name__ == "__main__":
    run_continuous_pipeline(csv_dataset="boiler_inspection_data.csv", record_seconds=5)