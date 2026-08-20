import os
from dotenv import load_dotenv, find_dotenv
from elevenlabs.client import ElevenLabs

# 1. Force load the .env file from the current working directory
load_dotenv(find_dotenv())

# 2. Get key from environment, or use your validated key as a fallback
api_key = os.getenv("ELEVENLABS_API_KEY") 

# 3. Initialize client explicitly with the key
elevenlabs_client = ElevenLabs(api_key=api_key)

def transcribe_audio(audio_filename="user_input.wav") -> str:
    """Sends recorded audio to ElevenLabs Scribe API for transcription."""
    if not os.path.exists(audio_filename):
        raise FileNotFoundError(f"File '{audio_filename}' not found.")

    try:
        print(f"\n[Transcribing] Processing {audio_filename} with ElevenLabs...")
        
        with open(audio_filename, "rb") as audio_file:
            transcription = elevenlabs_client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v2",
                tag_audio_events=False
            )

        text = transcription.text.strip()
        print(f"[Transcribed Text]: \"{text}\"")
        return text

    except Exception as e:
        print(f"[Error] Transcription failed: {str(e)}")
        raise

if __name__ == "__main__":
    transcribe_audio("user_input.wav")