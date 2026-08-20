import os
from dotenv import load_dotenv, find_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

# Load environment variables
load_dotenv(find_dotenv())

# Initialize ElevenLabs Client
api_key = os.getenv("ELEVENLABS_API_KEY")
elevenlabs_client = ElevenLabs(api_key=api_key)

def text_to_speech_and_play(text: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
    """
    Converts a text string to spoken audio using ElevenLabs TTS and plays it immediately.
    
    :param text: Text string returned by Gemini LLM.
    :param voice_id: ElevenLabs Voice ID (default: 'Adam').
    """
    if not text or not text.strip():
        raise ValueError("No text provided for text-to-speech conversion.")

    try:
        print(f"\n[TTS] Generating voice output with ElevenLabs...")

        # 1. Convert text to speech
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_flash_v2_5",  # Low-latency model optimized for real-time agents
            output_format="mp3_44100_128"
        )

        # 2. Play the generated audio stream
        print("[Playback] Playing audio response...")
        play(audio)
        print("✅ Playback finished.")

    except Exception as e:
        print(f"❌ Text-to-Speech playback failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Test Step 4 with sample text output from Gemini
    sample_llm_response = "Boiler number 3 has reported an abnormal temperature status."
    text_to_speech_and_play(sample_llm_response)