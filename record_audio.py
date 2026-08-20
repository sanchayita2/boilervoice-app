import numpy as np
import sounddevice as sd
import soundfile as sf


def record_audio(
    output_filename="user_input.wav",
    sample_rate=16000,
    energy_threshold=300,
    silence_limit_sec=1.5,
    max_duration_sec=30,
) -> str:
    """Captures audio dynamically using Voice Activity Detection (VAD).

    Starts recording when speech is detected and stops after a period of
    silence.
    """
    print("\n🎙️ Listening... Start speaking whenever you are ready.")

    chunk_size = 1024
    audio_frames = []
    speech_started = False
    silent_chunks = 0
    max_silent_chunks = int((sample_rate / chunk_size) * silence_limit_sec)
    max_total_chunks = int((sample_rate / chunk_size) * max_duration_sec)

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="int16"
    ) as stream:
        total_chunks = 0
        while True:
            data, _ = stream.read(chunk_size)
            audio_frames.append(data)
            total_chunks += 1

            # Calculate RMS energy to detect voice volume
            rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

            if rms > energy_threshold:
                if not speech_started:
                    print("🗣️ Speech detected! Recording...")
                    speech_started = True
                silent_chunks = 0
            elif speech_started:
                silent_chunks += 1
                if silent_chunks >= max_silent_chunks:
                    print("🛑 Silence detected. Stopping recording...")
                    break

            if total_chunks >= max_total_chunks:
                print("⏱️ Max recording time reached. Stopping...")
                break

    # Save final recording
    audio_data = np.concatenate(audio_frames, axis=0)
    sf.write(output_filename, audio_data, sample_rate)
    print(f"✅ Audio saved to {output_filename}")
    return output_filename


if __name__ == "__main__":
    record_audio()