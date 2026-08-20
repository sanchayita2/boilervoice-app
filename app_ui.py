import os
import time
import chromadb
import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai.errors import ServerError

# Load environment variables from .env
load_dotenv(find_dotenv())

# Initialize API clients
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Custom ChromaDB Embedding Class using Google GenAI SDK
# Custom ChromaDB Embedding Class using Google GenAI SDK
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        response = genai_client.models.embed_content(
            model="text-embedding-004",  # Removed models/ prefix
            contents=input,
        )
        return [e.values for e in response.embeddings]

google_ef = GeminiEmbeddingFunction()

# Page Configuration
st.set_page_config(
    page_title="BoilerVoice AI Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Animated Waveform Visualizer
st.markdown(
    """
    <style>
    .wave-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 60px;
        gap: 6px;
        background-color: #0e1117;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    .wave-bar {
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #ff4b4b, #ff8c00);
        border-radius: 3px;
        animation: wave 1.2s ease-in-out infinite;
    }
    .wave-bar:nth-child(2) { animation-delay: 0.1s; }
    .wave-bar:nth-child(3) { animation-delay: 0.2s; }
    .wave-bar:nth-child(4) { animation-delay: 0.3s; }
    .wave-bar:nth-child(5) { animation-delay: 0.4s; }
    .wave-bar:nth-child(6) { animation-delay: 0.5s; }
    .wave-bar:nth-child(7) { animation-delay: 0.6s; }

    @keyframes wave {
        0%, 100% { height: 15%; }
        50% { height: 100%; }
    }
    </style>
""",
    unsafe_allow_html=True,
)


def ensure_csv_indexed_in_chroma(csv_filename: str = "boiler_inspection_data.csv"):
    """Vectorizes CSV rows into ChromaDB using custom Gemini API embeddings."""
    if not os.path.exists(csv_filename):
        return None

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
    name="boiler_telemetry",
    embedding_function=google_ef
)
    

    # Only index if collection is currently empty
    if collection.count() == 0:
        df = pd.read_csv(csv_filename)
        documents, metadatas, ids = [], [], []

        for idx, row in df.iterrows():
            row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(row_str)
            metadatas.append({"row_index": idx})
            ids.append(f"telemetry_row_{idx}")

        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return collection


def transcribe_audio_bytes(audio_bytes) -> str:
    """Sends recorded audio bytes directly to ElevenLabs Scribe STT."""
    temp_input = "temp_web_input.wav"
    with open(temp_input, "wb") as f:
        f.write(audio_bytes)

    with open(temp_input, "rb") as audio_file:
        transcription = elevenlabs_client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2",
            tag_audio_events=False
        )
    return transcription.text.strip()


def query_llm_vector_rag(
    transcribed_text: str, csv_filename: str = "boiler_inspection_data.csv"
) -> str:
    """Queries Gemini using ChromaDB vector retrieval with API embeddings."""
    collection = ensure_csv_indexed_in_chroma(csv_filename)
    if collection is None:
        return f"Error: Dataset '{csv_filename}' not found."

    # Retrieve top 3 relevant records from ChromaDB
    results = collection.query(query_texts=[transcribed_text], n_results=3)
    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_context = "\n".join(retrieved_docs)

    system_instruction = (
        "You are an industrial boiler inspection assistant. "
        "Answer the user's question concisely in 1-2 natural sentences based strictly on the retrieved dataset records."
    )
    full_prompt = (
        f"RETRIEVED DATASET RECORDS:\n{retrieved_context}\n\nUSER QUESTION: {transcribed_text}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config={"system_instruction": system_instruction},
            )
            return response.text.strip()

        except ServerError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "⚠️ The AI service is currently busy (503). Please wait a few seconds and try again."
        except Exception as e:
            return f"Error: {e}"


def generate_tts_file(
    text: str, output_path: str = "response.mp3", voice_id="pNInz6obpgDQGcFmaJgB"
) -> str:
    """Generates audio using ElevenLabs TTS and saves it for browser playback."""
    audio = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return output_path


# --- UI LAYOUT ---
st.title("⚡ BoilerVoice AI Dashboard")
st.caption("Live Industrial Inspection Dashboard with Voice-Activated Vector RAG Pipeline")

col_left, col_right = st.columns([1, 1], gap="large")

# Left Column: Live Data Viewer
with col_left:
    st.subheader("📊 Live CSV Telemetry Data")
    csv_file = "boiler_inspection_data.csv"

    if os.path.exists(csv_file):
        df_data = pd.read_csv(csv_file)
        st.dataframe(df_data.head(100), use_container_width=True, height=400)
        st.caption(f"Showing preview of top rows from total {len(df_data)} records.")
    else:
        st.error(f"❌ Dataset file '{csv_file}' not found in current folder.")

# Right Column: Voice Control & Real-Time Interaction
with col_right:
    st.subheader("🎙️ Voice Assistant Control")

    audio_data = st.audio_input("Click the microphone to record your question")

    if audio_data is None:
        st.info("👈 Click the microphone icon above and speak your question.")
    else:
        audio_bytes = audio_data.read()

        if not audio_bytes:
            st.warning("⚠️ No audio data recorded. Please try speaking again.")
        else:
            # Step 1: Speech to Text
            with st.spinner("🎧 Transcribing audio..."):
                try:
                    transcribed_text = transcribe_audio_bytes(audio_bytes)
                except Exception as e:
                    transcribed_text = ""
                    st.error(f"Transcription error: {e}")

            st.markdown("**🗣️ Transcribed Text:**")
            if transcribed_text:
                st.info(transcribed_text)

                # Step 2: Query Gemini with Vector Retrieval
                with st.spinner("🤖 Searching ChromaDB & Querying Gemini AI..."):
                    llm_answer = query_llm_vector_rag(transcribed_text, csv_filename=csv_file)

                st.markdown("**💡 Gemini Response:**")
                st.success(llm_answer)

                # Step 3: Text to Speech & Playback
                if not llm_answer.startswith("⚠️") and not llm_answer.startswith("Error"):
                    with st.spinner("🔊 Generating audio response..."):
                        try:
                            audio_output_path = generate_tts_file(llm_answer)

                            st.markdown(
                                """
                                <div class="wave-container">
                                    <div class="wave-bar"></div>
                                    <div class="wave-bar"></div>
                                    <div class="wave-bar"></div>
                                    <div class="wave-bar"></div>
                                    <div class="wave-bar"></div>
                                    <div class="wave-bar"></div>
                                    <div class="wave-bar"></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.audio(audio_output_path, format="audio/mp3", autoplay=True)
                        except Exception as e:
                            st.error(f"Audio playback error: {e}")
            else:
                st.warning("⚠️ Could not recognize any speech. Please speak louder or check microphone permissions.")