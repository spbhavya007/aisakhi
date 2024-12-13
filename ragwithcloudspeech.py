from google.cloud import speech
from google.cloud import texttospeech
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import streamlit as st
import tempfile
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io
import base64
import os
# from streamlit_mic_recorder import mic_recorder
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from st_audiorec import st_audiorec
from pydub import AudioSegment


# Google Cloud credentials

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "venv/aisakhi-4d474f72023e.json"
# Save the credentials to a file (needed for the Google client libraries)

# Access the Google credentials from Streamlit secrets
google_credentials = st.secrets["google"]["GOOGLE_APPLICATION_CREDENTIALS"]
with open("google-credentials.json", "w") as f:
    f.write(google_credentials)

# Set GOOGLE_APPLICATION_CREDENTIALS environment variable to the file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-credentials.json"

# Function to record audio (for community cloud deployment)
# Custom audio processor to capture audio
#class AudioProcessor(AudioProcessorBase):
#    def __init__(self):
#        self.audio_data = None
#
#    def recv(self, frame):
#        # Process audio frame
#        self.audio_data = frame.to_ndarray()  # Store audio data for later use
#        return frame

# Function to record audio (for local deployment)
def record_audio(duration=5, samplerate=44100):
    print("Recording... Speak now!")
    st.info("Listening... Speak now.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    return np.squeeze(audio)

# Function to transcribe speech using Google Cloud Speech-to-Text
def transcribe_audio(audio_data, samplerate=44100):
    client = speech.SpeechClient()
    #wav_io = io.BytesIO()
    #write(wav_io, samplerate, audio_data)
    #wav_io.seek(0)
    
    # Convert byte data to an AudioSegment
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    # Convert to mono
    audio_segment = audio_segment.set_channels(1)
    # Export the mono audio as a byte stream
    mono_audio = io.BytesIO()
    audio_segment.export(mono_audio, format="wav")
    mono_audio.seek(0)

    #audio = speech.RecognitionAudio(content=wav_io.read())
    #audio = speech.RecognitionAudio(content=audio_data)
    audio = speech.RecognitionAudio(content=mono_audio.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=samplerate,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)
    
    if response.results:
        return response.results[0].alternatives[0].transcript
    else:
        st.error("No speech detected or transcription failed.")
        return ""


# Function to generate speech using Google Cloud Text-to-Speech
def synthesize_speech(text):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return response.audio_content

# my_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
my_api_key = st.secrets["openai"]["API_KEY"]

if my_api_key:
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=my_api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=my_api_key)
    vector_store = InMemoryVectorStore(embeddings)

# Initialize an empty list to store the uploaded files
uploaded_files = []

def load_files():
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                all_splits = text_splitter.split_documents(docs)
                vector_store.add_documents(documents=all_splits)
        except Exception as e:
            st.error(f"Error processing file '{uploaded_file.name}': {str(e)}")

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

st.title("AI Sakhi")
st.markdown("## For healthy, thriving women")
url = "https://www.aisakhi.org"
st.markdown(f"More information at [www.aisakhi.org]({url})")
st.image("./image.jpg")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    load_files()

st.markdown("##### Documents in Collection")
if uploaded_files:
    for file in uploaded_files:
        st.write(file.name)
else:
    st.markdown("##### No documents in the collection yet.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamlit WebRTC to capture audio
# audio_processor = AudioProcessor()

#webrtc_streamer(
#    key="audio-capture",
#    audio_processor_factory=lambda: audio_processor,
#    media_stream_constraints={"audio": True, "video": False}  # Only audio
#)

#if st.button("🎤 Speak"):
#if audio_processor.audio_data is not None:

wav_audio_data = st_audiorec() # tadaaaa! yes, that's it! :D
if wav_audio_data is not None:

    #audio = record_audio()

    #print("audio processor got something!")
    #user_prompt = transcribe_audio(audio_processor.audio_data)
    #print("prompt was: "+user_prompt)

    user_prompt = transcribe_audio(wav_audio_data)
    print("prompt was: "+user_prompt)

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    response = graph.invoke({"question": user_prompt})
    assistant_response = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    audio_content = synthesize_speech(assistant_response)
    # st.audio(audio_content, format="audio/mp3")

    # Embed the audio content as base64 and use autoplay in the HTML
    audio_base64 = base64.b64encode(audio_content).decode('utf-8')
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

if user_prompt := st.chat_input("Ask a question specific to women's health"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    response = graph.invoke({"question": user_prompt})
    assistant_response = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

