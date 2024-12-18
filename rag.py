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
import speech_recognition as sr
#import pyttsx3
#from google.cloud import texttospeech
import base64
from gtts import gTTS

from scipy.io.wavfile import write
import io
from io import BytesIO


# Function to capture and transcribe speech
def record_audio(duration=5, samplerate=44100):
    # Record audio using sounddevice
    print("Recording... Speak now!")
    st.info("Listening... Speak now.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    return np.squeeze(audio)

def process_audio(audio_data, samplerate=44100):
    # Convert NumPy audio data to AudioData for recognition."""
    # Save the audio as a WAV file in memory
    wav_io = io.BytesIO()
    write(wav_io, samplerate, audio_data)
    wav_io.seek(0)
    # Use SpeechRecognition to process the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        print("Recognizing...")
        audio = recognizer.record(source)
        return recognizer.recognize_google(audio)


def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.success("Processing your input...")
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            st.error("No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.error("Speech not recognized. Please speak clearly.")
        except sr.RequestError as e:
            st.error(f"Error with the speech recognition service: {e}")
    return ""

my_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

if my_api_key:
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=my_api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=my_api_key)
    vector_store = InMemoryVectorStore(embeddings)

# Initialize an empty list to store the uploaded files
uploaded_files = []

def load_files():
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                # Write the uploaded file to the temporary file
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
                # Load the PDF using the temporary file path
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                all_splits = text_splitter.split_documents(docs)
                # Index chunks
                _ = vector_store.add_documents(documents=all_splits)
        except Exception as e:
            st.error(f"Error processing file '{uploaded_file.name}': {str(e)}")


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

#response = graph.invoke({"question": "What are some chronic conditions in women ?"})
#print(response["answer"])

st.title("AI Sakhi")
st.markdown("## For healthy, thriving women")

url = "https://www.aisakhi.org"
#st.write("More information at [www.aisakhi.org](%s)" % url)
st.markdown("More information at [www.aisakhi.org](%s)" % url)

#st.markdown("*Streamlit* is **really** ***cool***.")
#st.markdown('''
#    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
#    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')
st.markdown(":tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

st.image("./image.jpg")

# Create two columns: one for the file uploader and one for other content
#col1, col2 = st.columns([1, 2]) 

# File upload section in the first column
st.markdown("##### Upload Domain Knowledge Documents")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Show uploaded files in the second column
if uploaded_files:
    load_files()

# Display the list of files in the collection
st.markdown("##### Documents in Collection")
if uploaded_files:
    st.markdown("##### The following documents are in the collection:")
    for file in uploaded_files:
        st.write(file.name)  # Display the names of uploaded files
else:
    st.markdown("##### No documents in the collection yet.")

# Display a summary of the collection
st.markdown("##### Collection Summary")
num_docs = len(uploaded_files)  # Number of documents in the collection
st.write(f"Total documents in the collection: {num_docs}")

# Initialize the text-to-speech engine
#tts_engine = pyttsx3.init()

# Initialize chat state in Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.button("🎤 Speak"):
    audio = record_audio()
    user_prompt = process_audio(audio)
    # st.success(f"You said: {user_input}")
    # user_prompt = get_speech_input()

    # Add user input to chat
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Process user input through the graph
    response = graph.invoke({"question": user_prompt})
    assistant_response = response["answer"]

    # Add assistant's response to chat
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    # Convert the bot's response to speech
    sound_file = BytesIO()
    tts = gTTS(assistant_response, lang='en')
    tts.write_to_fp(sound_file)
    # Convert audio to base64 for embedding in Streamlit
    audio_base64 = base64.b64encode(sound_file.getvalue()).decode("utf-8")
    audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

    #tts_engine.say(assistant_response)
    #tts_engine.runAndWait()


# Chat input from user
if user_prompt := st.chat_input("Ask a question specific to women's health"):
    # Add user input to chat
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Process user input through the graph
    response = graph.invoke({"question": user_prompt})
    assistant_response = response["answer"]

    # Add assistant's response to chat
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
