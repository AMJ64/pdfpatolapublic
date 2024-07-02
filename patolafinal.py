import os
import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from io import BytesIO
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import uuid

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to retrieve relevant text from the PDF content using TF-IDF
def retrieve_relevant_text(pdf_text, query):
    # Split the PDF text into chunks (e.g., paragraphs)
    chunks = pdf_text.split('\n\n')
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit_transform(chunks + [query])
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity between the query and the chunks
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()
    
    # Get the most relevant chunk
    most_relevant_index = np.argmax(cosine_similarities)
    relevant_text = chunks[most_relevant_index]
    
    return relevant_text

# Function to save chat history to a file
def save_chat_history(chat_history, session_id):
    filename = f"chat_history_{session_id}.json"
    with open(filename, "w") as file:
        json.dump(chat_history, file)

# Function to load chat history from a file
def load_chat_history(session_id):
    filename = f"chat_history_{session_id}.json"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return json.load(file)
    return []

# Function to clear chat history
def clear_chat_history(session_id):
    filename = f"chat_history_{session_id}.json"
    if os.path.exists(filename):
        os.remove(filename)

# Function to transcribe audio using Groq

def transcribe_audio(file, language):
    transcription = client.audio.transcriptions.create(
        file=(file.name, file.read()),
        model="whisper-large-v3",
        prompt="Specify context or spelling",  # Optional
        response_format="json",  # Optional
        language=language,  # Optional
        temperature=0.2 # Optional
    )
    return transcription.text

# Get the working directory every time we run the file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Setting page configuration
st.set_page_config(
    page_icon="✨",
    page_title="PDF Patola",
    layout="centered"
)

# Initialize session state for selected model
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3-70b-8192"

with st.sidebar:
    selected = option_menu(
        menu_title="Choose Functionality",
        options=["PDF Reader", "Speech Recognition"],
        # menu_icon="robot",
        icons=["file-earmark-pdf", "mic"],
        default_index=0
    )

# Model Selection Section (Always Visible)
st.sidebar.subheader("Select Model for PDF Reader")
model_options = ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
selected_model = st.sidebar.selectbox("Choose a model", model_options, index=model_options.index(st.session_state.selected_model))

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.experimental_rerun()

# PDF Reader Page
if selected == "PDF Reader":
    st.title("PDF Patola")
    st.write("Chai Piyoge ☕????")
    st.write('Pro tip - always use "clean chat history"')

    # Generate a unique session ID for each user
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    session_id = st.session_state.session_id

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Clear chat history, delete JSON file, and clear session state when a new PDF is uploaded
        if "uploaded_file" in st.session_state and st.session_state.uploaded_file != uploaded_file:
            if "chat_history" in st.session_state:
                del st.session_state["chat_history"]
            clear_chat_history(session_id)
            st.session_state.clear()  # Clear all session state data
            st.session_state.session_id = session_id  # Retain the session ID
        
        st.session_state.uploaded_file = uploaded_file

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        # st.write("Extracted Text:")
        # st.write(pdf_text)

        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = load_chat_history(session_id)

        # Display chat history
        st.markdown(
            """
            <style>
            .chat-message {
                display: flex;
                align-items: center;
                margin: 10px 0;
            }
            .chat-message.user .message {
                background-color: #dcf8c6;
                align-self: flex-end;
            }
            .chat-message.assistant .message {
                background-color: #f1f0f0;
                align-self: flex-start;
            }
            .message {
                padding: 10px;
                border-radius: 10px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .arrow {
                width: 0;
                height: 0;
                border-style: solid;
            }
            .user .arrow {
                border-width: 0 10px 10px 0;
                border-color: transparent #dcf8c6 transparent transparent;
                margin-left: 10px;
            }
            .assistant .arrow {
                border-width: 10px 10px 0 0;
                border-color: #f1f0f0 transparent transparent transparent;
                margin-right: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="chat-message user"><div class="message">{message}</div><div class="arrow user"></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant"><div class="arrow assistant"></div><div class="message">{message}</div></div>', unsafe_allow_html=True)

        # Input box at the bottom with embedded send button
        user_input = st.chat_input("Ask a question about the PDF content:")
        if user_input:
            # Retrieve relevant text from the PDF content
            relevant_text = retrieve_relevant_text(pdf_text, user_input)
            
            # Generate response using the Groq client
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. The following is the relevant content of the PDF: " + relevant_text + " act like you are a pdf"
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
                model=st.session_state.selected_model,
            )
            response = chat_completion.choices[0].message.content
            
            # Update chat history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response))
            
            # Save chat history
            save_chat_history(st.session_state.chat_history, session_id)
            
            # Clear the input box
            st.experimental_rerun()

        # Clear chat history button
        if st.button("Clear Chat History"):
            if "chat_history" in st.session_state:
                del st.session_state["chat_history"]
            clear_chat_history(session_id)
            st.experimental_rerun()

# Speech Recognition Page
if selected == "Speech Recognition":
    st.title("Speech to Text 🎤")
    
    uploaded_audio = st.file_uploader("Upload an audio file..", type=["m4a", "mp3", "wav"])
    
    # Add a dropdown menu for language selection
    language = st.selectbox(
        "Select language for transcription:",
        ("en", "hi", "es", "fr", "de", "ja", "ru")
    )
    
    if st.button("Transcribe Audio"):
        if uploaded_audio is not None:
            transcription_text = transcribe_audio(uploaded_audio, language)
            st.write("Transcription:")
            st.write(transcription_text)
