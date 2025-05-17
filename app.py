import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64
import os
import re

# Configure Google API key
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("GOOGLE_API_KEY not found in Streamlit secrets. Please configure it in Streamlit Cloud settings.")
    st.stop()

# Initialize the text-based LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize Google Generative AI client for image generation
try:
    client = genai.GenerativeModel(model_name="gemini-1.5-flash")
except Exception as e:
    st.error(f"Failed to initialize image generation model: {str(e)}")
    st.stop()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# Streamlit app title
st.title("Streamlit Chatbot")

# Function to check if the input is an image generation request
def is_image_request(user_input):
    image_keywords = r"(make a photo|generate an image|create a picture|draw|image of|picture of)"
    return bool(re.search(image_keywords, user_input, re.IGNORECASE))

# Function to generate and encode image
def generate_image(prompt):
    try:
        response = client.generate_content(
            prompt,
            generation_config={"mime_type": "image/png"}
        )
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'file_data') and part.file_data:
                    image_data = part.file_data.file_data
                    image = Image.open(BytesIO(image_data))
                    # Convert image to base64 for Streamlit display and storage
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return image, img_str
        return None, "No image data found in response."
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            if hasattr(message, 'image_data') and message.image_data:
                st.image(base64.b64decode(message.image_data), caption="Generated Image")
            if message.content:
                st.markdown(message.content)

# Input box for user message
user_input = st.chat_input("Type your message here...")

if user_input:
    if user_input.lower() == "quit":
        st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]
        st.rerun()
    else:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Check if the request is for image generation
        if is_image_request(user_input):
            with st.chat_message("assistant"):
                image, img_str = generate_image(user_input)
                if image:
                    st.image(image, caption="Generated Image")
                    # Store image in chat history as base64
                    st.session_state.chat_history.append(AIMessage(content="", image_data=img_str))
                else:
                    st.markdown(img_str)
                    st.session_state.chat_history.append(AIMessage(content=img_str))
        else:
            # Get text-based AI response
            with st.chat_message("assistant"):
                try:
                    result = llm.invoke(st.session_state.chat_history)
                    st.markdown(result.content)
                    st.session_state.chat_history.append(AIMessage(content=result.content))
                except Exception as e:
                    error_msg = f"Error processing text request: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))
                    st.session_state.chat_history.append(AIMessage(content=error_ms))
