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
    st.error("GOOGLE_API_KEY not found in Streamlit secrets. Please configure it.")
    st.stop()

# Initialize the text-based LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Initialize Google Generative AI client for image generation
client = genai.GenerativeModel(model_name="gemini-2.0-flash-preview-image-generation")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# Streamlit app title
st.title("Chat with AI Assistant")

# Function to check if the input is an image generation request
def is_image_request(user_input):
    image_keywords = r"(make a photo|generate an image|create a picture|draw|image of|picture of)"
    return bool(re.search(image_keywords, user_input, re.IGNORECASE))

# Function to generate and encode image
def generate_image(prompt):
    try:
        response = client.generate_content(
            prompt,
            generation_config={"response_mime_type": "image/png"}
        )
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = part.inline_data.data
                image = Image.open(BytesIO(image_data))
                # Convert image to base64 for Streamlit display and storage
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return image, img_str
        return None, None
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            if hasattr(message, 'image_data') and message.image_data:
                st.image(base64.b64decode(message.image_data), caption="Generated Image")
            if message.content:
                st.write(message.content)

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
            st.write(user_input)
        
        # Check if the request is for image generation
        if is_image_request(user_input):
            with st.chat_message("assistant"):
                image, img_str = generate_image(user_input)
                if image:
                    st.image(image, caption="Generated Image")
                    # Store image in chat history as base64
                    st.session_state.chat_history.append(AIMessage(content="", image_data=img_str))
                else:
                    error_msg = img_str or "Failed to generate image."
                    st.write(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))
        else:
            # Get text-based AI response
            with st.chat_message("assistant"):
                try:
                    result = llm.invoke(st.session_state.chat_history)
                    st.write(result.content)
                    st.session_state.chat_history.append(AIMessage(content=result.content))
                except Exception as e:
                    error_msg = f"Error processing text request: {str(e)}"
                    st.write(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))
