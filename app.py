import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# For Google GenAI SDK (Image Generation)
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os

# Initialize LangChain chat model for text
llm_text = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize Google GenAI client for image generation
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 Gemini Chatbot with Image Generation")

# Chat input
prompt = st.chat_input("Say something or request an image...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if "generate image:" in prompt.lower():
        image_prompt = prompt.lower().replace("generate image:", "").strip()
        with st.spinner("Generating image..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=image_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )
                text_response = ""
                image_data = None

                for part in response.candidates[0].content.parts:
                    if part.text:
                        text_response += part.text
                    elif part.inline_data:
                        image = Image.open(BytesIO(part.inline_data.data))
                        image_data = image

                # Store only text response in history, not image data
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": text_response, "is_image": True, "image_prompt": image_prompt}
                )
                # Display image immediately without storing in history
                st.session_state.chat_history.append({"role": "assistant", "image_data": image_data})

            except Exception as e:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Image generation failed: {str(e)}"}
                )

    else:
        with st.spinner("Thinking..."):
            try:
                response = llm_text.invoke(
                    [HumanMessage(content=prompt)]
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response.content}
                )
            except Exception as e:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Chat generation failed: {str(e)}"}
                )

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        if "image_data" in msg:
            st.chat_message("assistant").image(msg["image_data"], caption="Generated Image")
        else:
            st.chat_message("assistant").write(msg["content"])
