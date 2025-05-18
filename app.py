import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GenerativeModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os

# Set Google API Key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize the LLM for chat
chat_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize the Generative Model for image generation
image_model = GenerativeModel(model_name="gemini-pro-vision")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant capable of both text and image generation.")]

# Streamlit app title
st.title("Chat with AI Assistant (with Image Generation)")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            if hasattr(message.additional_kwargs, 'get') and message.additional_kwargs.get('image'):
                st.image(message.additional_kwargs['image'], caption="Generated Image")
                st.write(message.content)
            else:
                st.write(message.content)

# Input box for user message
user_input = st.chat_input("Type your message here (or ask for an image)...")

if user_input:
    if user_input.lower() == "quit":
        st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant capable of both text and image generation.")]
        st.experimental_rerun()
    else:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            if "image" in user_input.lower() or "picture" in user_input.lower() or "generate" in user_input.lower():
                try:
                    response = image_model.generate_content(user_input)
                    image_part = None
                    for part in response.parts:
                        if part.mime_type.startswith("image/"):
                            image_part = part.data
                            break

                    if image_part:
                        st.image(image_part, caption="Generated Image")
                        ai_response = AIMessage(content="Here is the image you requested.", additional_kwargs={'image': image_part})
                    else:
                        ai_response = AIMessage(content="I can generate images, but the response didn't contain one this time.")
                except Exception as e:
                    ai_response = AIMessage(content=f"Sorry, I encountered an error while generating the image: {e}")
            else:
                # Get AI response for text
                result = chat_llm.invoke(st.session_state.chat_history)
                ai_response = AIMessage(content=result.content)
                st.write(result.content)

            # Add AI response to chat history
            st.session_state.chat_history.append(ai_response)
