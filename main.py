import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# --- Check API Key ---
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set. Please define it in your .env file.")
    st.stop()

# --- Initialize external client ---
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- Set up the model ---
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# --- Define the agent ---
translator_agent = Agent(
    name='Translator Agent',
    instructions="""
    You are a translator agent. Translate the user's paragraph into the following five languages:
    - English
    - Spanish
    - French
    - Arabic
    - Chinese

    Format the output clearly like this:
    English: <translation>
    Spanish: <translation>
    French: <translation>
    Arabic: <translation>
    Chinese: <translation>
    """
)

# --- Async wrapper ---
async def translate_text(user_input):
    return await Runner.run(translator_agent, input=user_input, run_config=config)

# --- Streamlit UI ---
st.set_page_config(page_title="🌍 Translator App", layout="centered")
st.title("🌐 Multi-language Translator")
st.markdown("Translate your paragraph into **English, Spanish, French, Arabic, and Chinese**.")

user_input = st.text_area("✍️ Enter paragraph to translate:", height=150)

if st.button("🔁 Translate"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("🔄 Translating... Please wait..."):
            try:
                # Use asyncio.run if there's no running loop
                try:
                    output = asyncio.run(translate_text(user_input))
                except RuntimeError:
                    # For environments like Streamlit where event loop already exists
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    output = loop.run_until_complete(translate_text(user_input))

                st.success("✅ Translation Complete!")
                st.markdown("### 🌍 Translated Text:")
                st.text_area("📄 Output", value=output, height=300)
            except Exception as e:
                st.error(f"❌ An error occurred during translation:\n\n{str(e)}")