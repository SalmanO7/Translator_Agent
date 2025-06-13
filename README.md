import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
    st.stop()

# Initialize the external client and model
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Create the Agent
translator_agent = Agent(
    name='Translator Agent',
    instructions="""
    You are a translator agent. Translate the user's paragraph into five languages:
    English, Spanish, French, Arabic, and Chinese.
    """
)

# Streamlit UI
st.set_page_config(page_title="Multi-language Translator", layout="centered")
st.title("🌍 Multi-language Translator")
st.markdown("Enter a paragraph below to translate it into **English, Spanish, French, Arabic, and Chinese**.")

user_input = st.text_area("Enter your text", height=150)

if st.button("Translate"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            response = Runner.run_sync(
                translator_agent,
                input=user_input,
                run_config=config
            )
            st.success("✅ Translation complete!")

            st.markdown("### 🌐 Translated Output:")
            st.text_area("Output", value=response.final_output, height=250)
"# Translator_Agent" 
