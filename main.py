from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print(gemini_api_key)

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

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

# Write Agent
writer = Agent(
    name = 'Translator Agent',
    instructions= 
    """You are a translator agent. translate users paragraph into 5 five languages: English, Spanish, French, Arabic, and Chinese."""
)

response = Runner.run_sync(
    writer,
    input = 'ap kaise ho bhai, me thek hun',
    run_config = config
    )
print(response.final_output)



def main():
    pass

if __name__ == "__main__":
    main()
