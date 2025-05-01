import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
load_dotenv()

openai_key = os.getenv("MY_OPENAPI_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model_name=llm_name)

messages = [
    SystemMessage(content="You are a helpful assistant who is extremely competent as a Computer Scientist. "
                          "Your name is Rob."),
    HumanMessage(content="Who was the very first computer scientist"),
]

result = model.invoke(messages)
print(result)