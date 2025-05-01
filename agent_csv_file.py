from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
    create_csv_agent
)

load_dotenv()
openai_api_key = os.getenv("MY_OPENAPI_KEY")
llm_name = "gpt-3.5-turbo"

model = ChatOpenAI(openai_api_key=openai_api_key, model=llm_name)

df = pd.read_csv("data/salaries.csv").fillna(value=0)
print(df)

agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True)

result = agent.invoke("What is the Average Salary?")

print(result)




