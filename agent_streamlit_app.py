import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent
)
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("MY_OPENAPI_KEY")
llm_name = "gpt-3.5-turbo"

model = ChatOpenAI(openai_api_key=openai_api_key, model=llm_name)

df = pd.read_csv("data/salaries.csv").fillna(value=0)
print(df)

agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True)

st.title("Database AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head())

st.write("Ask A Question")
question = st.text_input("Enter your question about the dataset", "Which grade has the highest average base salary, and compare")

if st.button("Run Query"):
    QUERY = question
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.markdown(res["output"])
