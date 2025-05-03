import pandas as pd
from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
import os

load_dotenv()
# Create a DB from CSV File

database_file_path = "sqlite:///./db/salary.db"
engine = create_engine(database_file_path)
df = pd.read_csv('data/salaries.csv').fillna(value=0)
df.to_sql("salaries_2023", con=engine, if_exists='replace', index=False)


openai_api_key = os.getenv("MY_OPENAPI_KEY")
llm_name = "gpt-3.5-turbo"

model = ChatOpenAI(openai_api_key=openai_api_key, model=llm_name)


db = SQLDatabase.from_uri(database_file_path)
toolkit = SQLDatabaseToolkit(db=db, llm = model)

sql_agent = create_sql_agent(
    toolkit=toolkit,
    llm=model,
    verbose = True
)

QUESTION = "how many rows are there"
response = sql_agent.invoke(QUESTION)
print(response)
