import streamlit as st

# Chatbot dependencies
import pandas as pd  # Data frame minipulation
import psycopg2 as pg  # Postgres db connection
import re
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient  # LLM

# Program dependencies
from contextlib import redirect_stdout  # For capture the exec output
import io  # Store into a string stream
import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Loading ENV File
if load_dotenv():
    print("ENV loaded")


# Initialize DB connection
print("Connecting to Database...")

DB = os.getenv("DB_NAME")
USER = os.getenv("DB_USER")
PASS = os.getenv("DB_PASSWORD")
DHOST = os.getenv("DB_HOST")

connection = pg.connect(database=DB, user=USER, password=PASS, host=DHOST)
if connection:
    print("Database connection successful")

# Initializing Pandas DF
print("Initializing pandas DataFrame...")

sql_str = "SELECT * FROM public.aggregations;"
df = pd.read_sql(sql=sql_str, con=connection)

print("Dataframe Initialized")

# HF Initialization

print("Initializing HF Client...")
client = InferenceClient(token=os.getenv("HF_SECRET"))
print("Initialized HF Client")


@dataclass
class ExecutionResult:
    output: str
    plots: List[str]


def get_llm_output(query: str) -> str:
    sys_prompt = f"""
        You are a data analyst, and you're job is to write python code to evaluate the users query
        you have access to the following tools, please note you can only write python code:
            - pandas imported as pd
            - matplotlib imported as plt
            - a data frame df with columns ({df.columns})
            - if any plots are made they must be saved as png files in plots directory
            - output dataframe in markdown
        YOU MAY NOT INSTALL ANY ADDITIONAL PACKAGES AND IMPORT ANY PACKAGES
    """

    message_history = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query},
    ]

    out = client.chat_completion(
        model="microsoft/Phi-3.5-mini-instruct",
        messages=message_history,
        max_tokens=500,
        stream=False,
    )

    return out.choices[0].message.content


def parse_generated_code(response: str) -> str:
    pattern = r"```python\n([\s\S]*?)\n```"
    matches = re.search(pattern, response)
    return matches.groups()[0]


def execute_code(code_block: str) -> ExecutionResult:
    # Clean plots of prev executions
    for f in os.listdir("plots"):
        os.remove(os.path.join("plots", f))

    ss = io.StringIO()
    with redirect_stdout(ss):
        exec(code_block)

    plot_path = []

    for f in os.listdir("plots"):
        plot_path.append(os.path.join("./plots", f))

    return ExecutionResult(output=ss.getvalue(), plots=plot_path)


st.title("Database Chatbot demo (Streamlit)")

with st.expander("Look at the data"):
    st.dataframe(df.head())

message_container = st.container()
prompt = st.chat_input("Ask me something about the data")

if prompt:
    message_container.chat_message("user").write(prompt)

    output = get_llm_output(prompt)
    code_block = parse_generated_code(output)

    message_container.markdown("_LLM Generated Code_")
    message_container.chat_message("assistant").code(code_block, line_numbers=True)
    message_container.markdown("_Evaluated Result_")

    result = execute_code(code_block)

    if len(result.output) != 0:
        message_container.chat_message("assistant").markdown(result.output)
    if len(result.plots) != 0:
        message_container.chat_message("assistant").write(
            "These are the plots I have created"
        )
        message_container.chat_message("assistant").image(result.plots)
