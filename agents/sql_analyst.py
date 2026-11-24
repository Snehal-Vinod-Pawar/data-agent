import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.database import DatabaseUtil
from Models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END



def extract_response_text(response) -> str:
    """Extract plain text from a LangChain AIMessage."""

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "\n".join(text_parts).strip()

    raise TypeError(
        f"Unexpected response content type: {type(content)}"
    )


def clean_sql_query(query: str) -> str:
    query = query.strip()

    if query.startswith("```"):
        lines = query.splitlines()

        # Remove opening code fence (```sql or ```)
        lines = lines[1:]

        # Remove closing code fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        query = "\n".join(lines)

    return query.strip()


#-------------------------------- AI Agent Code ---------------------------------
def curate_ques(state: AgentSchema) -> AgentSchema:

    user_question = state.user_question
    llm = pick_llm("low")

    response = llm.invoke(
        f"Curate the following user question: {user_question}"
    )

    curated_question = extract_response_text(response)

    state.curated_ques = curated_question

    state.messages = state.messages + [
        AIMessage(content=curated_question)
    ]

    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:

    print("Running: prompt_query_context", flush=True)

    curated_question = state.curated_ques

    conn_details = {
        "host": os.getenv("host"),
        "port": int(os.getenv("port")),
        "database": os.getenv("database"),
        "user": os.getenv("user"),
        "password": os.getenv("password")
    }

    obj = DatabaseUtil(conn_details)

    schema_info = obj.schema_details("public")

    prompt = f"""
    You are an SQL analyst agent.

    Convert the user's natural language question into a PostgreSQL SQL query.

    Use the provided database schema and sample data to generate an accurate query.

    Instructions:
    1. Generate only read-only SELECT queries.
    2. Unless the user explicitly requests a specific number of rows, limit results to 10 rows.
    3. Use only tables and columns present in the schema.
    4. Return raw PostgreSQL SQL only.
    5. Do not include Markdown code fences or explanations.

    User Question:
    {curated_question}

    Database Schema:
    {schema_info}
    """

    state.prompt_query_context = prompt

    return state

def generate_sql(state: AgentSchema) -> AgentSchema:

    prompt = state.prompt_query_context
    llm = pick_llm("intermediate")

    response = llm.invoke(prompt)

    generated_sql_query = extract_response_text(response)
    generated_sql_query = clean_sql_query(generated_sql_query)

    print("Generated SQL:", generated_sql_query)

    state.generated_sql_query = generated_sql_query

    return state

def is_safe_sql(state: AgentSchema) -> AgentSchema:

    print("Running: is_safe_sql", flush=True)

    sql_query = state.generated_sql_query

    llm = pick_llm("intermediate")

    llm_judge = llm.with_structured_output(JudgeSchema)

    prompt = f"""
    You are an SQL security judge.

    Determine whether this query is safe for read-only database execution.

    Only SELECT queries without data modification or multiple statements
    should be considered safe.

    Reject INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE,
    GRANT, REVOKE, and other modifying or administrative commands.

    SQL Query:
    {sql_query}
    """

    response = llm_judge.invoke(prompt)

    state.is_safe = response.answer
    state.comments = response.comments

    print("SQL Safety:", state.is_safe, flush=True)
    print("Judge Comments:", state.comments, flush=True)

    return state


def canceled_sql(state: AgentSchema) -> AgentSchema:

    print("Running: canceled_sql", flush=True)

    comments = state.comments

    state.final_answer = (
        f"The SQL query was canceled due to safety concerns. "
        f"Reason: {comments}"
    )

    return state


def execute_sql(state: AgentSchema) -> AgentSchema:

    print("Running: execute_sql", flush=True)

    sql_query = state.generated_sql_query

    conn_details = {
        "host": os.getenv("host"),
        "port": int(os.getenv("port")),
        "database": os.getenv("database"),
        "user": os.getenv("user"),
        "password": os.getenv("password")
    }

    obj = DatabaseUtil(conn_details)

    try:
        execution_result = obj.execute_query(sql_query)

        if execution_result is None:
            execution_result = "Query returned no result."

        state.sql_query_execution_result = str(execution_result)

    except Exception as e:
        state.sql_query_execution_result = (
            f"Database query failed: {str(e)}"
        )

    return state

def represent_final_answer(state: AgentSchema) -> AgentSchema:

    print("Running: represent_final_answer", flush=True)

    execution_result = state.sql_query_execution_result
    curated_question = state.curated_ques

    if execution_result.startswith("Database query failed:"):
        state.final_answer = execution_result
        return state

    llm = pick_llm("low")

    prompt = f"""
    You are an SQL analyst agent.

    Provide a concise, clear, user-friendly answer based on the database result.

    Do not include SQL code or unnecessary technical details.

    Database Result:
    {execution_result}

    User Question:
    {curated_question}
    """

    response = llm.invoke(prompt)

    llm_response = extract_response_text(response)

    state.final_answer = llm_response

    state.messages = state.messages + [
        AIMessage(content=llm_response)
    ]

    return state


#------------------------------- State Graph ---------------------------------
sql_agent_graph = StateGraph(AgentSchema)

#Nodes
# Nodes
sql_agent_graph.add_node("curate_ques", curate_ques)
sql_agent_graph.add_node("prompt_query_context", prompt_query_context)
sql_agent_graph.add_node("generate_sql", generate_sql)
sql_agent_graph.add_node("is_safe_sql", is_safe_sql)
sql_agent_graph.add_node("canceled_sql", canceled_sql)
sql_agent_graph.add_node("execute_sql", execute_sql)
sql_agent_graph.add_node("represent_final_answer", represent_final_answer)

# Edges
sql_agent_graph.add_edge(START, "curate_ques")
sql_agent_graph.add_edge("curate_ques", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

# Conditional Edge Function
def is_safe_sql_edge(state: AgentSchema) -> str:
    is_safe = state.is_safe

    if is_safe.lower() == "yes":
        return "execute_sql"

    else :
        return "canceled_sql"

sql_agent_graph.add_conditional_edges("is_safe_sql", is_safe_sql_edge,
                                      {
                                          "execute_sql": "execute_sql",
                                          "canceled_sql": "canceled_sql"
                                      })

# sql_agent_graph.add_edge("is_safe_sql", "execute_sql")
# sql_agent_graph.add_edge("is_safe_sql", "canceled_sql")

sql_agent_graph.add_edge("canceled_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)


# Compile Graph
sql_analyst = sql_agent_graph.compile()

from IPython.display import display, Image

if __name__ == "__main__":

    # Graph Visualixation
    # from IPython.display import display, Image
    # img = Image(sql_analyst.get_graph().draw_mermaid_png())
    # with open("sql_analyst_graph.png", "wb") as f:
    #     f.write(img.data)

    input_schema = {
        "messages": [],
        "user_question": "What are the different types of Payment Methods we have in our database",
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

     # Execute the Graph
    sql_analyst_response = sql_analyst.invoke(input_schema)
    print(sql_analyst_response['messages'])  # Print the final output of the graph execution
    print("********************************")

    print(sql_analyst_response['generated_sql_query'])  # Print the generated SQL query

    print("********************************")

    print(sql_analyst_response['sql_query_execution_result'])  # Print the result of executing the SQL query

    print("********************************")

    print(sql_analyst_response['prompt_query_context'])  # Print the prompt query context