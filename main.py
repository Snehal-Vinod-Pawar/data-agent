
import os
import ast
from pathlib import Path

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

from agents.sql_analyst import sql_analyst


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Data Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f7f8fc;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8eaf0;
    }

    .brand {
        font-size: 25px;
        font-weight: 800;
        color: #242b45;
        letter-spacing: -1px;
        padding: 10px 0 4px 0;
    }

    .brand span {
        color: #6558e8;
    }

    .muted {
        color: #858ba0;
        font-size: 13px;
    }

    .hero {
        padding: 28px 0 20px 0;
    }

    .hero h1 {
        color: #20243a;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #7b8194;
        font-size: 15px;
    }

    .answer-card {
        background: #ffffff;
        border: 1px solid #e8eaf1;
        border-radius: 14px;
        padding: 20px;
        margin: 12px 0;
    }

    .answer-label {
        font-size: 12px;
        font-weight: 700;
        color: #6558e8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .status-card {
        background: #ffffff;
        border: 1px solid #e8eaf1;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8eaf1;
        padding: 15px;
        border-radius: 12px;
    }

    .section-title {
        color: #252a40;
        font-size: 20px;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    .stChatMessage {
        border: 1px solid #eceef4;
        border-radius: 12px;
        background: white;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "SQL Analyst"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_database_connection():

    config = {
        "host": os.getenv("host"),
        "port": os.getenv("port"),
        "database": os.getenv("database"),
        "user": os.getenv("user"),
        "password": os.getenv("password"),
    }

    if not all(config.values()):
        raise ValueError("Database configuration is incomplete in .env")

    return psycopg2.connect(
        host=config["host"],
        port=int(config["port"]),
        dbname=config["database"],
        user=config["user"],
        password=config["password"],
        connect_timeout=5,
    )


def check_database_status():

    connection = None

    try:
        connection = get_database_connection()
        return True, "Connected"

    except Exception as error:
        return False, str(error)

    finally:
        if connection:
            connection.close()


# --------------------------------------------------
# SQL RESULT PROCESSING
# --------------------------------------------------

def parse_query_result(result):

    if result is None:
        return None

    if isinstance(result, str):
        try:
            result = ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return None

    if isinstance(result, list):
        if not result:
            return pd.DataFrame()

        if all(isinstance(row, (tuple, list)) for row in result):
            return pd.DataFrame(result)

        if all(isinstance(row, dict) for row in result):
            return pd.DataFrame(result)

    return None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">Data<span>Agent.</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="muted">Your intelligent data workspace</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("##### WORKSPACE")

    pages = {
        "SQL Analyst": "💬",
        "ETL Analyst": "🔄",
        "Data Sources": "🗄️",
    }

    for page_name, icon in pages.items():

        if st.button(
            f"{icon}   {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_name else "secondary",
        ):
            st.session_state.page = page_name
            st.rerun()

    st.divider()

    st.markdown("##### DATABASE STATUS")

    db_connected, db_message = check_database_status()

    if db_connected:
        st.success("PostgreSQL Connected")
    else:
        st.error("PostgreSQL Disconnected")
        with st.expander("Connection details"):
            st.caption(db_message)

    st.divider()

    st.caption("DATA AGENT")
    st.caption("Powered by LangGraph & Gemini")
    st.caption("Version 1.0")


# --------------------------------------------------
# SQL ANALYST PAGE
# --------------------------------------------------

def render_sql_analyst():

    st.markdown("""
    <div class="hero">
        <h1>Your Data, One Conversation Away.</h1>
        <p>Ask questions in natural language. Get SQL queries,
        database insights and meaningful answers.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Agent", "SQL Analyst")

    with col2:
        st.metric(
            "Database",
            "Connected" if db_connected else "Offline"
        )

    with col3:
        st.metric("Conversation", len(st.session_state.chat_history))

    st.markdown("### Ask your data")

    st.caption(
        "Ask anything about your users, payments, rides, ratings or vehicles."
    )

    # Suggested questions
    suggestions = [
        "What are the different payment methods?",
        "Show the top 10 users by signup date.",
        "What is the average ride fare?",
    ]

    st.markdown("##### Try asking")

    suggestion_cols = st.columns(3)

    for index, question in enumerate(suggestions):
        with suggestion_cols[index]:
            if st.button(
                question,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question
                st.rerun()

    st.divider()

    # Display conversation history
    for item in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant", avatar="🤖"):

            st.markdown("**Answer**")
            st.write(item["answer"])

            if item.get("error"):
                st.error(item["error"])

            if item.get("sql"):
                with st.expander("Generated SQL", expanded=False):
                    st.code(item["sql"], language="sql")

            if item.get("safety"):

                if item["safety"].lower() == "yes":
                    st.success("SQL Safety Check: Approved")
                else:
                    st.warning("SQL Safety Check: Rejected")

            result_df = parse_query_result(item.get("result"))

            if result_df is not None:
                with st.expander("Database Results", expanded=True):

                    if result_df.empty:
                        st.info("The query returned no rows.")
                    else:
                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.caption(f"{len(result_df)} rows returned")

    # Chat input
    question = st.chat_input("Ask a question about your database...")

    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:

        st.session_state.chat_history.append({
            "question": question,
            "answer": "Processing your request...",
            "sql": "",
            "result": None,
            "safety": "",
        })

        history_index = len(st.session_state.chat_history) - 1

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):

            try:
                with st.spinner("Analyzing your question and querying the database..."):

                    input_schema = {
                        "messages": [],
                        "user_question": question,
                        "curated_ques": "",
                        "prompt_query_context": "",
                        "generated_sql_query": "",
                        "is_safe": "No",
                        "comments": "",
                        "sql_query_execution_result": "",
                        "final_answer": "",
                    }

                    response = sql_analyst.invoke(input_schema)

                answer = response.get(
                    "final_answer",
                    "No final answer was generated."
                )

                sql = response.get("generated_sql_query", "")
                result = response.get("sql_query_execution_result")
                safety = response.get("is_safe", "No")

                error = None

                if isinstance(result, str) and result.startswith(
                    "Database query failed:"
                ):
                    error = result

                st.session_state.chat_history[history_index] = {
                    "question": question,
                    "answer": answer,
                    "sql": sql,
                    "result": result,
                    "safety": safety,
                    "error": error,
                }

                st.markdown("**Answer**")
                st.write(answer)

                if safety.lower() == "yes":
                    st.success("SQL Safety Check: Approved")
                else:
                    st.warning("SQL Safety Check: Rejected")

                if sql:
                    with st.expander("Generated SQL"):
                        st.code(sql, language="sql")

                result_df = parse_query_result(result)

                if result_df is not None:
                    with st.expander("Database Results", expanded=True):
                        if result_df.empty:
                            st.info("No rows returned.")
                        else:
                            st.dataframe(
                                result_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                if error:
                    st.error(error)

            except Exception as error:

                error_message = str(error)

                st.session_state.chat_history[history_index] = {
                    "question": question,
                    "answer": "I couldn't complete your request.",
                    "sql": "",
                    "result": None,
                    "safety": "",
                    "error": error_message,
                }

                st.error(f"Agent execution failed: {error_message}")

        st.rerun()

    if st.session_state.chat_history:

        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# --------------------------------------------------
# ETL ANALYST PAGE
# --------------------------------------------------

def render_etl_analyst():

    st.markdown("""
    <div class="hero">
        <h1>ETL Analyst</h1>
        <p>Extract, transform and load your data into structured databases.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "The ETL Analyst workflow is not yet implemented. "
        "This section will be enabled once the ETL backend is ready."
    )

    st.markdown("### Planned capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📥 Extract")
        st.caption("Import CSV, Excel, API and database data.")

    with col2:
        st.markdown("#### ⚙️ Transform")
        st.caption("Clean, validate and standardize data.")

    with col3:
        st.markdown("#### 📤 Load")
        st.caption("Load processed data into PostgreSQL.")


# --------------------------------------------------
# DATA SOURCES PAGE
# --------------------------------------------------

def render_data_sources():

    st.markdown("""
    <div class="hero">
        <h1>Data Sources</h1>
        <p>Explore your available datasets and database connection.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### PostgreSQL Connection")

    if db_connected:
        st.success("Your PostgreSQL database is connected.")
    else:
        st.error("PostgreSQL connection failed.")
        st.caption(db_message)

    st.divider()

    st.markdown("### Available CSV Datasets")

    if not DATA_DIR.exists():
        st.warning("The data directory was not found.")
        return

    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        st.info("No CSV files found in the data directory.")
        return

    total_rows = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            total_rows += len(df)

            with st.container(border=True):

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**📄 {csv_file.name}**")
                    st.caption("CSV Dataset")

                with col2:
                    st.metric("Rows", len(df))

                with col3:
                    st.metric("Columns", len(df.columns))

                with st.expander("Preview dataset"):
                    st.dataframe(
                        df.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as error:
            st.error(f"Could not read {csv_file.name}: {error}")

    st.divider()

    st.metric("Total rows across CSV files", total_rows)


# --------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------

if st.session_state.page == "SQL Analyst":
    render_sql_analyst()

elif st.session_state.page == "ETL Analyst":
    render_etl_analyst()

elif st.session_state.page == "Data Sources":
    render_data_sources()
