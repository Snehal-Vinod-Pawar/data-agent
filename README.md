# Data Agent

An AI-powered data analysis agent that automates data extraction, database operations, and SQL-based analytics using LLMs.

<img width="1906" height="900" alt="image" src="https://github.com/user-attachments/assets/503cec2c-2efa-431c-8b6f-30c3b822f51d" />

## Features

- **ETL Agent:** Extracts data from external APIs and processes it into structured datasets.
- **SQL Analyst Agent:** Converts natural language questions into SQL queries and retrieves relevant insights.
- **Database Integration:** Manages structured data and executes SQL operations.
- **Schema Awareness:** Uses database schema information to generate context-aware SQL queries.
- **Agentic Workflow:** Organizes data processing and analysis through specialized AI agents.

## Tech Stack

- **Language:** Python
- **AI/LLM:** LangChain, LLM APIs
- **Database:** SQL
- **Data Processing:** Pandas
- **Environment Management:** uv

## Project Structure

```text
Data-Agent/
├── Models/             # Data schemas
├── agents/             # ETL and SQL analyst agents
├── architecture/       # System architecture diagrams
├── data/               # Datasets and extracted data
├── utils/              # Database and ETL utilities
├── feed_db.py          # Database population
├── main.py             # Application entry point
├── pyproject.toml      # Project dependencies
└── uv.lock             # Dependency lock file
```

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Snehal-Vinod-Pawar/data-agent.git
cd data-agent
```

**2. Install dependencies**

```bash
uv sync
```

**3. Configure environment variables**

Create a `.env` file in the root directory and add the required API keys and database credentials.

**4. Run the application**

```bash
uv run python main.py
```

## Architecture

The project uses specialized agents to handle data extraction, database operations, and natural-language-to-SQL analysis.

## Future Improvements

- Add support for multiple data sources.
- Improve SQL query validation and error handling.
- Introduce automated data quality checks.
- Add interactive data visualization.
