# S-RAG: Structured Retrieval Augmented Generation

A Python implementation of the S-RAG system from the research paper **"Structured RAG for Answering Aggregative Questions"** (arXiv:2511.08505).

## Overview

S-RAG transforms unstructured documents into structured databases that can be queried using natural language. The system uses large language models (LLMs) to extract structured data, generate SQL queries, and produce accurate answers to aggregative questions.

### Pipeline

```
Schema Prediction → Record Extraction → SQL Database → Text-to-SQL → Answer Generation
```

## Features

- **Automatic Schema Prediction**: Analyzes documents to identify recurring patterns and attributes
- **Schema Refinement**: Uses sample questions to ensure all necessary fields are captured
- **Structured Data Extraction**: Converts unstructured text into structured records
- **SQL Database Generation**: Creates queryable SQLite databases from extracted data
- **Natural Language Querying**: Translates questions into SQL and generates natural language answers
- **Evaluation Framework**: LLM-based judging system for accuracy assessment

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from srag import SRAG

# Initialize
srag = SRAG(api_key="your-openai-api-key")

# Load documents
documents = ["Your document text here...", "Another document..."]

# Predict schema
schema = srag.predict_schema(documents[:10])

# Extract records
records = srag.extract_records(documents)

# Build database
db_conn = srag.build_database()

# Calculate statistics
stats = srag.calculate_statistics()

# Query the database
result = srag.answer_query("How many items have feature X?")
print(result['answer'])
```

## Example Usage

Run the complete pipeline on the AI21 Hotels dataset:

```bash
export OPENAI_API_KEY="your-api-key"
python example.py
```

This will:
1. Load the AI21 Hotels dataset from HuggingFace
2. Predict and refine a schema
3. Extract structured records from documents
4. Build a SQL database
5. Calculate column statistics
6. Test queries and generate answers
7. Save artifacts (schema, records, statistics)
8. Optionally evaluate accuracy

## API Reference

### SRAG Class

#### Initialization

```python
srag = SRAG(api_key=None, model="gpt-4o")
```

- `api_key`: OpenAI API key (defaults to `OPENAI_API_KEY` environment variable)
- `model`: OpenAI model to use (default: "gpt-4o")

#### Methods

##### `predict_schema(documents: list[str]) -> dict`

Analyzes documents and generates a JSON schema capturing recurring patterns.

##### `refine_schema_with_questions(schema: dict, sample_questions: list[str], sample_docs: list[str]) -> dict`

Refines schema by identifying missing fields needed to answer sample questions.

##### `extract_records(documents: list[str], schema: dict = None) -> list[dict]`

Extracts structured records from documents based on the schema.

##### `build_database(schema: dict = None, records: list[dict] = None) -> sqlite3.Connection`

Creates an in-memory SQLite database from schema and records.

##### `calculate_statistics(conn: sqlite3.Connection = None, schema: dict = None) -> dict`

Calculates column-wise statistics (min, max, avg, distinct values).

##### `generate_sql(user_query: str, schema: dict = None, statistics: dict = None) -> tuple[str, str]`

Converts natural language query to SQL. Returns (SQL query, reasoning).

##### `generate_answer(user_query: str, sql_results: str) -> str`

Generates natural language answer from SQL query results.

##### `answer_query(query: str, conn: sqlite3.Connection = None, schema: dict = None, stats: dict = None, verbose: bool = True) -> dict`

Full pipeline: query → SQL → results → answer.

##### `evaluate_accuracy(questions_data: list[dict], limit: int = 20) -> float`

Evaluates system accuracy using LLM-based judging.

##### `save_artifacts(output_dir: str = ".")`

Saves schema, records, and statistics to JSON files.

## Architecture

### 1. Schema Prediction

The system analyzes sample documents to identify recurring patterns and attributes. It focuses on:
- Structured data (locations, dates, numbers)
- Boolean amenities/features
- Categorical attributes
- Avoiding nested structures

### 2. Schema Refinement

Sample questions are analyzed to identify missing fields that are needed to answer queries but weren't captured in the initial schema.

### 3. Record Extraction

Each document is processed to extract structured data according to the schema. The LLM:
- Standardizes values (e.g., "1M" → 1000000)
- Removes formatting (e.g., "$250" → 250)
- Uses null for missing fields
- Ensures type correctness

### 4. Database Construction

An in-memory SQLite database is created with:
- Schema-based table structure
- Type mapping (string→TEXT, integer→INTEGER, etc.)
- All extracted records inserted

### 5. Statistics Calculation

Column-wise statistics are computed:
- **Numeric fields**: min, max, average, count
- **Categorical fields**: distinct values, counts

### 6. Text-to-SQL

Natural language queries are converted to SQL using:
- Schema DDL with descriptions
- Column statistics
- Strict rules to prevent hallucination
- Single-table queries only

### 7. Answer Generation

SQL results are converted to natural language answers that:
- Include specific numbers from results
- Are concise and clear
- Stay grounded in the data

## Evaluation

The system includes an LLM-based evaluation framework that:
- Compares generated answers to gold standard answers
- Uses GPT-4 as a judge
- Calculates accuracy metrics
- Provides reasoning for each judgment

## Use Cases

S-RAG is ideal for:
- **Aggregative Questions**: "How many hotels have WiFi and parking?"
- **Comparative Analysis**: "What's the average rating of hotels in Sydney vs Melbourne?"
- **Filtering & Counting**: "How many products cost less than $100?"
- **Statistical Queries**: "What's the maximum capacity across all venues?"

## Limitations

- Works best with structured or semi-structured documents
- Requires consistent document formats within a dataset
- Limited to flat schemas (no nested objects)
- Depends on LLM quality for extraction accuracy

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{srag2025,
  title={Structured RAG for Answering Aggregative Questions},
  author={AI21 Labs},
  journal={arXiv preprint arXiv:2511.08505},
  year={2025}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This implementation is based on the S-RAG system described in the AI21 Labs research paper. The example uses the AI21 Hotels dataset from HuggingFace.
