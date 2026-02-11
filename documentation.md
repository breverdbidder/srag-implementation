# S-RAG System Documentation

**Author:** Manus AI

**Date:** 2026-02-11

## 1. Introduction

This document provides a comprehensive overview and technical documentation of the **Structured Retrieval Augmented Generation (S-RAG)** system, an implementation based on the research paper "Structured RAG for Answering Aggregative Questions" [1]. S-RAG is designed to bridge the gap between unstructured text and structured query answering by creating a robust pipeline that transforms documents into a queryable database.

### 1.1. System Purpose

The primary goal of S-RAG is to enable users to ask complex, aggregative questions about a collection of documents in natural language and receive precise, data-driven answers. It achieves this by extracting structured information from unstructured text, storing it in a relational database, and using a text-to-SQL model to query the data.

### 1.2. Architectural Overview

The S-RAG system follows a sequential pipeline, where the output of each stage serves as the input for the next. This modular architecture allows for independent testing and refinement of each component.

**Figure 1: S-RAG System Architecture**

```
+----------------------+
|   Unstructured Text  |
+----------------------+
           |
           v
+----------------------+
|  1. Schema Prediction  |
+----------------------+
           |
           v
+----------------------+
|  2. Schema Refinement  |
+----------------------+
           |
           v
+----------------------+
|  3. Record Extraction  |
+----------------------+
           |
           v
+----------------------+
|  4. Database Building  |
+----------------------+
           |
           v
+----------------------+
| 5. Statistics Calculation |
+----------------------+
           |
           v
+----------------------+
|   6. Text-to-SQL     |
+----------------------+
           |
           v
+----------------------+
|   7. Answer Generation |
+----------------------+
           |
           v
+----------------------+
|   Natural Language   |
|        Answer        |
+----------------------+
```

## 2. Core Components

This section details each major component of the S-RAG pipeline, referencing the corresponding functions in the `srag.py` implementation.

### 2.1. Schema Prediction

**Function:** `predict_schema(documents: list[str]) -> dict`

The first step in the pipeline is to derive a structured schema from a sample of unstructured documents. The system uses a large language model (LLM) to analyze the text and identify recurring concepts, attributes, and patterns.

> The prompt instructs the model to focus on creating a flat schema (no nested objects), use boolean fields for amenities, and capture all relevant structured data points like locations, dates, and numbers. This ensures the resulting schema is well-suited for a relational database.

| Parameter   | Type          | Description                                      |
|-------------|---------------|--------------------------------------------------|
| `documents` | `list[str]`   | A list of document texts to be analyzed.         |

**Returns:** A JSON dictionary representing the predicted schema, including properties, types, descriptions, and examples for each field.

### 2.2. Schema Refinement

**Function:** `refine_schema_with_questions(schema: dict, sample_questions: list[str], sample_docs: list[str]) -> dict`

While the initial schema prediction is comprehensive, it may miss fields that are implicitly required to answer certain types of questions. The schema refinement step addresses this by analyzing a set of sample questions.

> The system prompts an LLM to identify concepts in the questions that are not present in the current schema. If missing fields are found, they are automatically added to the schema, ensuring it is robust enough to handle the expected query load.

| Parameter            | Type          | Description                                      |
|----------------------|---------------|--------------------------------------------------|
| `schema`             | `dict`        | The initial schema predicted in the previous step. |
| `sample_questions`   | `list[str]`   | A list of sample questions to analyze.           |
| `sample_docs`        | `list[str]`   | Sample documents for providing context.          |

**Returns:** A refined JSON schema dictionary with the newly added fields.

### 2.3. Record Extraction

**Functions:** `predict_record(document_text: str, schema: dict) -> dict`, `extract_records(documents: list[str], schema: dict = None) -> list[dict]`

With a finalized schema, the system iterates through all documents and extracts structured records. The `predict_record` function processes a single document, while `extract_records` orchestrates the extraction for an entire dataset.

> The prompt for this stage is highly constrained, instructing the model to adhere strictly to the provided schema, standardize values (e.g., removing currency symbols), use correct data types, and output only valid JSON. This minimizes errors and ensures data consistency.

| Parameter         | Type          | Description                                      |
|-------------------|---------------|--------------------------------------------------|
| `document_text`   | `str`         | The text of a single document.                   |
| `schema`          | `dict`        | The refined schema to guide extraction.          |
| `documents`       | `list[str]`   | A list of all document texts to be processed.    |

**Returns:** A list of dictionaries, where each dictionary represents a structured record extracted from a document.

### 2.4. Database Construction

**Function:** `build_database(schema: dict = None, records: list[dict] = None) -> sqlite3.Connection`

This component transforms the extracted records into a queryable, in-memory SQLite database. It dynamically creates a table based on the schema and inserts all the records.

| Parameter   | Type                | Description                                      |
|-------------|---------------------|--------------------------------------------------|
| `schema`    | `dict`              | The schema defining the table structure.         |
| `records`   | `list[dict]`        | The list of records to be inserted.              |

**Returns:** A `sqlite3.Connection` object for the in-memory database.

### 2.5. Statistics Calculation

**Function:** `calculate_statistics(conn: sqlite3.Connection, schema: dict) -> dict`

To help the text-to-SQL model generate more accurate and efficient queries, the system calculates summary statistics for each column in the database.

- **For numeric fields:** It computes the minimum, maximum, average, and count.
- **For categorical fields:** It identifies the most common distinct values.

These statistics are provided as context to the LLM during the SQL generation phase.

| Parameter   | Type                | Description                                      |
|-------------|---------------------|--------------------------------------------------|
| `conn`      | `sqlite3.Connection`| The connection to the database.                  |
| `schema`    | `dict`              | The schema used to identify field types.         |

**Returns:** A dictionary containing the calculated statistics for each field.

### 2.6. Text-to-SQL Generation

**Function:** `generate_sql(user_query: str, schema: dict, statistics: dict) -> tuple[str, str]`

This is the core of the query-answering process. The function takes a user's natural language question and converts it into a valid SQLite query.

> The prompt is heavily engineered with strict rules to prevent common text-to-SQL pitfalls. It includes the database schema (DDL), column statistics, and explicit instructions to only query the single `hotels` table, avoid `JOIN` operations, and use correct boolean logic. This grounding ensures high-quality, executable SQL.

| Parameter      | Type          | Description                                      |
|----------------|---------------|--------------------------------------------------|
| `user_query`   | `str`         | The user's question in natural language.         |
| `schema`       | `dict`        | The database schema.                             |
| `statistics`   | `dict`        | The calculated column statistics.                |

**Returns:** A tuple containing the generated SQL query and the LLM's reasoning.

### 2.7. Answer Generation

**Function:** `generate_answer(user_query: str, sql_results: str) -> str`

After executing the SQL query, the raw results are passed to this function to be synthesized into a clear, concise natural language answer.

> The prompt instructs the model to base its answer *only* on the provided SQL results, ensuring the response is grounded in the data and avoids hallucination.

| Parameter       | Type          | Description                                      |
|-----------------|---------------|--------------------------------------------------|
| `user_query`    | `str`         | The original user question for context.          |
| `sql_results`   | `str`         | The results from the SQL query execution.        |

**Returns:** A natural language answer to the user's question.

### 2.8. Evaluation Framework

**Functions:** `judge_answer(query: str, gold_answer: str, judged_answer: str) -> bool`, `evaluate_accuracy(questions_data: list[dict], limit: int = 20) -> float`

The S-RAG system includes a robust evaluation framework based on the "LLM-as-a-Judge" pattern. The `judge_answer` function compares the system's generated answer to a ground-truth ("gold") answer and determines if it is correct. The `evaluate_accuracy` function orchestrates this process over a set of evaluation questions to compute an overall accuracy score.

## 3. References

[1] AI21 Labs. (2025). *Structured RAG for Answering Aggregative Questions*. arXiv preprint arXiv:2511.08505.
