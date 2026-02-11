# S-RAG Adaptation Guide

**Author:** Manus AI

**Date:** 2026-02-11

## 1. Introduction

This guide provides a step-by-step process for adapting the **Structured Retrieval Augmented Generation (S-RAG)** system to new datasets and domains. The modular architecture of S-RAG is designed for flexibility, and with careful attention to the initial schema generation phase, it can be effectively applied to a wide variety of unstructured text sources.

## 2. Core Principles of Adaptation

The success of S-RAG in a new domain hinges on the quality of the schema it generates. The entire downstream pipeline—from record extraction to SQL generation—is dependent on a well-defined and comprehensive schema. Therefore, the adaptation process primarily focuses on ensuring the system can create an accurate schema for your specific data.

Key considerations include:

- **Document Consistency:** The system performs best when documents within a dataset share a similar structure and format.
- **Data Richness:** The source documents should contain identifiable structured data (e.g., names, dates, numbers, categories, features) for the system to extract.
- **Query Types:** The system is optimized for **aggregative questions**. Consider the types of questions you want to answer when preparing your data and evaluation set.

## 3. Step-by-Step Adaptation Process

Adapting the S-RAG system involves preparing your data and running the initial stages of the pipeline to generate a domain-specific schema.

### Step 1: Prepare Your Dataset

First, gather your documents into a format that can be easily loaded. The `srag.py` script uses a `load_jsonl` function that expects a JSONL (JSON Lines) file where each line is a JSON object containing a `document_content` key.

```json
{"document_content": "This is the text of the first document..."}
{"document_content": "This is the text of the second document..."}
```

**Recommendations:**

1.  **Create a Representative Sample:** Select a small but diverse subset of your documents (10-20 is often sufficient) to be used for schema prediction. This sample should cover the full range of concepts and attributes present in your dataset.
2.  **Prepare Sample Questions:** Write a list of 10-15 natural language questions that you would want to ask about your data. These will be used to refine the schema and ensure it captures all necessary information.

### Step 2: Generate and Refine the Schema

This is the most critical step. You will use your prepared data samples to guide the S-RAG system in creating a new schema.

1.  **Modify `example.py` (or create a new script):** Update the script to load your new dataset and sample questions.

    ```python
    # In your adaptation script
    from srag import SRAG

    # Load your custom documents and questions
    my_docs = load_your_custom_data("path/to/your/data.jsonl")
    my_questions = load_your_sample_questions("path/to/your/questions.txt")

    # Select a sample for schema generation
    schema_sample_docs = my_docs[:15]
    ```

2.  **Run Schema Prediction and Refinement:** Execute the `predict_schema` and `refine_schema_with_questions` functions. The prompts within these functions are designed to be domain-agnostic and should work well for most datasets.

    ```python
    srag = SRAG()

    # Predict the initial schema
    initial_schema = srag.predict_schema(schema_sample_docs)

    # Refine it with your sample questions
    refined_schema = srag.refine_schema_with_questions(
        initial_schema,
        my_questions,
        schema_sample_docs
    )

    # Save the new schema
    with open("my_domain_schema.json", "w") as f:
        json.dump(refined_schema, f, indent=2)
    ```

At the end of this step, you will have a `my_domain_schema.json` file tailored to your specific dataset.

### Step 3: Run the Full Pipeline

With the new schema, you can now run the rest of the S-RAG pipeline on your full dataset.

1.  **Extract Records:** Use the `extract_records` function with your full document list and the newly generated schema.
2.  **Build Database:** The `build_database` function will automatically create a table structure based on your new schema.
3.  **Calculate Statistics & Answer Queries:** The remaining steps (`calculate_statistics`, `answer_query`) will function automatically using the new database and schema.

### Step 4: Evaluate Performance

To validate the system's performance on your new domain, you need a set of evaluation questions with known, ground-truth answers.

1.  **Create an Evaluation Set:** Prepare a list of dictionaries, where each dictionary contains a `question` and its corresponding `answer`.
2.  **Run Evaluation:** Use the `evaluate_accuracy` function to measure how well the adapted system performs.

    ```python
    evaluation_set = [
        {"question": "How many products are in category X?", "answer": "There are 42 products in category X."},
        # ... more questions
    ]

    accuracy = srag.evaluate_accuracy(evaluation_set)
    print(f"Accuracy on my domain: {accuracy:.1f}%")
    ```

## 4. Example Scenario: Adapting to Scientific Papers

Let's consider adapting S-RAG to answer questions about a collection of scientific research papers.

1.  **Data Preparation:**
    -   **Documents:** The abstracts of thousands of papers on machine learning.
    -   **Sample Questions:**
        -   "How many papers were published in 2023?"
        -   "What is the average number of authors for papers mentioning 'transformer' models?"
        -   "Which papers from the University of Toronto have more than 100 citations?"

2.  **Schema Generation:**
    -   Running `predict_schema` on a sample of abstracts would likely produce fields like: `title`, `publication_year`, `journal_name`, `author_count`, `citation_count`.
    -   `refine_schema_with_questions` might add fields like `author_affiliation` or boolean fields like `mentions_transformer_model`.

3.  **Pipeline Execution:**
    -   The system would extract records for each paper, build a `papers` table in the database, and calculate statistics (e.g., min/max `publication_year`, average `citation_count`).

4.  **Querying:**
    -   A user could then ask: "What is the most cited paper from 2022?"
    -   The system would generate the SQL: `SELECT title, MAX(citation_count) FROM papers WHERE publication_year = 2022;`
    -   Finally, it would generate a natural language answer based on the query result.

## 5. Troubleshooting and Best Practices

-   **Poor Schema Prediction:** If the initial schema is missing key fields, your document sample may not be representative enough. Try adding more diverse documents to the sample.
-   **Incorrect Data Extraction:** If the LLM struggles to extract data correctly, the documents may be too unstructured or varied. Consider pre-processing the text to add more consistent formatting.
-   **Bad SQL Generation:** Incorrect SQL is almost always a symptom of a flawed schema. If the schema does not accurately reflect the content of the documents, the text-to-SQL model will struggle. Revisit the schema prediction and refinement steps.
-   **Iterate:** The adaptation process may require a few iterations. Start with a small sample, evaluate the results, and refine your data or schema as needed before scaling to the full dataset.
