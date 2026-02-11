"""
Example usage of the S-RAG system with the AI21 Hotels dataset.
"""

from srag import SRAG
import json


def main():
    """Run S-RAG pipeline on AI21 Hotels dataset."""
    
    # Initialize S-RAG
    print("Initializing S-RAG system...")
    srag = SRAG(model="gpt-4.1-mini")
    
    # Dataset URLs
    TRAIN_CORPUS_URL = "https://huggingface.co/datasets/ai21labs/aggregative_questions/resolve/main/datasets/AI21-Hotels_train/corpus.jsonl"
    TRAIN_QUESTIONS_URL = "https://huggingface.co/datasets/ai21labs/aggregative_questions/resolve/main/datasets/AI21-Hotels_train/questions.jsonl"
    EVAL_CORPUS_URL = "https://huggingface.co/datasets/ai21labs/aggregative_questions/resolve/main/datasets/AI21-Hotels_eval/corpus.jsonl"
    EVAL_QUESTIONS_URL = "https://huggingface.co/datasets/ai21labs/aggregative_questions/resolve/main/datasets/AI21-Hotels_eval/questions.jsonl"
    
    # Load datasets
    print("\n" + "="*80)
    print("STEP 1: Loading datasets")
    print("="*80)
    
    train_corpus = srag.load_jsonl(TRAIN_CORPUS_URL)
    train_questions = srag.load_jsonl(TRAIN_QUESTIONS_URL)
    eval_corpus = srag.load_jsonl(EVAL_CORPUS_URL)
    eval_questions = srag.load_jsonl(EVAL_QUESTIONS_URL)
    
    # Extract document content
    train_docs = [item['document_content'] for item in train_corpus]
    eval_docs = [item['document_content'] for item in eval_corpus]
    
    # Use train docs for schema prediction, eval docs for building database
    schema_sample_docs = train_docs[:12]
    all_documents = eval_docs  # Use eval split for database
    
    # Extract questions
    test_questions = [item['question'] for item in eval_questions]
    sample_test_questions = test_questions[:10]
    
    print(f"\nTrain corpus: {len(train_docs)} docs")
    print(f"Eval corpus: {len(eval_docs)} docs")
    print(f"Train questions: {len(train_questions)}")
    print(f"Eval questions: {len(test_questions)}")
    print(f"\nSample document: {all_documents[0][:200]}...")
    print(f"\nSample questions:")
    for i, q in enumerate(test_questions[:3], 1):
        print(f"  {i}. {q}")
    
    # Predict schema
    print("\n" + "="*80)
    print("STEP 2: Predicting schema from documents")
    print("="*80)
    
    schema = srag.predict_schema(schema_sample_docs)
    print(f"\nPredicted schema with {len(schema['properties'])} fields:")
    print(', '.join(schema['properties'].keys()))
    
    # Refine schema with questions
    print("\n" + "="*80)
    print("STEP 3: Refining schema with sample questions")
    print("="*80)
    
    refined_schema = srag.refine_schema_with_questions(
        schema,
        sample_test_questions,
        schema_sample_docs
    )
    
    print(f"\nFinal schema has {len(refined_schema['properties'])} fields:")
    print(', '.join(refined_schema['properties'].keys()))
    
    # Extract records
    print("\n" + "="*80)
    print("STEP 4: Extracting structured records from documents")
    print("="*80)
    
    records = srag.extract_records(all_documents)
    print(f"\nSample record:")
    print(json.dumps(records[0], indent=2))
    
    # Build database
    print("\n" + "="*80)
    print("STEP 5: Building SQL database")
    print("="*80)
    
    db_conn = srag.build_database()
    
    # Verify database
    import pandas as pd
    df = pd.read_sql_query("SELECT * FROM hotels LIMIT 3", db_conn)
    print("\nSample data from database:")
    print(df.to_string())
    
    # Calculate statistics
    print("\n" + "="*80)
    print("STEP 6: Calculating column statistics")
    print("="*80)
    
    stats = srag.calculate_statistics()
    print("\nSample statistics:")
    for field, stat in list(stats.items())[:3]:
        print(f"\n{field}: {stat}")
    
    # Test queries
    print("\n" + "="*80)
    print("STEP 7: Testing query pipeline")
    print("="*80)
    
    test_queries = [
        "How many hotels offer free WiFi?",
        "What is the average guest rating?",
        "Which hotels have both a swimming pool and spa?"
    ]
    
    results = []
    for query in test_queries:
        result = srag.answer_query(query)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("QUERY RESULTS SUMMARY")
    print("="*80)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['question']}")
        print(f"   → {r['answer']}")
    
    # Save artifacts
    print("\n" + "="*80)
    print("STEP 8: Saving artifacts")
    print("="*80)
    
    srag.save_artifacts(".")
    
    # Optional: Run evaluation
    print("\n" + "="*80)
    print("STEP 9: Evaluation (optional)")
    print("="*80)
    
    run_eval = input("\nRun evaluation on test questions? (y/n): ").lower().strip()
    if run_eval == 'y':
        accuracy = srag.evaluate_accuracy(eval_questions, limit=10)
        print(f"\nFinal accuracy: {accuracy:.1f}%")
    
    print("\n" + "="*80)
    print("S-RAG pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
