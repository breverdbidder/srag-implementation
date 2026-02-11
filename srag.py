"""
S-RAG: Structured Retrieval Augmented Generation

Replicates the S-RAG system from "Structured RAG for Answering Aggregative Questions" 
(arXiv:2511.08505).

Pipeline: Schema Prediction → Record Extraction → SQL Database → Text-to-SQL → Answer Generation
"""

from openai import OpenAI
import pandas as pd
import sqlite3
import json
from typing import Any, Optional
from tqdm import tqdm
import requests
import os


class SRAG:
    """S-RAG system for structured retrieval augmented generation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-mini"):
        """
        Initialize S-RAG system.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.schema = None
        self.records = []
        self.db_conn = None
        self.statistics = {}
    
    @staticmethod
    def load_jsonl(url: str) -> list:
        """Download and parse JSONL file from URL."""
        print(f"Downloading {url.split('/')[-1]}...")
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        return [json.loads(line) for line in lines if line.strip()]
    
    def predict_schema(self, documents: list[str]) -> dict[str, Any]:
        """
        Predict JSON schema from documents using structured output.
        
        Args:
            documents: List of document texts to analyze
            
        Returns:
            JSON schema dictionary
        """
        docs_text = "\n\n---\n\n".join([f"Document {idx+1}:\n{doc}" 
                                        for idx, doc in enumerate(documents)])
        
        prompt = f"""Analyze these documents and create a comprehensive JSON schema capturing ALL recurring concepts and attributes.

Documents:
{docs_text}

Requirements:
- Focus on patterns appearing across multiple documents
- Avoid nested objects and arrays
- Use boolean fields instead of arrays of choices
- Avoid lengthy string fields (e.g., descriptions)
- Include locations, dates, numbers, and other structured data
- **CRITICAL**: Extract ALL amenity/facility fields as boolean (true/false):
  * Basic amenities: free_wifi, breakfast_included, parking, airport_shuttle, pet_friendly
  * Facilities: fitness_center, swimming_pool, spa, business_center, bar, restaurant
  * Services: laundry_service, room_service, dry_cleaning, concierge
  * Other: distance_to_city (number), distance_to_airport (number)
- If an amenity is mentioned in ANY document, include it in the schema
- For each field provide: type (string/integer/number/boolean), description, and examples

Return a JSON schema with this structure:
{{
  "title": "SchemaName",
  "type": "object",
  "properties": {{
    "fieldName": {{
      "type": "string",
      "description": "Detailed description",
      "examples": ["example1", "example2"]
    }}
  }},
  "required": ["fieldName"]
}}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a database schema expert. Return only valid JSON. Be comprehensive and include ALL amenities and attributes found in the documents."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")
        
        self.schema = json.loads(content)
        return self.schema
    
    def refine_schema_with_questions(self, schema: dict[str, Any], 
                                     sample_questions: list[str], 
                                     sample_docs: list[str]) -> dict[str, Any]:
        """
        Refine schema by analyzing what fields are needed to answer sample questions.
        
        Args:
            schema: Initial schema to refine
            sample_questions: Sample questions to analyze
            sample_docs: Sample documents for context
            
        Returns:
            Refined schema dictionary
        """
        current_fields = list(schema['properties'].keys())
        fields_str = ', '.join(current_fields)
        
        docs_sample = "\n\n".join([f"Doc {idx+1}: {doc[:500]}..." 
                                   for idx, doc in enumerate(sample_docs[:5])])
        questions_str = "\n".join([f"{idx+1}. {q}" 
                                   for idx, q in enumerate(sample_questions[:15])])
        
        prompt = f"""You have a schema with these fields: {fields_str}

Sample documents show accommodations with various attributes and amenities.

Sample Questions:
{questions_str}

Sample Documents:
{docs_sample}

Task: Identify fields mentioned in the questions that are MISSING from the current schema.

Look for:
- Amenities/facilities: bar, business_center, restaurant, laundry_service, dry_cleaning, room_service, concierge
- Measurements: distance_to_city, distance_to_airport, distance_to_beach
- Other attributes that questions ask about but aren't in the schema

For each missing field, provide:
- Field name (snake_case)
- Type (string/integer/number/boolean)
- Description

Return JSON:
{{
  "missing_fields": [
    {{
      "name": "field_name",
      "type": "boolean",
      "description": "Description of the field",
      "examples": [true, false]
    }}
  ],
  "reasoning": "Explanation of why these fields are needed"
}}

If no fields are missing, return empty "missing_fields" array."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a schema analysis expert. Identify missing fields needed to answer questions."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")
        
        result = json.loads(content)
        missing_fields = result.get('missing_fields', [])
        
        if missing_fields:
            print(f"Found {len(missing_fields)} missing fields:")
            for field in missing_fields:
                print(f"  - {field['name']} ({field['type']}): {field['description']}")
            
            # Add missing fields to schema
            for field in missing_fields:
                schema['properties'][field['name']] = {
                    'type': field['type'],
                    'description': field['description'],
                    'examples': field.get('examples', [])
                }
            
            print(f"\nReasoning: {result.get('reasoning', '')}")
        else:
            print("No missing fields found. Schema is complete!")
        
        self.schema = schema
        return schema
    
    def predict_record(self, document_text: str, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Extract structured record from document.
        
        Args:
            document_text: Text of document to extract from
            schema: Schema to use for extraction
            
        Returns:
            Extracted record dictionary
        """
        schema_info = []
        for field_name, field_spec in schema['properties'].items():
            examples_str = f" Examples: {', '.join(map(str, field_spec.get('examples', [])))}" if field_spec.get('examples') else ""
            schema_info.append(f"- {field_name} ({field_spec['type']}): {field_spec['description']}{examples_str}")
        
        prompt = f"""Extract data from this document according to the schema.

Document:
{document_text}

Schema:
{chr(10).join(schema_info)}

Rules:
- Use correct types (string, integer, float, boolean)
- Standardize values (e.g., "1M" → 1000000)
- Remove currency symbols (e.g., "$250" → 250)
- Use null for missing fields
- Return ONLY valid JSON matching the schema fields"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")
        
        return json.loads(content)
    
    def extract_records(self, documents: list[str], schema: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Extract records from all documents.
        
        Args:
            documents: List of document texts
            schema: Schema to use (defaults to self.schema)
            
        Returns:
            List of extracted records
        """
        schema = schema or self.schema
        if schema is None:
            raise ValueError("Schema must be provided or predicted first")
        
        self.records = []
        print(f"Extracting records from {len(documents)} documents...")
        
        for doc in tqdm(documents):
            record = self.predict_record(doc, schema)
            self.records.append(record)
        
        print(f"\nExtracted {len(self.records)} records")
        return self.records
    
    def build_database(self, schema: Optional[dict[str, Any]] = None, 
                      records: Optional[list[dict[str, Any]]] = None) -> sqlite3.Connection:
        """
        Create SQL database from schema and records.
        
        Args:
            schema: Schema to use (defaults to self.schema)
            records: Records to insert (defaults to self.records)
            
        Returns:
            SQLite connection
        """
        schema = schema or self.schema
        records = records or self.records
        
        if schema is None or not records:
            raise ValueError("Schema and records must be provided")
        
        type_mapping = {
            'string': 'TEXT',
            'integer': 'INTEGER',
            'number': 'REAL',
            'boolean': 'BOOLEAN'
        }
        
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create table
        properties = schema['properties']
        columns = [f"{name} {type_mapping.get(spec['type'], 'TEXT')}" 
                  for name, spec in properties.items()]
        cursor.execute(f"CREATE TABLE hotels ({', '.join(columns)})")
        
        # Insert records
        col_names = list(properties.keys())
        placeholders = ', '.join(['?' for _ in col_names])
        
        for record in records:
            values = [record.get(col) for col in col_names]
            cursor.execute(f"INSERT INTO hotels ({', '.join(col_names)}) VALUES ({placeholders})", values)
        
        conn.commit()
        print(f"Created database with {len(records)} records")
        
        self.db_conn = conn
        return conn
    
    def calculate_statistics(self, conn: Optional[sqlite3.Connection] = None, 
                            schema: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Calculate column statistics.
        
        Args:
            conn: Database connection (defaults to self.db_conn)
            schema: Schema to use (defaults to self.schema)
            
        Returns:
            Statistics dictionary
        """
        conn = conn or self.db_conn
        schema = schema or self.schema
        
        if conn is None or schema is None:
            raise ValueError("Database connection and schema must be provided")
        
        cursor = conn.cursor()
        statistics = {}
        
        for field_name, field_spec in schema['properties'].items():
            field_type = field_spec['type']
            
            try:
                if field_type in ['integer', 'number']:
                    cursor.execute(f"""
                        SELECT MIN({field_name}), MAX({field_name}), AVG({field_name}), COUNT({field_name})
                        FROM hotels WHERE {field_name} IS NOT NULL
                    """)
                    min_val, max_val, avg_val, count = cursor.fetchone()
                    statistics[field_name] = {
                        "type": "numeric",
                        "min": min_val,
                        "max": max_val,
                        "avg": round(avg_val, 2) if avg_val else None,
                        "count": count
                    }
                else:
                    cursor.execute(f"""
                        SELECT DISTINCT {field_name}, COUNT(*) as cnt
                        FROM hotels WHERE {field_name} IS NOT NULL
                        GROUP BY {field_name} ORDER BY cnt DESC LIMIT 20
                    """)
                    values = [row[0] for row in cursor.fetchall()]
                    statistics[field_name] = {
                        "type": "categorical",
                        "distinct_values": values,
                        "count": len(values)
                    }
            except Exception as e:
                print(f"Error calculating statistics for {field_name}: {e}")
        
        self.statistics = statistics
        return statistics
    
    def generate_sql(self, user_query: str, schema: Optional[dict[str, Any]] = None, 
                    statistics: Optional[dict[str, Any]] = None) -> tuple[str, str]:
        """
        Generate SQL query using JSON mode.
        
        Args:
            user_query: Natural language query
            schema: Schema to use (defaults to self.schema)
            statistics: Statistics to use (defaults to self.statistics)
            
        Returns:
            Tuple of (SQL query, reasoning)
        """
        schema = schema or self.schema
        statistics = statistics or self.statistics
        
        if schema is None:
            raise ValueError("Schema must be provided")
        
        type_mapping = {
            'string': 'TEXT',
            'integer': 'INTEGER',
            'number': 'REAL',
            'boolean': 'BOOLEAN'
        }
        
        # Build schema DDL
        ddl_lines = []
        for field_name, field_spec in schema['properties'].items():
            sql_type = type_mapping.get(field_spec['type'], 'TEXT')
            ddl_lines.append(f"  {field_name} {sql_type}  -- {field_spec['description']}")
        
        columns_str = ',\n'.join(ddl_lines)
        ddl = f"CREATE TABLE hotels (\n{columns_str}\n)"
        
        # Format statistics
        stats_lines = []
        for field_name, stats in statistics.items():
            if stats.get('type') == 'numeric':
                stats_lines.append(f"{field_name}: MIN={stats.get('min')}, MAX={stats.get('max')}, AVG={stats.get('avg')}")
            elif stats.get('type') == 'categorical':
                values = ', '.join(str(v) for v in stats.get('distinct_values', [])[:5])
                stats_lines.append(f"{field_name}: {values}...")
        
        stats_str = '\n'.join(stats_lines)
        
        prompt = f"""Translate this question into a SQLite query.

Question: {user_query}

Schema:
{ddl}

Statistics:
{stats_str}

Rules:
1. **SINGLE TABLE ONLY**: You must query ONLY the 'hotels' table. Do NOT hallucinate other tables like 'amenities' or 'reviews'.
2. **NO JOINS**: The schema is flat. Do not use JOIN.
3. **CHECK COLUMNS**: Use ONLY columns listed in the Schema above. If a column (e.g., 'bar') is not in the schema, you CANNOT use it in WHERE or SELECT.
4. **Boolean Logic**: 1=true, 0=false. To check if a service exists, use `service_name = 1`.
5. **Aggregations**: Use COUNT, AVG, MAX, MIN as needed.

Return JSON with "query" (the SQL) and "reasoning"."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a SQL expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")
        
        result = json.loads(content)
        return result['query'], result['reasoning']
    
    def generate_answer(self, user_query: str, sql_results: str) -> str:
        """
        Generate answer using JSON mode.
        
        Args:
            user_query: Original natural language query
            sql_results: SQL query results as string
            
        Returns:
            Natural language answer
        """
        prompt = f"""Answer based ONLY on the query results.

Question: {user_query}

Results:
{sql_results}

Provide a clear, concise answer with specific numbers from the results.
Return JSON with an "answer" field."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")
        
        result = json.loads(content)
        return result['answer']
    
    def answer_query(self, query: str, conn: Optional[sqlite3.Connection] = None,
                    schema: Optional[dict[str, Any]] = None,
                    stats: Optional[dict[str, Any]] = None,
                    verbose: bool = True) -> dict[str, Any]:
        """
        Full S-RAG pipeline: query → SQL → results → answer.
        
        Args:
            query: Natural language query
            conn: Database connection (defaults to self.db_conn)
            schema: Schema to use (defaults to self.schema)
            stats: Statistics to use (defaults to self.statistics)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with question, SQL, answer, and results
        """
        conn = conn or self.db_conn
        schema = schema or self.schema
        stats = stats or self.statistics
        
        if conn is None or schema is None:
            raise ValueError("Database connection and schema must be provided")
        
        if verbose:
            print(f"\n{'='*80}\nQuestion: {query}\n{'='*80}")
        
        # Generate SQL
        sql, reasoning = self.generate_sql(query, schema, stats)
        if verbose:
            print(f"\nSQL: {sql}")
            print(f"Reasoning: {reasoning}")
        
        # Execute
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
            
            result_str = json.dumps([dict(zip(cols, row)) for row in results], indent=2) if results else "No results"
            if verbose:
                print(f"\nResults:\n{result_str}")
            
            # Generate answer
            answer = self.generate_answer(query, result_str)
            if verbose:
                print(f"\nAnswer: {answer}")
            
            return {
                "question": query,
                "sql": sql,
                "answer": answer,
                "results": results
            }
        
        except Exception as e:
            if verbose:
                print(f"\nError: {e}")
            return {
                "question": query,
                "sql": sql,
                "error": str(e),
                "answer": f"Error: {e}"
            }
    
    def judge_answer(self, query: str, gold_answer: str, judged_answer: str) -> bool:
        """
        Use LLM to judge if judged_answer matches gold_answer.
        
        Args:
            query: Original query
            gold_answer: Ground truth answer
            judged_answer: Answer to evaluate
            
        Returns:
            True if correct, False otherwise
        """
        prompt = f"""<instructions>
You are given a query, a gold answer, and a judged answer.
Decide if the judged answer is a correct answer for the query, based
on the gold answer.
Do not use any external or prior knowledge. Only use the gold answer.
Answer true if the judged answer is a correct answer
for the query, and false otherwise.

<query>
{query}
</query>
<gold_answer>
{gold_answer}
</gold_answer>
<judged_answer>
{judged_answer}
</judged_answer>
</instructions>

Return JSON with:
- "is_correct": boolean (true if judged answer is correct, false otherwise)
- "reasoning": brief explanation of your decision"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an evaluation expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")
        
        result = json.loads(content)
        return result['is_correct']
    
    def evaluate_accuracy(self, questions_data: list[dict], limit: int = 20) -> float:
        """
        Evaluate S-RAG accuracy using Answer Comparison metric.
        
        Args:
            questions_data: List of dicts with 'question' and 'answer' keys
            limit: Maximum number of questions to evaluate
            
        Returns:
            Accuracy percentage
        """
        correct = 0
        total = 0
        
        print(f"Evaluating on up to {limit} questions...\n")
        
        for idx, item in enumerate(questions_data[:limit]):
            query = item['question']
            gold_answer = item['answer']
            
            # Skip if not hotel-related
            if 'hotel' not in query.lower():
                continue
            
            print(f"\n[{total + 1}] Question: {query}")
            print(f"Gold Answer: {gold_answer}")
            
            # Get S-RAG answer
            try:
                result = self.answer_query(query, verbose=False)
                judged_answer = result['answer']
                
                # Judge if correct
                is_correct = self.judge_answer(query, gold_answer, judged_answer)
                
                if is_correct:
                    correct += 1
                    print("✓ CORRECT")
                else:
                    print("✗ INCORRECT")
                
                total += 1
            
            except Exception as e:
                print(f"✗ ERROR: {e}")
                total += 1
            
            if total >= limit:
                break
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Correct: {correct}/{total}")
        print(f"S-RAG Accuracy: {accuracy:.1f}%")
        
        return accuracy
    
    def save_artifacts(self, output_dir: str = "."):
        """
        Save schema, records, and statistics to files.
        
        Args:
            output_dir: Directory to save files to
        """
        if self.schema:
            with open(f"{output_dir}/schema.json", "w") as f:
                json.dump(self.schema, f, indent=2)
            print(f"Saved schema to {output_dir}/schema.json")
        
        if self.records:
            with open(f"{output_dir}/records.json", "w") as f:
                json.dump(self.records, f, indent=2)
            print(f"Saved records to {output_dir}/records.json")
        
        if self.statistics:
            with open(f"{output_dir}/statistics.json", "w") as f:
                json.dump(self.statistics, f, indent=2)
            print(f"Saved statistics to {output_dir}/statistics.json")
