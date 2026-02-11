"""
Simple test script for S-RAG system.
Tests basic functionality without running the full pipeline.
"""

from srag import SRAG
import json


def test_basic_functionality():
    """Test basic S-RAG functionality with minimal data."""
    
    print("="*80)
    print("S-RAG BASIC FUNCTIONALITY TEST")
    print("="*80)
    
    # Initialize
    print("\n1. Initializing S-RAG...")
    srag = SRAG(model="gpt-4.1-mini")
    print("✓ S-RAG initialized successfully")
    
    # Test with sample hotel documents
    print("\n2. Testing with sample documents...")
    sample_docs = [
        """
        Grand Plaza Hotel
        Location: Sydney, Australia
        Rating: 4.5 stars
        Price: $250 per night
        Amenities: Free WiFi, Swimming Pool, Fitness Center, Restaurant
        Distance to city center: 2 km
        Guest Rating: 8.5/10
        """,
        """
        Ocean View Resort
        Location: Melbourne, Australia
        Rating: 5 stars
        Price: $350 per night
        Amenities: Free WiFi, Spa, Bar, Room Service, Beach Access
        Distance to city center: 5 km
        Guest Rating: 9.2/10
        """,
        """
        Budget Inn Downtown
        Location: Brisbane, Australia
        Rating: 3 stars
        Price: $120 per night
        Amenities: Free WiFi, Parking, Breakfast Included
        Distance to city center: 1 km
        Guest Rating: 7.8/10
        """
    ]
    
    print(f"✓ Loaded {len(sample_docs)} sample documents")
    
    # Test schema prediction
    print("\n3. Testing schema prediction...")
    try:
        schema = srag.predict_schema(sample_docs)
        print(f"✓ Schema predicted with {len(schema['properties'])} fields")
        print(f"  Fields: {', '.join(list(schema['properties'].keys())[:5])}...")
    except Exception as e:
        print(f"✗ Schema prediction failed: {e}")
        return False
    
    # Test record extraction
    print("\n4. Testing record extraction...")
    try:
        records = srag.extract_records(sample_docs[:2])  # Test with 2 docs
        print(f"✓ Extracted {len(records)} records")
        print(f"  Sample record keys: {list(records[0].keys())[:5]}...")
    except Exception as e:
        print(f"✗ Record extraction failed: {e}")
        return False
    
    # Test database building
    print("\n5. Testing database creation...")
    try:
        db_conn = srag.build_database()
        print("✓ Database created successfully")
    except Exception as e:
        print(f"✗ Database creation failed: {e}")
        return False
    
    # Test statistics calculation
    print("\n6. Testing statistics calculation...")
    try:
        stats = srag.calculate_statistics()
        print(f"✓ Statistics calculated for {len(stats)} fields")
    except Exception as e:
        print(f"✗ Statistics calculation failed: {e}")
        return False
    
    # Test query answering
    print("\n7. Testing query answering...")
    test_queries = [
        "How many hotels are in the database?",
        "What is the average price?"
    ]
    
    for query in test_queries:
        try:
            result = srag.answer_query(query, verbose=False)
            print(f"✓ Query: {query}")
            print(f"  Answer: {result['answer']}")
        except Exception as e:
            print(f"✗ Query failed: {e}")
    
    # Test artifact saving
    print("\n8. Testing artifact saving...")
    try:
        srag.save_artifacts(".")
        print("✓ Artifacts saved successfully")
    except Exception as e:
        print(f"✗ Artifact saving failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    
    return True


if __name__ == "__main__":
    import os
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run tests
    success = test_basic_functionality()
    exit(0 if success else 1)
