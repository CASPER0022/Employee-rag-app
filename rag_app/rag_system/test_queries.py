#!/usr/bin/env python3
"""
Test script for RAG System
Demonstrates SQL vs RAG routing
"""

import sys
from pathlib import Path

# Add rag_system to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from query_llm import ask

def test_queries():
    """Test both SQL and RAG queries"""
    
    test_cases = [
        # SQL queries (analytical)
        ("How many employees are there?", "SQL"),
        ("What is the average salary?", "SQL"),
        ("Who are the top 5 highest paid employees?", "SQL"),
        ("How many employees in each department?", "SQL"),
        ("List all employees with salary above 100000", "SQL"),
        
        # RAG queries (specific lookups)
        ("What is John Smith's email?", "RAG"),
        ("Tell me about Sarah's role", "RAG"),
        ("What department is Michael in?", "RAG"),
    ]
    
    print("RAG System Test")
    print("=" * 60)
    print()
    
    for i, (question, expected_type) in enumerate(test_cases, 1):
        print(f"Test {i}: {question}")
        print(f"Expected: {expected_type}")
        
        try:
            result = ask(question)
            actual_type = result['query_type']
            status = "✓" if actual_type == expected_type else "✗"
            
            print(f"Actual: {actual_type} {status}")
            print(f"Answer: {result['answer'][:100]}...")
            
            if actual_type == "SQL" and 'sql' in result:
                print(f"SQL: {result['sql'][:80]}...")
            
            print()
        except Exception as e:
            print(f"ERROR: {e}")
            print()
    
    print("=" * 60)

if __name__ == "__main__":
    test_queries()
