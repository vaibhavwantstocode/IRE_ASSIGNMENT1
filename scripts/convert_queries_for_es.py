"""
Query Converter: SelfIndex Format → Elasticsearch Format

USAGE: Internal conversion when querying Elasticsearch
       NOT for creating separate query files!

This converter is used ON-THE-FLY when querying Elasticsearch to translate
our universal query format into ES Query DSL.

Universal Query File: queries/test_queries.txt (used by both systems)
- SelfIndex: Parses queries directly with custom parser
- Elasticsearch: Uses this converter to create Query DSL

Conversions:
- PHRASE "multi word" → ES match_phrase query
- "term1" AND "term2" → ES query_string with boolean operators
- Simple terms → ES match query
"""

import re
from typing import List
from pathlib import Path


class QueryConverter:
    """Converts between SelfIndex and Elasticsearch query formats"""
    
    @staticmethod
    def selfindex_to_elasticsearch(query: str) -> str:
        """
        Convert SelfIndex query format to Elasticsearch query_string format
        
        Conversions:
        - PHRASE "multi word phrase" → "multi word phrase" (ES phrase query)
        - "term1" AND "term2" → term1 AND term2 (ES query_string)
        - "term1" OR NOT "term2" → term1 OR NOT term2
        - quoted single terms → unquoted (ES doesn't need them)
        
        Args:
            query: SelfIndex format query string
            
        Returns:
            Elasticsearch-compatible query string
        """
        # Remove PHRASE keyword, keep the phrase content in quotes
        # PHRASE "machine learning" → "machine learning"
        es_query = re.sub(r'PHRASE\s+"([^"]+)"', r'"\1"', query)
        
        # Remove quotes around single terms with boolean operators
        # "government" AND "policy" → government AND policy
        es_query = re.sub(r'"([^"\s]+)"', r'\1', es_query)
        
        return es_query.strip()
    
    @staticmethod
    def to_elasticsearch_dsl(query: str) -> dict:
        """
        Convert to Elasticsearch Query DSL (JSON format)
        
        Returns proper Query DSL for different query types
        """
        # Check for phrase query
        phrase_match = re.search(r'PHRASE\s+"([^"]+)"', query)
        if phrase_match:
            phrase_text = phrase_match.group(1)
            return {
                "query": {
                    "match_phrase": {
                        "content": phrase_text
                    }
                }
            }
        
        # Check for boolean operators
        if any(op in query.upper() for op in ['AND', 'OR', 'NOT']):
            # Use query_string for boolean queries
            es_query = QueryConverter.selfindex_to_elasticsearch(query)
            return {
                "query": {
                    "query_string": {
                        "query": es_query,
                        "default_field": "content"
                    }
                }
            }
        
        # Simple term or multi-term query
        clean_query = query.strip('"')
        return {
            "query": {
                "match": {
                    "content": clean_query
                }
            }
        }


def convert_query_file(input_file: str, output_file: str):
    """
    Convert entire query file from SelfIndex to Elasticsearch format
    
    Args:
        input_file: Path to SelfIndex format queries
        output_file: Path to output ES-compatible queries
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    converted_count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            original = line.strip()
            if not original or original.startswith('#'):
                # Keep comments and blank lines
                fout.write(line)
                continue
            
            # Convert query
            converted = QueryConverter.selfindex_to_elasticsearch(original)
            fout.write(converted + '\n')
            converted_count += 1
    
    print(f"✓ Converted {converted_count} queries")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")


def show_conversion_examples():
    """Display example conversions for verification"""
    examples = [
        'PHRASE "machine learning"',
        'PHRASE "renewable energy technology development"',
        '"government" AND "policy"',
        '"apple" AND NOT "fruit"',
        '("machine" AND "learning") OR ("artificial" AND "intelligence")',
        'quantum mechanics',
        '"climate" OR "environment"',
        'PHRASE "natural language processing"',
    ]
    
    print("\n=== Query Conversion Examples ===\n")
    for original in examples:
        converted = QueryConverter.selfindex_to_elasticsearch(original)
        print(f"SelfIndex:      {original}")
        print(f"Elasticsearch:  {converted}")
        print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert SelfIndex queries to Elasticsearch format'
    )
    parser.add_argument(
        '--input',
        default='queries/test_queries.txt',
        help='Input query file (SelfIndex format)'
    )
    parser.add_argument(
        '--output',
        default='queries/test_queries_es.txt',
        help='Output query file (Elasticsearch format)'
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Show conversion examples'
    )
    
    args = parser.parse_args()
    
    if args.examples:
        show_conversion_examples()
    else:
        convert_query_file(args.input, args.output)
        print("\nUse --examples to see conversion examples")
