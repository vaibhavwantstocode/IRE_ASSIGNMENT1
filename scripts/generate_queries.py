"""
Query Set Generator for IR Assignment
Generates 256 diverse queries testing different system properties

Usage:
    python scripts/generate_queries.py
    python scripts/generate_queries.py --count 512 --output queries/large_set.txt
"""

import json
import random
import argparse
from pathlib import Path

# Query categories and examples
QUERY_CATEGORIES = {
    # Category 1: Single-term queries (simple, test index lookup)
    "single_term_common": [
        "technology", "science", "government", "education", "health",
        "economy", "sports", "music", "film", "history",
        "computer", "research", "election", "climate", "covid",
        "energy", "culture", "society", "politics", "business"
    ],
    
    "single_term_rare": [
        "nanotechnology", "cryptocurrency", "genomics", "quantum",
        "blockchain", "metaverse", "pandemic", "semiconductor",
        "renewable", "sustainability", "biotechnology", "robotics",
        "cybersecurity", "telemedicine", "bioinformatics"
    ],
    
    # Category 2: Two-term queries (test AND operation implicitly)
    "two_term_related": [
        "machine learning", "artificial intelligence", "climate change",
        "renewable energy", "public health", "space exploration",
        "quantum computing", "data science", "cyber security",
        "global warming", "electric vehicles", "neural networks",
        "social media", "financial markets", "natural language",
        "deep learning", "computer vision", "software engineering",
        "genetic engineering", "stem cells", "particle physics",
        "sustainable development", "digital transformation", "cloud computing"
    ],
    
    "two_term_unrelated": [
        "apple tree", "bank river", "python language", "turkey country",
        "mouse computer", "java island", "cricket sport", "golf club",
        "Mars planet", "Amazon forest", "windows operating", "jaguar animal"
    ],
    
    # Category 3: Boolean AND queries (explicit)
    "boolean_and_simple": [
        '"machine" AND "learning"',
        '"climate" AND "change"',
        '"artificial" AND "intelligence"',
        '"quantum" AND "computing"',
        '"renewable" AND "energy"',
        '"data" AND "science"',
        '"neural" AND "network"',
        '"deep" AND "learning"',
        '"cyber" AND "security"',
        '"space" AND "exploration"',
        '"genetic" AND "engineering"',
        '"social" AND "media"',
        '"cloud" AND "computing"',
        '"natural" AND "language"'
    ],
    
    "boolean_and_complex": [
        '"machine" AND "learning" AND "algorithm"',
        '"climate" AND "change" AND "policy"',
        '"artificial" AND "intelligence" AND "ethics"',
        '"renewable" AND "energy" AND "solar"',
        '"data" AND "science" AND "python"',
        '"quantum" AND "computing" AND "cryptography"',
        '"neural" AND "network" AND "deep"',
        '"cyber" AND "security" AND "threat"'
    ],
    
    # Category 4: Boolean OR queries
    "boolean_or": [
        '"covid" OR "pandemic"',
        '"technology" OR "innovation"',
        '"climate" OR "environment"',
        '"election" OR "voting"',
        '"economy" OR "finance"',
        '"sports" OR "athletics"',
        '"music" OR "concert"',
        '"film" OR "movie"',
        '"research" OR "study"',
        '"government" OR "policy"',
        '"science" OR "technology"',
        '"education" OR "learning"',
        '"health" OR "medical"',
        '"culture" OR "society"'
    ],
    
    # Category 5: Boolean NOT queries
    "boolean_not": [
        '"apple" AND NOT "fruit"',
        '"python" AND NOT "snake"',
        '"java" AND NOT "island"',
        '"mars" AND NOT "candy"',
        '"amazon" AND NOT "river"',
        '"windows" AND NOT "glass"',
        '"turkey" AND NOT "bird"',
        '"jaguar" AND NOT "animal"',
        '"virus" AND NOT "computer"',
        '"mercury" AND NOT "planet"',
        '"delta" AND NOT "airline"',
        '"corona" AND NOT "beer"'
    ],
    
    # Category 6: Complex Boolean queries
    "boolean_complex": [
        '("machine" AND "learning") OR ("artificial" AND "intelligence")',
        '("climate" OR "environment") AND "policy"',
        '("renewable" OR "solar") AND "energy" AND NOT "fossil"',
        '("data" AND "science") OR ("machine" AND "learning")',
        '("quantum" OR "classical") AND "computing"',
        '("neural" AND "network") OR ("deep" AND "learning")',
        '("covid" OR "pandemic") AND "vaccine" AND NOT "misinformation"',
        '("election" AND "voting") OR ("democracy" AND "government")',
        '("artificial" OR "machine") AND "intelligence" AND NOT "human"',
        '("cyber" AND "security") OR ("data" AND "privacy")',
        '("genetic" OR "gene") AND "engineering" AND NOT "clone"',
        '("space" AND "exploration") OR ("astronomy" AND "research")'
    ],
    
    # Category 7: Phrase queries (test position information)
    "phrase_simple": [
        'PHRASE "machine learning"',
        'PHRASE "climate change"',
        'PHRASE "artificial intelligence"',
        'PHRASE "renewable energy"',
        'PHRASE "quantum computing"',
        'PHRASE "neural network"',
        'PHRASE "deep learning"',
        'PHRASE "public health"',
        'PHRASE "cyber security"',
        'PHRASE "data science"',
        'PHRASE "space exploration"',
        'PHRASE "genetic engineering"'
    ],
    
    "phrase_long": [
        'PHRASE "natural language processing"',
        'PHRASE "global climate change policy"',
        'PHRASE "renewable energy technology development"',
        'PHRASE "artificial intelligence ethics concerns"',
        'PHRASE "machine learning algorithm optimization"',
        'PHRASE "quantum computing cryptography applications"'
    ],
    
    # Category 8: Domain-specific queries (News)
    "news_politics": [
        "election results", "presidential campaign", "senate vote",
        "supreme court", "foreign policy", "diplomatic relations",
        "trade agreement", "immigration policy", "legislative session",
        "congressional hearing", "political debate", "voter turnout",
        "cabinet nomination", "executive order"
    ],
    
    "news_business": [
        "stock market", "economic growth", "inflation rate",
        "unemployment data", "corporate earnings", "merger acquisition",
        "cryptocurrency price", "federal reserve", "interest rates",
        "market volatility", "GDP growth", "trade deficit",
        "consumer confidence", "quarterly earnings"
    ],
    
    "news_sports": [
        "championship game", "olympic medal", "world cup",
        "basketball playoffs", "football season", "tennis tournament",
        "marathon race", "swimming competition", "soccer match",
        "baseball series", "hockey finals", "golf championship",
        "track field", "winter olympics"
    ],
    
    "news_technology": [
        "tech innovation", "smartphone release", "software update",
        "data breach", "tech startup", "app launch",
        "cybersecurity threat", "cloud service", "AI breakthrough",
        "5G network", "autonomous vehicle", "tech conference"
    ],
    
    # Category 9: Domain-specific queries (Wikipedia)
    "wiki_science": [
        "photosynthesis process", "black hole theory", "DNA structure",
        "periodic table", "chemical reaction", "gravitational wave",
        "quantum mechanics", "theory relativity", "atomic structure",
        "cellular biology", "molecular chemistry", "particle accelerator",
        "nuclear fusion", "electromagnetic spectrum"
    ],
    
    "wiki_history": [
        "world war", "ancient civilization", "industrial revolution",
        "cold war", "renaissance period", "french revolution",
        "roman empire", "medieval europe", "american revolution",
        "cultural revolution", "age enlightenment", "great depression",
        "victorian era", "ancient egypt"
    ],
    
    "wiki_geography": [
        "mountain range", "ocean current", "tropical rainforest",
        "desert climate", "river delta", "volcanic activity",
        "tectonic plate", "continental drift", "glacier formation",
        "coral reef", "savanna ecosystem", "polar region",
        "island formation", "canyon erosion"
    ],
    
    "wiki_technology": [
        "internet protocol", "computer architecture", "operating system",
        "database management", "network topology", "programming language",
        "web development", "mobile computing", "distributed system",
        "encryption algorithm", "compiler design", "software testing"
    ],
    
    # Category 10: Multi-word technical queries
    "technical_cs": [
        "convolutional neural network", "gradient descent optimization",
        "recurrent neural network", "support vector machine",
        "random forest algorithm", "k-means clustering",
        "natural language processing", "computer vision system",
        "reinforcement learning agent", "transformer architecture",
        "attention mechanism model", "backpropagation algorithm"
    ],
    
    "technical_science": [
        "photovoltaic cell efficiency", "nuclear fusion reaction",
        "genetic engineering technique", "stem cell research",
        "particle accelerator physics", "superconductor material",
        "crispr gene editing", "nanotechnology application",
        "quantum entanglement phenomenon", "dark matter theory"
    ],
    
    # Category 11: Trending topics (realistic user queries)
    "trending_recent": [
        "chatgpt ai", "generative ai", "climate summit",
        "semiconductor shortage", "inflation economy",
        "covid variant", "space telescope", "electric car",
        "renewable transition", "chip manufacturing",
        "ai regulation", "quantum breakthrough", "metaverse platform",
        "crypto regulation", "green hydrogen"
    ],
    
    # Category 12: Long-tail queries (rare combinations)
    "long_tail": [
        "renewable energy policy impact climate",
        "machine learning healthcare diagnosis accuracy",
        "quantum computing cryptography security implications",
        "artificial intelligence bias ethics concerns",
        "blockchain technology supply chain transparency",
        "genetic engineering crop yield improvement",
        "neural network image recognition performance",
        "natural language processing sentiment analysis",
        "deep learning autonomous vehicle safety",
        "cyber security threat detection system"
    ],
    
    # Category 13: Information seeking (realistic patterns)
    "informational": [
        "how does photosynthesis work", "what is quantum computing",
        "why climate change important", "when was internet invented",
        "who discovered penicillin", "where is great barrier reef",
        "define machine learning", "explain neural network",
        "describe renewable energy", "compare solar wind energy"
    ],
    
    # Category 14: Mixed complexity
    "mixed_simple_boolean": [
        '"government" OR "administration"',
        '"research" AND "development"',
        '"education" OR "training"',
        '"health" AND "wellness"',
        '"culture" OR "tradition"',
        '"innovation" AND "technology"'
    ]
}

def generate_query_set(total_queries=256, seed=42):
    """Generate diverse queries with balanced distribution"""
    random.seed(seed)
    queries = []
    metadata = []
    
    # Calculate distribution across categories
    total_categories = len(QUERY_CATEGORIES)
    base_per_category = total_queries // total_categories
    extra = total_queries % total_categories
    
    category_list = list(QUERY_CATEGORIES.items())
    random.shuffle(category_list)  # Randomize category order
    
    for i, (category, query_list) in enumerate(category_list):
        # Number of queries from this category
        n = base_per_category + (1 if i < extra else 0)
        
        # Sample queries (with replacement if needed)
        if len(query_list) >= n:
            sampled = random.sample(query_list, n)
        else:
            # If not enough queries, cycle through the list
            sampled = (query_list * ((n // len(query_list)) + 1))[:n]
        
        for q in sampled:
            queries.append(q)
            metadata.append({
                "query": q,
                "category": category,
                "length": len(q.split()),
                "has_boolean": any(op in q.upper() for op in ["AND", "OR", "NOT"]),
                "has_phrase": "PHRASE" in q.upper(),
                "has_parens": "(" in q,
                "is_single_term": len(q.split()) == 1 and '"' not in q
            })
    
    # Shuffle to mix categories
    combined = list(zip(queries, metadata))
    random.shuffle(combined)
    queries, metadata = zip(*combined)
    
    return list(queries), list(metadata)

def save_queries(queries, metadata, output_dir="queries"):
    """Save queries and metadata to files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save queries as plain text (one per line)
    query_file = output_path / "test_queries.txt"
    with open(query_file, 'w', encoding='utf-8') as f:
        for q in queries:
            f.write(f"{q}\n")
    
    # Save metadata as JSON
    metadata_file = output_path / "query_metadata.json"
    stats = calculate_statistics(metadata)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_queries": len(queries),
            "generation_seed": 42,
            "categories": list(QUERY_CATEGORIES.keys()),
            "statistics": stats,
            "queries": metadata
        }, f, indent=2)
    
    return query_file, metadata_file, stats

def calculate_statistics(metadata):
    """Calculate query set statistics"""
    total = len(metadata)
    single_term = sum(1 for m in metadata if m['is_single_term'])
    multi_term = sum(1 for m in metadata if m['length'] > 1 and not m['has_boolean'] and not m['has_phrase'])
    boolean = sum(1 for m in metadata if m['has_boolean'])
    phrase = sum(1 for m in metadata if m['has_phrase'])
    complex_queries = sum(1 for m in metadata if m['has_parens'])
    avg_length = sum(m['length'] for m in metadata) / total if total > 0 else 0
    
    # Category distribution
    category_counts = {}
    for m in metadata:
        cat = m['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return {
        "total": total,
        "single_term": single_term,
        "multi_term": multi_term,
        "boolean": boolean,
        "phrase": phrase,
        "complex": complex_queries,
        "avg_length": round(avg_length, 2),
        "category_distribution": category_counts
    }

def print_statistics(stats):
    """Print formatted statistics"""
    print(f"\n{'='*60}")
    print(f"QUERY SET STATISTICS")
    print(f"{'='*60}")
    print(f"Total queries:          {stats['total']}")
    print(f"Single-term queries:    {stats['single_term']:4d} ({100*stats['single_term']/stats['total']:.1f}%)")
    print(f"Multi-term queries:     {stats['multi_term']:4d} ({100*stats['multi_term']/stats['total']:.1f}%)")
    print(f"Boolean queries:        {stats['boolean']:4d} ({100*stats['boolean']/stats['total']:.1f}%)")
    print(f"Phrase queries:         {stats['phrase']:4d} ({100*stats['phrase']/stats['total']:.1f}%)")
    print(f"Complex queries:        {stats['complex']:4d} ({100*stats['complex']/stats['total']:.1f}%)")
    print(f"Average query length:   {stats['avg_length']} words")
    print(f"{'='*60}")
    
    print(f"\nCategory Distribution:")
    print(f"{'-'*60}")
    for category, count in sorted(stats['category_distribution'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {category:30s} {count:4d} queries")
    if len(stats['category_distribution']) > 10:
        print(f"  ... and {len(stats['category_distribution']) - 10} more categories")

def main():
    parser = argparse.ArgumentParser(description="Generate diverse IR test queries")
    parser.add_argument("--count", type=int, default=256,
                       help="Number of queries to generate (default: 256)")
    parser.add_argument("--output", type=str, default="queries",
                       help="Output directory (default: queries/)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    print(f"Generating {args.count} diverse queries...")
    queries, metadata = generate_query_set(total_queries=args.count, seed=args.seed)
    
    print(f"Saving to {args.output}/...")
    query_file, metadata_file, stats = save_queries(queries, metadata, args.output)
    
    print(f"\n✓ Generated {len(queries)} queries")
    print(f"✓ Saved to: {query_file}")
    print(f"✓ Metadata: {metadata_file}")
    
    print_statistics(stats)
    
    # Print sample queries
    print(f"\nSample Queries (first 10):")
    print(f"{'-'*60}")
    for i, q in enumerate(queries[:10], 1):
        print(f"{i:3d}. {q}")
    
    print(f"\n{'='*60}")
    print(f"READY TO USE!")
    print(f"{'='*60}")
    print(f"Run evaluation with: python evaluate.py --queries {query_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
