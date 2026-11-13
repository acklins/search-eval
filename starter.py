from turtle import distance

import chromadb

#Define ids, documents, and metadatas lists
# ── Documents ───────────────────
ids = [
    "doc_01",
    "doc_02",
    "doc_03",
    "doc_04",
    "doc_05",
    "doc_06",
    "doc_07",
    "doc_08",
    "doc_09",
    "doc_10",
    "doc_11",
    "doc_12",
]

documents = [
       # Python fundamentals
    "Python lists store ordered collections of items and support indexing.",
    "Python dictionaries store data as key-value pairs for fast lookup.",
    "Python functions organize reusable code using parameters and return values.",

    # Pandas and data analysis
    "Pandas DataFrames organize tabular data into rows and columns.",
    "Pandas read_csv loads CSV files into a DataFrame for data analysis.",
    "Pandas groupby groups rows and calculates aggregate statistics.",

    # FastAPI and web APIs
    "FastAPI uses Pydantic models to validate incoming API request data.",
    "FastAPI supports GET and POST endpoints for building REST APIs.",
    "FastAPI raises HTTPException to return error responses such as 404.",

    # Embeddings and vector databases
    "Sentence Transformers convert text into numerical embedding vectors.",
    "ChromaDB stores embeddings and retrieves documents using similarity search.",
    "Cosine similarity measures how closely two embedding vectors align.",
]

metadatas = [
    {"topic": "python"},
    {"topic": "python"},
    {"topic": "python"},

    {"topic": "pandas"},
    {"topic": "pandas"},
    {"topic": "pandas"},

    {"topic": "fastapi"},
    {"topic": "fastapi"},
    {"topic": "fastapi"},

    {"topic": "embeddings"},
    {"topic": "embeddings"},
    {"topic": "embeddings"},
]

# ── ChromaDB setup ─────────────────
#Create an in-memory ChromaDB database
client = chromadb.EphemeralClient()  

# Createa collection for cosine distance
collection = client.get_or_create_collection(
    name="eval_collection", 
    metadata={"hnsw:space": "cosine"},
)

# Add all 12 documents with their IDs and metadata
collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
)

print(f"ChromaDB collection created with {collection.count()} documents.")

#Evaluation queries
eval_set = [
    {
        "query": "How do Python dictionaries store information?",
        "relevant_ids": ["doc_02"],
    },
    {
        "query": "How can I load and analyze CSV data using Pandas?",
        "relevant_ids": ["doc_04", "doc_05", "doc_06"],
    },
    {
        "query": "How does FastAPI validate incoming request data?",
        "relevant_ids": ["doc_07"],
    },
    {
        "query": "How do I create REST API endpoints and handle errors?",
        "relevant_ids": ["doc_08", "doc_09"],
    },
    {
        "query": "How do embeddings help find similar documents?",
        "relevant_ids": ["doc_10", "doc_11", "doc_12"],
    },
    {
        "query": "How do Python functions organize reusable code?",
        "relevant_ids": ["doc_03"],
    },
]


# ── Evaluation function ────────────
def evaluate(n_results: int, distance_threshold: float = None):
    """
    Evaluate semantic search using the precision and recall.

    Args:
        n_results: Maximum documents to retrieve per query.
        distance_threshold: Optional maximum cosine distance.
                            Lower distances indicate better matches.
    Returns:
        A tuple containing average precision and average recall.
    """
    print(
        f"\n=== Evaluation: n_results={n_results}, "
        f"threshold={distance_threshold} ==="
    )

    precisions = []
    recalls    = []

    for i, entry in enumerate(eval_set, 1):
        query        = entry["query"]
        relevant_ids = set(entry["relevant_ids"])

        # 1. Search the ChromaDB collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances"],
        )

        #2. Extract retrieved document IDs and distances
        retrieved_ids = results["ids"][0]
        distances = results["distances"][0]

        # 3. Apply the optional distance threshold

        if distance_threshold is not None:
            retrieved_ids = [
                doc_id
                for doc_id, distance in zip(retrieved_ids, distances)
                if distance <= distance_threshold
            ]

        # 4. Count correctly retrieved documents
        overlap = len(relevant_ids.intersection(retrieved_ids))

        #5. Calculate precision and recall
        precision = (
            overlap / len(retrieved_ids) 
            if retrieved_ids else 0.0
        )
        recall = (
            overlap / len(relevant_ids) 
            if relevant_ids else 0.0
        )

        # 6. Save scores for calculating averages
        precisions.append(precision)
        recalls.append(recall)

        #7. Display individual query results
        print(
            f"  Query {i}: "
            f"P={precision * 100:.1f}%  "
            f"R={recall * 100:.1f}%  "
            f"| '{query[:50]}'"
        )

        print(f"              Expected: {sorted(relevant_ids)}")
        print(f"              Retrieved: {sorted(retrieved_ids)}")

    #8. Calculate average precision and recall
    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    print(
        f"  AVERAGE: "
        f"P={avg_p * 100:.1f}% "
        f"R={avg_r * 100:.1f}%"
    )
    return avg_p, avg_r


# ── Run at 3 settings ─────────
#Call evaluate() with 3 different combinations of n_results / threshold

#Setting 1: Retrieve the top 3 documents
print("\nSETTING 1: Top 3 results")
p1, rl = evaluate(n_results=3)

#Setting 2: Retrieve the top 5 documents
print("\nSETTING 2: Top 5 results")
p2, r2 = evaluate(n_results=5)

#Setting 3: Retrieve up to 5 documents, 
#but exclude those with cosine distance above 0.5
print("\nSETTING 3: Top 5 results with distance threshold 0.5")
p3, r3 = evaluate(n_results=5, distance_threshold=0.5)

#Compare average presicion and recall
print("\n=== SETTINGS COMPARISON ===")
print(f"Setting 1 (Top 3):          P={p1*100:.1f}%  R={rl*100:.1f}%")
print(f"Setting 2 (Top 5):          P={p2*100:.1f}%  R={r2*100:.1f}%")
print(f"Setting 3 (Top 5, d<=0.5): P={p3*100:.1f}%  R={r3*100:.1f}%")

# ── Analysis ─────────────────────
print("\n=== ANALYSIS ===")

print("""
1. SEARCH QUALITY

The evaluation tested six queries against 12 documents
covering Python, Pandas, FastAPI, and embeddings.

With n_results=3, average precision was 61.1% and
average recall was 100.0%.

This means the system retrieved all expected relevant
documents, but some returned results were irrelevant.

2. EFFECT OF INCREASING N_RESULTS

Increasing n_results from 3 to 5 reduced average
precision from 61.1% to 36.7%.

Recall remained at 100.0%.

The additional documents introduced more irrelevant
results without finding any additional relevant ones.

3. EFFECT OF DISTANCE THRESHOLD

Using n_results=5 and distance_threshold=0.5 increased
average precision to 80.6%.

However, recall decreased to 75.0%.

The threshold successfully removed many irrelevant
results, but also excluded some relevant documents.

4. WHAT WORKED WELL

The top-3 setting achieved perfect average recall.
All six queries retrieved their expected relevant
documents within the first three results.

5. WHAT NEEDS IMPROVEMENT

The system returned too many irrelevant documents when
n_results increased to 5.

The strict distance threshold also removed some
relevant documents.

6. RECOMMENDED IMPROVEMENTS

- Test intermediate distance thresholds such as 0.6
  and 0.7 to balance precision and recall.
- Improve document descriptions with clearer wording.
- Expand the evaluation set with more challenging queries.
- Review the retrieved document IDs for each query.
- Consider reranking results to improve relevance.

CONCLUSION

For this evaluation set, n_results=3 provides the best
recall without returning as many irrelevant documents
as n_results=5.

The distance-filtered setting provides the highest
precision but sacrifices some recall.

7. INDIVIDUAL QUERY ANALYSIS

Query 1 - Python Dictionaries:
Precision was 50.0% and recall was 100.0%.
The correct document (doc_02) was retrieved, but doc_01
was also returned even though it was not labeled relevant.

Query 2 - Pandas CSV Analysis:
Precision was 100.0% and recall was 66.7%.
Documents doc_04 and doc_05 were retrieved correctly.
However, doc_06 was missed.

Query 3 - FastAPI Validation:
Precision was only 33.3%, while recall was 100.0%.
The correct document (doc_07) was found, but doc_08
and doc_09 were also returned.

Query 4 - REST API Endpoints:
Precision was 100.0%, but recall was only 50.0%.
The system retrieved doc_08 but missed doc_09,
which explains FastAPI error handling.

Query 5 - Embeddings and Similarity:
Precision was 100.0%, but recall was only 33.3%.
The system retrieved doc_11 but missed doc_10
and doc_12.

Query 6 - Python Functions:
Precision and recall were both 100.0%.
The system correctly retrieved doc_03 without
including irrelevant documents.

8. FINAL RECOMMENDATIONS

Query 3 needs better precision.
A stricter distance threshold or improved ranking
could help exclude unrelated FastAPI documents.

Query 5 needs better recall.
A less restrictive distance threshold may help
retrieve the missing embedding documents.

Query 6 demonstrates that the search system works
very well when the query closely matches the
content of a specific document.

I would test additional thresholds between 0.5
and 0.7 to find a better balance between
precision and recall.
""")
