import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import google.generativeai as genai
from tqdm import tqdm
import re
import time
import random
from dotenv import load_dotenv
from bson import ObjectId
import traceback

# ─── 🔐 Load Env & Setup Gemini ────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

# ─── 📡 Connect to MongoDB ─────────────────────────────────────────────────────
client = MongoClient(os.getenv("MONGO_URI_MIMIR"))
db = client["Docs"]
collection = db["documents"]

# ─── 🧾 Load previous progress ─────────────────────────────────────────────────
qa_file = "qa_dataset.csv"
processed_ids = set()

if os.path.exists(qa_file):
    old_df = pd.read_csv(qa_file)
    if "mongo_id" in old_df.columns:
        processed_ids = set(old_df["mongo_id"].dropna().astype(str).tolist())

# ─── 🧠 Fetch new documents ────────────────────────────────────────────────────
docs = list(collection.find({
    "summary_embedding": {"$exists": True},
    "_id": {"$nin": list(map(lambda x: ObjectId(x), processed_ids))}
}))

print(f"📄 Total documents fetched: {len(docs)}")

embeddings = np.array([doc["summary_embedding"] for doc in docs])
contents = [doc["content"] for doc in docs]

if embeddings.size == 0:
    raise ValueError("🚨 No new documents to process.")

def TokenCount(content):
    result = model.count_tokens(content)
    return result.total_tokens

# ─── 🔍 Optimal K using Silhouette Score ───────────────────────────────────────
def optimal_k(X, max_k=50):
    if len(X) < 3:
        return 1
    best_score, best_k = -1, 2
    for k in range(2, min(max_k + 1, len(X))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

n_clusters = optimal_k(embeddings, max_k=10)
print(f"🔍 Clustering into {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(embeddings)

# ─── 🧩 Organize Clusters ───────────────────────────────────────────────────────
cluster_map = {}
for idx, label in enumerate(clusters):
    cluster_map.setdefault(label, []).append(idx)

# ─── 🔄 Retry Logic with Key Exhaustion Handling ───────────────────────────────
def gemini_retry(prompt, retries=3, delay=2):
    for i in range(retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            if "quota" in str(e).lower() or "exceeded" in str(e).lower():
                print("🛑 API key exhausted. Saving progress and exiting...")
                raise SystemExit
            elif i == retries - 1:
                raise e
            time.sleep(delay * (2 ** i))

# ─── ✨ Helper Functions ────────────────────────────────────────────────────────
def clean_summary(text):
    return re.sub(r"[^\x00-\x7F]+", " ", text).strip()

def save_incremental(data, filepath):
    df = pd.DataFrame(data)
    if os.path.exists(filepath):
        old_df = pd.read_csv(filepath)
        df = pd.concat([old_df, df], ignore_index=True).drop_duplicates(subset=["mongo_id", "question"])
    df.to_csv(filepath, index=False)

# ─── 🚀 Cluster-based Golden Set Generation ────────────────────────────────────
qa_pairs = []

for label, indexes in tqdm(cluster_map.items(), desc="✨ Generating QAs per cluster"):
    cluster_docs = [docs[i] for i in indexes]
    cluster_contents = [clean_summary(docs[i]["content"]) for i in indexes if isinstance(docs[i]["content"], str)]
    if not cluster_contents:
        continue

    merged_content = "\n\n".join(cluster_contents)

    try:
        prompt = f"""
You are generating a golden Q&A dataset for benchmarking a retrieval-augmented generation system.
The input consists of multiple documents from the same cluster. Generate 4 factual questions and corresponding answers based on the ENTIRE CONTENT.

### Guidelines:
- Merge all relevant facts (e.g. all departments, eligibility, locations, dates) from the full content.
- Avoid pronouns or vague references (e.g. this/that/event).
- Questions must be specific and answerable based on the content.
- Use clear, formal, student-friendly tone.

### Output Format:
1. Q: <Question 1>
A: <Answer 1>
2. Q: <Question 2>
A: <Answer 2>
3. Q: <Question 3>
A: <Answer 3>
4. Q: <Question 4>
A: <Answer 4>

Documents:
{merged_content}
"""

        response = gemini_retry(prompt)
        text = response.text.strip()

        matches = re.findall(r"\d+\.\s+Q:\s*(.*?)\s*A:\s*(.*?)(?=\n\d+\.|$)", text, re.DOTALL)
        new_pairs = [
            {
                "mongo_id": str(docs[indexes[0]]["_id"]),
                "document_index": indexes[0],
                "cluster_id": label,
                "question": q.strip(),
                "answer": a.strip()
            } for q, a in matches
        ]

        qa_pairs.extend(new_pairs)
        save_incremental(new_pairs, qa_file)

    except SystemExit:
        break
    except Exception as e:
        print(f"❌ Error in cluster {label}: {e}")
        traceback.print_exc()

# ─── 💾 Final Save ─────────────────────────────────────────────────────────────
if qa_pairs:
    df = pd.read_csv(qa_file)
    df = df.sort_values(by=["cluster_id", "document_index"])
    df.to_csv(qa_file, index=False)
    df.to_json("qa_dataset.json", orient="records", indent=2)

print(f"✅ Finished. Total Q&A pairs now: {len(pd.read_csv(qa_file))}")
