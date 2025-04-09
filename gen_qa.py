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
import asyncio
from functools import partial
from dotenv import load_dotenv
from bson import ObjectId
import traceback
import random
from datetime import datetime

# ─── 🕓 Logging Helper ──────────────────────────────────────────────
def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ─── 🔐 Load Env & Setup Gemini ─────────────────────────────────────
log("🔐 Loading environment variables and configuring Gemini...")
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

# ─── 📡 Connect to MongoDB ──────────────────────────────────────────
log("📡 Connecting to MongoDB...")
client = MongoClient(os.getenv("MONGO_URI_MIMIR"))
db = client["Docs"]
collection = db["documents"]

# ─── 💾 Load previous progress ──────────────────────────────────────
log("💾 Loading previous QA progress...")
qa_file = "qa_dataset.csv"
processed_ids = set()
if os.path.exists(qa_file):
    old_df = pd.read_csv(qa_file)
    if "mongo_id" in old_df.columns:
        processed_ids = set(old_df["mongo_id"].dropna().astype(str).tolist())

# ─── 🧠 Fetch new documents ─────────────────────────────────────────
log("🧠 Fetching new documents from DB...")
docs = list(collection.find({
    "summary_embedding": {"$exists": True},
    "_id": {"$nin": list(map(lambda x: ObjectId(x), processed_ids))}
}))

log(f"📄 Total documents fetched: {len(docs)}")

embeddings = np.array([doc["summary_embedding"] for doc in docs])
contents = [doc["content"] for doc in docs]


if embeddings.size == 0:
    raise ValueError("🚨 No new documents to process.")

# ─── 🔍 Optimal K using Silhouette Score ────────────────────────────
def optimal_k(X, max_k=50):
    log("🔍 Finding optimal number of clusters...")
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
    log(f"✅ Optimal clusters determined: {best_k}")
    return best_k

n_clusters = optimal_k(embeddings, max_k=10)
log(f"🔍 Clustering into {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(embeddings)

# ─── 🧩 Organize Clusters ───────────────────────────────────────────
log("🧩 Organizing documents into clusters...")
cluster_map = {}
for idx, label in enumerate(clusters):
    cluster_map.setdefault(label, []).append(idx)

# ─── 🔄 Async Rate-Limited Gemini Call ─────────────────────────────
semaphore = asyncio.Semaphore(1)
last_request_time = 0
rate_limit_interval = 15  # seconds

async def gemini_async(prompt):
    global last_request_time
    async with semaphore:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < rate_limit_interval:
            sleep_time = rate_limit_interval - elapsed
            log(f"⏳ Waiting {sleep_time:.2f}s to respect rate limit...")
            await asyncio.sleep(sleep_time)
        last_request_time = time.time()
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, partial(model.generate_content, prompt))
            log("✅ Gemini response received.")
            return response
        except Exception as e:
            log(f"💥 Gemini request error: {e}")
            raise

# ─── 🔁 Retry Wrapper ───────────────────────────────────────────────
async def gemini_async_retry(prompt, retries=5, delay=5):
    for i in range(retries):
        try:
            log(f"📡 Sending prompt to Gemini (attempt {i + 1})...")
            return await gemini_async(prompt)
        except Exception as e:
            if "quota" in str(e).lower() or "exceeded" in str(e).lower() or "429" in str(e):
                wait_time = delay * (2 ** i)
                log(f"🚩 Rate limit hit. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            elif i == retries - 1:
                log("❌ Final retry failed.")
                raise
            else:
                await asyncio.sleep(delay * (2 ** i))

# ─── ✨ Helper Functions ────────────────────────────────────────────
def TokenCount(content):
    try:
        result = model.count_tokens(content)
        return result.total_tokens
    except:
        return 0

def clean_summary(text):
    return re.sub(r"[^\x00-\x7F]+", " ", text).strip()

def save_incremental(data, filepath):
    log(f"💾 Saving {len(data)} new QA pairs to CSV...")
    df = pd.DataFrame(data)
    if os.path.exists(filepath):
        old_df = pd.read_csv(filepath)
        df = pd.concat([old_df, df], ignore_index=True).drop_duplicates(subset=["mongo_id", "question"])
    df.to_csv(filepath, index=False)

def chunk_by_tokens(contents, max_tokens=200000):
    log("📦 Chunking content by token count...")
    chunks = []
    current_chunk = []
    current_indices = []
    current_tokens = 0
    for idx, doc in enumerate(contents):
        doc = doc[:50000]
        tokens = TokenCount(doc)
        if tokens > max_tokens:
            continue
        if current_tokens + tokens <= max_tokens:
            current_chunk.append(doc)
            current_indices.append(idx)
            current_tokens += tokens
        else:
            chunks.append(("\n\n".join(current_chunk), current_indices))
            current_chunk = [doc]
            current_indices = [idx]
            current_tokens = tokens
    if current_chunk:
        chunks.append(("\n\n".join(current_chunk), current_indices))
    log(f"✅ Total chunks created: {len(chunks)}")
    return chunks

# ─── 🚀 Cluster-based Golden Set Generation ─────────────────────────
qa_pairs = []

async def generate_qa_from_chunk(merged_content, cluster_id, doc_indices, chunk_id):
    log(f"🚀 Generating Q&A for cluster {cluster_id}, chunk {chunk_id}...")
    prompt = f"""
You are generating high-quality, factual question-answer (QA) pairs to evaluate a retrieval-augmented generation (RAG) system.

The input is a chunk of academic or technical content. Based ONLY on the information in the chunk, generate 1–2 precise and unambiguous QA pairs.

### Strict Guidelines:
- Do NOT use pronouns (e.g., “this,” “they,” “it,” “these”) in questions or answers.
- Each question must be **explicit** and **fully self-contained**, referencing all necessary context.
- Answers must be **factually correct** and directly supported by the content.
- Avoid speculation, summarization, or external knowledge.
- Use a formal, student-friendly tone appropriate for academic use.

### Output Format:
1. Q: <Fully self-contained question 1>
A: <Direct, factual answer 1>
2. Q: <Fully self-contained question 2>
A: <Direct, factual answer 2>

Chunk:
{merged_content}
"""
    try:
        response = await gemini_async_retry(prompt)
        text = response.text.strip()
        matches = re.findall(r"\d+\.\s+Q:\s*(.*?)\s*A:\s*(.*?)(?=\n\d+\.|$)", text, re.DOTALL)
        return [
            {
                "mongo_id": str(docs[doc_indices[0]]["_id"]) if doc_indices else None,
                "document_index": doc_indices[0] if doc_indices else -1,
                "cluster_id": cluster_id,
                "chunk_id": chunk_id,
                "question": q.strip(),
                "answer": a.strip()
            } for q, a in matches
        ]
    except Exception as e:
        log(f"❌ Error in cluster {cluster_id}, chunk {chunk_id}: {e}")
        traceback.print_exc()
        return []

async def process_clusters():
    log("🔄 Starting cluster-based QA generation...")

    for label, indexes in tqdm(cluster_map.items(), desc="✨ Generating QAs per cluster"):
        cluster_docs = [docs[i] for i in indexes]
        cluster_contents = [clean_summary(doc["content"]) for doc in cluster_docs if isinstance(doc["content"], str)]

        if not cluster_contents:
            continue

        # 🧪 Randomly select up to 5 documents
        doc_sample = random.sample(list(enumerate(cluster_contents)), min(5, len(cluster_contents)))
        sampled_contents = [doc for _, doc in doc_sample]
        sampled_indices = [indexes[i] for i, _ in doc_sample]

        # 📦 Chunk the sampled docs
        content_chunks = chunk_by_tokens(sampled_contents, max_tokens=200000)

        tasks = [
            generate_qa_from_chunk(chunk, label, sampled_indices[:len(doc_indices)], i)
            for i, (chunk, doc_indices) in enumerate(content_chunks)
        ]
        results = await asyncio.gather(*tasks)

        for qa_list in results:
            if qa_list:
                qa_pairs.extend(qa_list)
                save_incremental(qa_list, qa_file)

        await asyncio.sleep(3)


# ─── ↻ Run async processing ────────────────────────────────────────
log("🚀 Launching async QA generation process...")
asyncio.run(process_clusters())

# ─── 💾 Final Save ────────────────────────────────────────────────
if qa_pairs:
    log("📊 Saving final dataset to CSV and JSON...")
    df = pd.read_csv(qa_file)
    df = df.sort_values(by=["cluster_id", "document_index", "chunk_id"])
    df.to_csv(qa_file, index=False)
    df.to_json("qa_dataset.json", orient="records", indent=2)

log(f"✅ Finished. Total Q&A pairs now: {len(pd.read_csv(qa_file))}")
