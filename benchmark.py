import aiohttp
import asyncio
import csv
import os
import json
import time
import random
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# 🔐 Load Gemini API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
judge_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

INPUT_CSV = "qa_dataset.csv"
OUTPUT_CSV = "results.csv"

# 📝 Setup logger (file only, no console spam)
logger = logging.getLogger("benchmark_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("benchmark.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 📊 Concurrency control
semaphore = asyncio.Semaphore(2)

# 🧹 Completed cache
completed_questions = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        completed_questions.update(row["question"] for row in reader)

# 🧠 Load dataset
dataset = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row.get("question", "").strip()
        a = row.get("answer", "").strip()
        if q and a and q not in completed_questions:
            dataset.append({"question": q, "ground_truth": a})


def extract_json_from_gemini(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines[1:] if not line.strip().startswith("```") and not line.strip().startswith("json"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from Gemini:\n{text}")
        return None


def gemini_judge_retry(prompt, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            response = judge_model.generate_content(prompt)
            raw_output = response.text.strip()
            return extract_json_from_gemini(raw_output)
        except Exception as e:
            if "quota" in str(e).lower():
                raise RuntimeError("Token quota exhausted.")
            elif attempt == max_retries - 1:
                raise
            jitter = random.uniform(0, 1)
            sleep_time = base_delay * (2 ** attempt) + jitter
            time.sleep(sleep_time)


async def evaluate_qa(session, question, ground_truth, max_retries=5, base_delay=2):
    url = "http://127.0.0.1:8000/api/chat/qa"
    params = {"query": question}

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, params=params, timeout=60) as resp:
                    if resp.status != 200:
                        if resp.status == 429 or "quota" in await resp.text().lower():
                            raise RuntimeError("Token quota exhausted.")
                        raise Exception(f"HTTP {resp.status}")

                    data = await resp.json()
                    prediction = data[0].get("response_text", "")

                    judge_prompt = f"""
You are an academic evaluator. Based on the following:

- A question asked by a student,
- The correct answer according to university documentation,
- The predicted answer generated by a system,

Evaluate how well the predicted answer responds to the question and aligns with the correct answer.

### Instructions:
- Focus on factual accuracy, completeness, and relevance.
- Return only a JSON object with two fields:
  {{
    "score": int (1 to 5),
    "explanation": "Brief justification for the score"
  }}

### Question:
{question}

### Ground Truth Answer:
{ground_truth}

### Predicted Answer:
{prediction}
"""

                    eval_result = gemini_judge_retry(judge_prompt)
                    if not eval_result:
                        return None

                    result = {
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "score": eval_result.get("score", 0),
                        "explanation": eval_result.get("explanation", "")
                    }
                    logger.info(f"[✓] Evaluated: {question[:60]}... → Score: {result['score']}")
                    return result

            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as e:
                jitter = random.uniform(0, 1)
                sleep_time = base_delay * (2 ** attempt) + jitter
                logger.warning(f"[Retry] {attempt + 1}/{max_retries} for: {question[:40]}... ({e})")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(f"[ERROR] {question[:40]}... -> {e}")
                if "quota" in str(e).lower():
                    raise RuntimeError("Token quota exhausted.")
                return None

        logger.error(f"[FAIL] {question[:40]}... Max retries reached.")
        return None


async def run_benchmark():
    if not dataset:
        logger.info("All questions already processed.")
        return

    logger.info(f"Starting benchmark for {len(dataset)} questions...")

    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [evaluate_qa(session, item["question"], item["ground_truth"]) for item in dataset]

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if result:
                    results.append(result)
            except RuntimeError as token_exhaustion:
                logger.critical(str(token_exhaustion))
                logger.info("Saving progress. Please re-run when tokens are replenished.")
                break

    if results:
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "prediction", "score", "explanation"])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(results)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
