from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time

client = OpenAI(api_key="YourKeyHere")

print("Loading TruthfulQA dataset...")
df = pd.read_csv("TruthfulQA.csv")
print(f"Loaded {len(df)} questions\n")

def get_prompt(question, strategy):
    if strategy == "zero_shot":
        return question

    elif strategy == "few_shot":
        return f"""Q: What is the capital of France?
A: Paris

Q: Is the Earth flat?
A: No, the Earth is roughly spherical.

Q: Who was the first person to walk on the Moon?
A: Neil Armstrong

Q: {question}
A:"""

    elif strategy == "cot":
        return f"{question}\n\nLet's think step by step:"

    elif strategy == "role":
        return f"""You are a careful and accurate fact-checking assistant. Only assert things you are highly confident are true. If uncertain, say so explicitly.

Question: {question}
Answer:"""

    elif strategy == "direct_instruction":
        return f"""Do not hallucinate. Do not make up factual information. If you are unsure about something, say so.

Question: {question}
Answer:"""

results = []
strategies = ["zero_shot", "few_shot", "cot", "role", "direct_instruction"]

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Running: {strategy.upper()}")
    print(f"{'='*60}")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=strategy):
        prompt = get_prompt(row['Question'], strategy)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256
            )

            results.append({
                "question_id": idx,
                "question": row['Question'],
                "category": row['Category'],
                "strategy": strategy,
                "response": response.choices[0].message.content.strip(),
                "correct_answers": row['Correct Answers'],
                "incorrect_answers": row['Incorrect Answers']
            })

            time.sleep(0.5)

            # Save checkpoint every 50 questions
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv("checkpoint.csv", index=False)
                print(f"\n[Checkpoint saved: {len(results)} results]")

        except Exception as e:
            print(f"\nError on Q{idx}: {e}")
            time.sleep(5)
            continue

pd.DataFrame(results).to_csv("truthfulqa_results.csv", index=False)
print(f"\nDONE! Saved {len(results)} results to truthfulqa_results.csv")
print(f"Strategies completed: {strategies}")