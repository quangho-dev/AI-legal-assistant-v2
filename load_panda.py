import pandas as pd

df = pd.read_csv("/Users/admin/Documents/Personal projects/AI assistant/evals/experiments/20260430-150406_naiverag.csv")
df["faithfulness_score"] = pd.to_numeric(df["faithfulness_score"], errors="coerce")
avg_score = df["faithfulness_score"].mean()
print(f"Average Faithfulness Score: {avg_score:.2f}")