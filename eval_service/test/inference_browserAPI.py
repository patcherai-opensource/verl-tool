import pandas as pd
import concurrent.futures
from openai import OpenAI
import os
import tqdm
import numpy as np

# 配置
data_path = "/home/zhiheng/cogito/verl-tool/data/wikiQA_debug/test.parquet"
result_dir = "/home/zhiheng/cogito/verl-tool/eval_service/result"
os.makedirs(result_dir, exist_ok=True)

model_name = "/home/zhiheng/cogito/base_models/qwen2.5-3b-baseline-step10"
result_path = os.path.join(result_dir, f"{os.path.basename(model_name)}_result.csv")

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://0.0.0.0:5000")

df = pd.read_parquet(data_path)

def build_messages(row):
    # 这里可以根据实际情况调整prompt
    prompt = row['prompt']
    if isinstance(prompt, str):
        import ast
        prompt = ast.literal_eval(prompt)
    return prompt

def infer_one(row):
    messages = build_messages(row)
    extra_body = row['extra_info']
    if isinstance(extra_body, str):
        import ast
        extra_body = ast.literal_eval(extra_body)
    # 兼容 np.array
    if hasattr(extra_body, 'item'):
        extra_body = extra_body.item()
    # 递归将所有ndarray转为list
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray(x) for x in obj]
        else:
            return obj
    extra_body = convert_ndarray(extra_body)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=10240,
            top_p=1,
            n=1,
            extra_body=extra_body
        )
        content = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason
    except Exception as e:
        content = f"[ERROR]{str(e)}"
        finish_reason = "error"
    return {
        "id": row.get("id", None),
        "question": row.get("question", None),
        "golden_answers": row.get("golden_answers", None),
        "output": content,
        "finish_reason": finish_reason
    }

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(infer_one, row) for idx, row in df.iterrows()]
    for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="推理进度"):
        results.append(f.result())

# 保存结果
result_df = pd.DataFrame(results)
result_df.to_csv(result_path, index=False)
print(f"推理完成，结果已保存到 {result_path}")