import pandas as pd
import requests
import json
import time
import os

# 全局参数（可后续改为命令行参数）
PARQUET_PATH = "/home/zhiheng/cogito/verl-tool/data/wikiQA_debug/dev.parquet"
MODEL_NAME = "/home/zhiheng/cogito/base_models/qwen2.5-3b-baseline-step10"
MAX_TOKENS = 10240
API_URL = "http://0.0.0.0:5000/chat/completions"

# 结果保存路径
model_name = os.path.basename(MODEL_NAME)
parquet_name = os.path.splitext(os.path.basename(PARQUET_PATH))[0]
result_dir = "leval_service/result"
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, f"{model_name}_{parquet_name}_result.csv")

# 读取数据
df = pd.read_parquet(PARQUET_PATH)

results = []

for idx, row in df.iterrows():
    # 构造messages
    prompt = row["prompt"]
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except Exception:
            prompt = [{'role': 'system', 'content': str(prompt)}]
    messages = prompt if isinstance(prompt, list) else [{'role': 'system', 'content': str(prompt)}]

    # 构造extra_body
    extra_body = row.get("extra_info", {})
    # numpy类型转list/dict
    if hasattr(extra_body, 'item'):
        extra_body = extra_body.item()
    if hasattr(extra_body, 'to_dict'):
        extra_body = extra_body.to_dict()
    if isinstance(extra_body, dict):
        for k, v in extra_body.items():
            if hasattr(v, 'item'):
                extra_body[k] = v.item()
            elif hasattr(v, 'tolist'):
                extra_body[k] = v.tolist()

    # 构造请求体
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "top_p": 1,
        "n": 1,
        "extra_body": extra_body
    }
    print(f"==== [Sample {idx}] ====")
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        output = data["choices"][0]["message"]["content"] if "choices" in data and data["choices"] else ""
        finish_reason = data["choices"][0].get("finish_reason", "") if "choices" in data and data["choices"] else ""
    except Exception as e:
        output = f"[Error] {e}"
        finish_reason = "error"
    results.append({
        "id": row.get("id", idx),
        "question": row.get("question", ""),
        "golden_answers": row.get("golden_answers", ""),
        "output": output,
        "finish_reason": finish_reason
    })
    time.sleep(1)

# 保存结果
df_result = pd.DataFrame(results)
df_result.to_csv(result_path, index=False)
print(f"推理完成，结果已保存到 {result_path}")
