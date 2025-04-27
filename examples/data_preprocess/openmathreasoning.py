
import fire
import json
import datasets
import os
from transformers import AutoTokenizer

from tqdm import tqdm

def main(
    num_proc=32,
    dataset_path="nvidia/OpenMathReasoning",
    max_tokens=4096,
):
    
    dataset = datasets.load_dataset(dataset_path, split='tir')
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    def preprocess_func(item):
        item['solution_len'] = len(tokenizer.encode(item['generated_solution']))
        item['num_tool_calls'] = item['generated_solution'].count("<tool_call>")
        item['generated_solution'] = item['generated_solution'].replace("<tool_call>", "```python").replace("</tool_call>", "```")
        return item
    
    def filter_func(item):
        return item['solution_len'] < max_tokens
    
    dataset = dataset.map(preprocess_func, num_proc=num_proc)
    dataset.push_to_hub("VerlTool/openmathreasoning_tir", split="train")
    print(f"Dataset size after preprocessing: {len(dataset)}")
    print(f"Dataset size before filtering: {len(dataset)}")
    dataset = dataset.filter(filter_func, num_proc=num_proc)
    print(f"Dataset size after filtering: {len(dataset)}")
    print(f"Average solution length: {sum(dataset['solution_len']) / len(dataset)}")
    print(f"Average number of tool calls: {sum(dataset['num_tool_calls']) / len(dataset)}")
    dataset.push_to_hub("VerlTool/openmathreasoning_tir", "max_len_4096", split="train")
    with open("openmathreasoning_preprocessed.json", "w") as f:
        json.dump([x for x in dataset], f, indent=4)
    print("Preprocessed dataset saved to openmathreasoning_preprocessed.json")
        
if __name__ == "__main__":
    fire.Fire(main)