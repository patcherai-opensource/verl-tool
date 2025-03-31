# Contain functions to create dataset from raw data
# create_dataset.py

import os
import csv
import argparse
import random

from datasets import Dataset

import json

TEMPLATES = {
    'qwen-instruct': {
        "system":(
        # ---- system message ----
        "<|im_start|>system\n"
        "Here's the information you'll have:\n"
        "The user's objective: This is the task you're trying to complete.\n"
        "The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\n"
        "The current web page's URL: This is the page you're currently navigating.\n"
        "The open tabs: These are the tabs you have open.\n"
        "The previous action: This is the action you just performed. It may be helpful to track your progress.\n\n"

        "The actions you can perform fall into several categories:\n\n"

        "Page Operation Actions:\n"
        "`click [id]`: This action clicks on an element with a specific id on the webpage.\n"
        "`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n"
        "`hover [id]`: Hover over an element with id.\n"
        "`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n"
        "`scroll [down|up]`: Scroll the page up or down.\n\n"

        "Tab Management Actions:\n"
        "`new_tab`: Open a new, empty browser tab.\n"
        "`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n"
        "`close_tab`: Close the currently active tab.\n\n"

        "URL Navigation Actions:\n"
        "`goto [url]`: Navigate to a specific URL.\n"
        "`go_back`: Navigate to the previously viewed page.\n"
        "`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\n"

        "Completion Action:\n"
        "`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\n"

        "To be successful, it is very important to follow the following rules:\n"
        "1. You should only issue an action that is valid given the current observation.\n"
        "2. You should only issue one action at a time.\n"
        "3. You should follow the examples to reason step by step and then issue the next action.\n"
        "4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.\n"
        "5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences. For example:\n"
        "   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>\n"
        "   ```click [1234]```\n"
        "6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.\n"
        "<|im_end|>\n"
        """7. Always format actions correctly: 
```command [parameters]```
For example, if searching for "death row inmates in the US" in a search field with ID `21`, correctly format it as:
```type [21] [death row inmates in the US] [1]```
Avoid incorrect formats that omit brackets around parameters or numeric values.\n\n"""
        ),
        # ---- user message ----
        "user":("<|im_start|>user\n"
        "Objective: {objective}\n\n"
        "URL: {url}\n"
        "Observation:\n"
        "{observation}\n"
        "Parsed Previous Action:\n"
        "{previous_action}\n"
        "<|im_end|>\n\n"
        ),
        "assistant":"<|im_start|>assistant {pred}\n<|im_end|>",
     },
}

def main():
    parser = argparse.ArgumentParser(description="Generate puzzle dataset from CSV.")
    parser.add_argument("--output_dir", type=str, default="data/wikiQA", help="Output directory.")
    # parser.add_argument("--train_size", type=int, default=900000, help="How many training instances to take.")
    # parser.add_argument("--test_size", type=int, default=1500, help="How many test instances to take.")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from datasets import load_dataset

    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
    print(dataset)

    def build_dataset(data_split, split_name):
        """
        data_split: 对应 train_data / dev_data / test_data
        split_name: 'train' / 'dev' / 'test'
        """
        instances = []
        index_offset = 0

        for row_i, row in enumerate(data_split):
            question = row["question"]
            golden_answers = row["golden_answers"]
            if len(golden_answers) == 0:
                answer = "N/A"
            else:
                answer = golden_answers[0]

            system_prompt = (
                "Here's the information you'll have:\n"
                "The user's objective: This is the task you're trying to complete.\n"
                "The current web page's accessibility tree: This is a simplified representation of the webpage,\n"
                "  providing key information.\n"
                "The current web page's URL: This is the page you're currently navigating.\n"
                "The open tabs: These are the tabs you have open.\n"
                "The previous action: This is the action you just performed.\n\n"
                "The actions you can perform fall into several categories:\n\n"

                "Page Operation Actions:\n"
                "`click [id]`: This action clicks on an element with a specific id on the webpage.\n"
                "`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id.\n"
                "  By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n"
                "`hover [id]`: Hover over an element with id.\n"
                "`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n"
                "`scroll [direction=down|up]`: Scroll the page up or down.\n\n"

                "Tab Management Actions:\n"
                "`new_tab`: Open a new, empty browser tab.\n"
                "`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n"
                "`close_tab`: Close the currently active tab.\n\n"

                "URL Navigation Actions:\n"
                "`goto [url]`: Navigate to a specific URL.\n"
                "`go_back`: Navigate to the previously viewed page.\n"
                "`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\n"

                "Completion Action:\n"
                "`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a\n"
                "text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete,\n"
                "provide the answer as \"N/A\" in the bracket.\n\n"

                "To be successful, it is very important to follow the following rules:\n"
                "1. You should only issue an action that is valid given the current observation.\n"
                "2. You should only issue one action at a time.\n"
                "3. You should follow the examples to reason step by step and then issue the next action.\n"
                "4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.\n"
                "5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences.\n"
                "   For example:\n"
                "   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>\n"
                "   ```click [1234]```\n"
                "6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.\n"
            )

            user_prompt = (
                "OBJECTIVE: {question}\n\n"
                "URL: {url}\n"
                f"OBSERVATION:\n"
                "{observation}\n"
            )
            instance = {
                "data_source": "wiki_qa",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "ability": "wiki",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": golden_answers,
                },
                "extra_info": {
                    "split": split_name,
                    # "seed": row_i + index_offset,
                    "index": row_i + index_offset,
                    "id": row_i + index_offset,
                    "question": question,
                    "golden_answers": golden_answers,
                    "selected_answer": answer,
                }
            }

            instances.append(instance)
            index_offset += 1

        return instances

    train_data = dataset["train"]
    dev_data = dataset["dev"]
    test_data = dataset["test"]

    train_instances = build_dataset(train_data, "train")
    dev_instances = build_dataset(dev_data, "dev")
    test_instances = build_dataset(test_data, "test")

    train_dataset = Dataset.from_list(train_instances)
    dev_dataset = Dataset.from_list(dev_instances)
    test_dataset = Dataset.from_list(test_instances)

    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    dev_dataset.to_parquet(os.path.join(args.output_dir, "dev.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))

    print("Done! Train size:", len(train_dataset), "Dev size:", len(dev_dataset), "Test size:", len(test_dataset))

if __name__ == "__main__":
    main()
