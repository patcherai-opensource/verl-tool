
import fire
import json
import datasets
import openai
import os
import time
import openai
import threading
from llm_engines import LLMEngine
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from tqdm import tqdm

def call_openai_model(prompt: str, model_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Call OpenAI API with a single prompt.
    
    Args:
        prompt: The text prompt to send to the model
        model_name: The name of the OpenAI model to use
        api_key: Optional API key (will use environment variable if not provided)
        
    Returns:
        The API response as a dictionary
    """
    # Use provided API key or get from environment
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Please provide it as an argument or set OPENAI_API_KEY environment variable.")
    
    try:
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return {
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "success": True,
            "model": model_name,
            "usage": response.usage,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "response": None,
            "success": False,
            "error": str(e),
            "model": model_name,
            "timestamp": time.time()
        }

def process_prompts_with_openai(
    prompts: List[str], 
    model_name: str, 
    max_workers: Optional[int] = None,
    api_key: Optional[str] = None,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple prompts in parallel using threading with tqdm progress bar.
    
    Args:
        prompts: List of text prompts to send to the model
        model_name: The name of the OpenAI model to use
        max_workers: Maximum number of parallel threads (defaults to number of CPUs * 5)
        api_key: Optional API key (will use environment variable if not provided)
        show_progress: Whether to show a progress bar (default: True)
        
    Returns:
        List of API responses
    """
    # Thread-safe list to store results in order
    results_lock = threading.Lock()
    results_list = [None] * len(prompts)
    
    # Counter for completed tasks (for progress bar)
    completed = 0
    completed_lock = threading.Lock()
    
    # Default to number of CPUs * 5 for I/O bound tasks if not specified
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 5)
    
    # Set up progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(prompts), desc="Processing prompts")
    
    def process_prompt(index, prompt):
        nonlocal completed
        try:
            result = call_openai_model(prompt, model_name, api_key)
            # Store the result in the correct position
            with results_lock:
                results_list[index] = result
        except Exception as e:
            # Handle any unexpected exceptions
            with results_lock:
                results_list[index] = {
                    "prompt": prompt,
                    "response": None,
                    "success": False,
                    "error": f"Thread execution failed: {str(e)}",
                    "model": model_name,
                    "timestamp": time.time()
                }
        finally:
            # Update progress bar
            if show_progress:
                with completed_lock:
                    completed += 1
                    pbar.update(1)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each prompt with its index
        futures = [executor.submit(process_prompt, i, prompt) for i, prompt in enumerate(prompts)]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This ensures we catch any exceptions
    
    # Close the progress bar
    if pbar:
        pbar.close()
    
    # Return the results in the original order
    return [result for result in results_list if result is not None]

prompt_template = """
I have the following message history, which is used to train a coder model to better refine its responses based on each turn's user feedback. I want to transform the entire conversation into a thinking process, where the model learns to generate both the code and the user's feedback as internal self-reflection.

Here are the requirements:
1. Only transform the messages by adding connection words like "however", "wait", "but", "therefore", etc. Do **not** change the main contents of either the feedback or the original responses.
2. For each assistant message, only add connection words, change the feedback from user into self-reflection, and add the "output" markdown block for the execution result
3. For each user message, if it's an "Execution result: ...", then simply put it in the "output" markdown block **directly** after the corresponding code block like "```python ... ```\n```output ... ```".
4. For each user message, if it's just an general feedback, then transform it into a self-reflection instead of a feedback from user.
5. In the transformed response, always use "I" to refer to yourself; never refer to "assistant.". never say 'feedback from user', always say in the self-reflection way like 'However, I think ...', 'wait, there is a problem ...', 'Given the execution result, I think ...', etc.
6. Don't summarize the execution result, just copy it as is.
7. Not every code block has a corresponding execution result, so for the code block **without** corresponding execution result, **don't** add the "output" markdown block.

Here is the original question:
{original_question}

Here is the full message history you need to transform:
{message_history}

### Now transform the above message history into a single thinking process:

Transformed thinking process as a single response:
(Your transformed thinking process here)
Okay, the user ...
"""
def main(
    model_name="gpt-4o-mini",
    num_proc=32,
    dataset_path="m-a-p/Code-Feedback",
):
    
    dataset = datasets.load_dataset(dataset_path, split='train')
    
    # filter to keep only items with execution result
    has_execution_dataset = []
    for item in dataset:
        if any("Execution result:" in message['content'] for message in item['messages']) and any('python' in message['content'] for message in item['messages']):
            has_execution_dataset.append(item)
    print(f"Keeping {len(has_execution_dataset)}/{len(dataset)} items with execution and python code.")
    dataset = has_execution_dataset
    
    original_questions = []
    message_histories = []
    for item in dataset:
        messages = item['messages']
        original_questions.append(f"{messages[0]['role']}: {messages[0]['content']}")
        message_history = ""
        for message in messages[1:]:
            message_history += f"{message['role']}: {message['content']}\n"
        message_histories.append(message_history)
        
    
    prompts = [prompt_template.format(original_question=original_question, message_history=message_history) for original_question, message_history in zip(original_questions, message_histories)]
    
    # llm = LLMEngine()
    # llm.load_model(model_name, num_workers=num_proc, engine='openai')
    # results = llm.batch_call_model(model_name, prompts, num_proc=num_proc, disable_batch_api=True)
    
    results = process_prompts_with_openai(
        prompts=prompts,
        model_name=model_name,
        max_workers=num_proc,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    results = [result['response'] if result['success'] else None for result in results]
    
    thinking_processes = []
    for result in results:
        # keep from Okay
        okay_idx = result.find("Okay, the user") if result else -1
        if okay_idx != -1:
            result = result[okay_idx:]
        thinking_processes.append(result)
    
    final_dataset = [
        {
            'id': dataset[i]['id'],
            'messages': [
                {
                    'role': 'user',
                    'content': dataset[i]['messages'][0]['content']
                },
                {
                    'role': 'assistant',
                    'content': thinking_process
                }
            ],
            'original_message': dataset[i]['messages'],
        } for i, thinking_process in enumerate(thinking_processes) if thinking_process is not None
    ]
    final_dataset = datasets.Dataset.from_list(final_dataset)
    def get_invalid(item):
        ori_num_output = len([x for x in item['original_message'] if "Execution result:" in x['content'] and x['role'] == 'user'])
        new_num_output = item['messages'][1]['content'].count("```output") if item['messages'][1]['content'] else 0
        item['ori_num_output'] = ori_num_output
        item['new_num_output'] = new_num_output
        item['valid'] = ori_num_output == new_num_output
        return item
    final_dataset = final_dataset.map(get_invalid, num_proc=num_proc)
    valid_dataset = final_dataset.filter(lambda x: x['valid'])
    invalid_dataset = final_dataset.filter(lambda x: not x['valid'])
    print(f"Valid dataset: {len(valid_dataset)}, Invalid dataset: {len(invalid_dataset)}")
    with open("invalid_thinking_processes.json", "w") as f:
        json.dump([x for x in invalid_dataset], f, indent=4)
    print("Invalid dataset saved to invalid_thinking_processes.json")
    with open("valid_thinking_processes.json", "w") as f:
        json.dump([x for x in valid_dataset], f, indent=4)
    print("Valid dataset saved to valid_thinking_processes.json")
        
if __name__ == "__main__":
    fire.Fire(main)