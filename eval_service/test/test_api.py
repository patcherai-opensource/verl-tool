system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test the code of your solution, include \"<|calling system for feedback|>\" at the end of your response for the current turn. Then the system will execute the code in the markdown block and provide the standard output and error. Make sure you also write test cases for the code you write so you can get non-empty execution results for debugging. If you think the solution is complete and don't need to test, don't include \"<|calling system for feedback|>\" in the response and put your final answer in a markdown code block like this: ```python\nyour code here\n``` without appending anything.
"""

math_problem = """### Progress Tracker\n\nYou are tasked with implementing a `ProgressTracker` class that monitors the progress of a data processing task. The task processes data in blocks, and each block can vary in size. The `ProgressTracker` should be able to handle updates about the number of blocks processed and the size of each block, and provide the current progress as a percentage of the total expected size.\n\n#### Class: ProgressTracker\n\n**Constructor**:\n- `__init__(self, total_size: int)`: Initializes the tracker with the total size of the task in units.\n\n**Methods**:\n- `update(self, blocks: int = 1, block_size: int = 1) -> None`: Updates the tracker with the number of blocks processed and the size of each block.\n  - `blocks`: Number of blocks processed in this update.\n  - `block_size`: Size of each block.\n- `get_progress(self) -> float`: Returns the current progress as a percentage of the total size, rounded to two decimal places.\n  - If the total size is reached or exceeded, return 100.00.\n\n#### Example\n\n```python\ntracker = ProgressTracker(100)\ntracker.update(blocks=2, block_size=10)\nprint(tracker.get_progress())  # Output: 20.00\n\ntracker.update(blocks=3, block_size=15)\nprint(tracker.get_progress())  # Output: 65.00\n\ntracker.update(blocks=2, block_size=20)\nprint(tracker.get_progress())  # Output: 100.00\n```\n\n#### Constraints:\n- `1 <= total_size <= 10^9`\n- `1 <= blocks, block_size <= 10^6`\n- The number of updates will not exceed 10^5.\n\n### Function Signature\n\n```python\nclass ProgressTracker:\n    def __init__(self, total_size: int):\n        pass\n\n    def update(self, blocks: int = 1, block_size: int = 1) -> None:\n        pass\n\n    def get_progress(self) -> float:\n        pass\n```
"""

from openai import OpenAI
client = OpenAI(api_key="sk-proj-1234567890", base_url="http://0.0.0.0:5000") # Replace with your local server address
model_name="VerlTool/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-mtrl-sys8-110-step"

completion = client.chat.completions.create(
    model=model_name,
    messages=[
		{
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": math_problem
        }
    ],
    temperature=0,
    max_tokens=2048,
    top_p=1,
    n=1,
)
print(completion.choices[0].message.content)

tempalte = """<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{math_problem}
<|im_end|>
<|im_start|>assistant
"""
completion = client.completions.create(
    model=model_name,
    prompt=tempalte.format(system_prompt=system_prompt, math_problem=math_problem),
    temperature=0,
    max_tokens=2048,
    top_p=1,
    n=1,
)
print(completion.choices[0].text)


completion = client.completions.create(
    model=model_name,
    prompt=f"system\n{system_prompt}\n\nuser\n{math_problem}\nassistant\n",
    temperature=0,
    max_tokens=1241241,
    top_p=1,
    n=1,
)
print(completion.choices[0].text)