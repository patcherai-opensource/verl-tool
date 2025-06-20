import fire
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from transformers import AutoTokenizer
import json
import sys

def main(
    model_name: str,
    base_url: str,
    repo: str = "canvg/canvg",
    ref: str = "937668eced93e0335c67a255d0d2277ea708b2cb",
    api_key: str = "sk-proj-1234567890",
    temperature: float = 0.6,
    max_tokens: int = 30000,
    top_p: float = 0.95,
    n: int = 1,
):
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Create a test case similar to vulnerability POC dataset
    system_prompt = """You are a security researcher analyzing a codebase for vulnerabilities. 
You have access to a shell environment with the repository cloned. Use shell commands to explore the codebase.
To execute shell commands, wrap them in <shell_tool>...</shell_tool> tags."""

    user_prompt = """Please analyze the repository structure. First, check the current directory with 'pwd', 
then list the contents with 'ls'. Then, inspect the manifest files in the package in your current working directory.
Once you are confident, write a bash script that I could run to install the package. The entire script should be wrapped in <bash_script>...</bash_script> tags.
I will write these contents to a file and run it. NOTE: this should happen over multiple turns - do not try to solve this all at once - interact with the shell to collect information before providing your final response. Your response in each turn should only contain a single <shell_tool> block OR a single <bash_script> block."""

    # Create the extra_fields structure
    extra_fields = [{
        "repository_reference": {
            "repo": repo,
            "ref": ref,
        }
    }]

    messages = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt)
    ]

    print(f"Testing VMS shell tool rollout with model {model_name} at {base_url}", flush=True)
    print("\nSending request with repository info:", json.dumps(extra_fields, indent=2))
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            extra_body={"extra_infos": extra_fields}
        )
        
        print("\nFull conversation history:")
        if hasattr(completion, "conversation_history"):
            for i, msg in enumerate(completion.conversation_history):
                print(f"\nTurn {i}:")
                print(f"Role: {msg['role']}")
                print(f"Content: {msg['content']}")
        
        print("\nFinish reason:", completion.choices[0].finish_reason)
            
        # print("\nFinal response:")
        # content = completion.choices[0].message.content
        # print(content if content is not None else "No content returned")
        # Check if the final response contains a bash script
        # if content is not None and "<bash_script>" not in content:
        #     print("\nWarning: Final response does not contain a bash script")
            
    except Exception as e:
        print("\nError occurred:")
        print(f"Failed to process request: {str(e)}")
        
        # Print the exact error for debugging
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    fire.Fire(main)

"""
Example usage:
python eval_service/test/test_vms_shell.py \
    --model_name Qwen/Qwen3-8B \
    --base_url http://0.0.0.0:8000 \
    --repo canvg/canvg \
    --ref main
""" 