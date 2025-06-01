#!/usr/bin/env python
"""Test cases for the firejail sandbox environment.

Test coverage:
1. Execution Test - Basic command execution in sandbox environment
2. Return Result Test - STDOUT, STDERR
2. Timeout Test - Handling of long-running process termination
3. Modules Test - Verification of essential math package availability, e.g. numpy, pandas, sympy, scipy, etc.
4. Multiprocess Press Test - Stability under concurrent process execution
"""
import json
import requests
import fire
import logging
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm


# {"trajectory_id": "6d8d07eb-598c-4823-bce7-fd24e26e0d5d", "metadata": {"turns": 3}, "previous_obs": [{"action": "def find_least_recent_backup(files: List[Tuple[str, str]]) -> Tuple[str, str]:", "is_valid": true, "observation": "Error:\nFile \"<string>\", line 2\n    def find_least_recent_backup(files: List[Tuple[str, str]]) -> Tuple[str, str]:\n                                                                                  ^\nIndentationError: expected an indented block after function definition on line 2", "extra_field": {"finish": false}}, {"action": "Input: files = [\n    ('file1.txt', '2023-01-01 10:00:00'),\n    ('file2.txt', '2023-01-02 09:00:00'),\n    ('file3.txt', '2022-12-31 23:59:59')\n]\nOutput: ('file3.txt', '2022-12-31 23:59:59')\n\nInput: files = [\n    ('file1.txt', '2023-01-01 10:00:00'),\n    ('file2.txt', '2023-01-02 09:00:00'),\n    ('file3.txt', '2023-01-01 10:00:00')\n]\nOutput: ('file3.txt', '2023-01-01 10:00:00')\n\nInput: files = [\n    ('file1.txt', '2023-01-01 10:00:00'),\n    ('file2.txt', '2023-01-02 09:00:00'),\n    ('file3.txt', '2023-01-02 09:00:00')\n]\nOutput: ('file2.txt', '2023-01-02 09:00:00')", "is_valid": true, "observation": "Error:\nFile \"<string>\", line 2\n    Input: files = [\n    ^\nIndentationError: expected an indented block after function definition on line 1", "extra_field": {"finish": false}}, {"action": "Input: files = [\n    ('file1.txt', '2023-01-01 10:00:00'),\n    ('file2.txt', '2023-01-02 09:00:00'),\n    ('file3.txt', '2022-12-31 23:59:59')\n]\nOutput: ('file3.txt', '2022-12-31 23:59:59')\n\nInput: files = [\n    ('file1.txt', '2023-01-01 10:00:00'),\n    ('file2.txt', '2023-01-02 09:00:00'),\n    ('file3.txt', '2023-01-01 10:00:00')\n]\nOutput: ('file3.txt', '2023-01-01 10:00:00')\n\nInput: files = [\n    ('file1.txt', '2023-01-01 10:00:00'),\n    ('file2.txt', '2023-01-02 09:00:00'),\n    ('file3.txt', '2023-01-02 09:00:00')\n]\nOutput: ('file2.txt', '2023-01-02 09:00:00')", "is_valid": true, "observation": "", "extra_field": {"finish": false}}]}

# Add parent directory to path to import FirejailPythonCodeTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..tools.firejail_python_code import FirejailPythonCodeTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_firejail_python(
    url: str = None,
    trajectory_id: str = "test-firejail-python-001",
):
    """Test Firejail Python code execution with various test cases"""
    
    print("--- Test 1: Taco test cases ---")
    action = "```python\nimport math\n\ndef race(v1, v2, g):\n\tif v2 < v1:\n\t\treturn None\n\tseconds = 0.1\n\twhile v1 / 3600 * seconds + g >= v2 / 3600 * seconds:\n\t\tseconds += 0.05\n\thours = seconds / 3600\n\thoursRest = seconds % 3600\n\tminutes = hoursRest / 60\n\tseconds = hoursRest % 60\n\treturn [math.floor(hours), math.floor(minutes), math.floor(seconds)]\n\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"fn_name": "race", "inputs": [[720, 850, 70], [80, 91, 37], [80, 100, 40], [720, 850, 37], [720, 850, 370], [120, 850, 37], [820, 850, 550], [820, 81, 550]], "outputs": [[[0, 32, 18]], [[3, 21, 49]], [[2, 0, 0]], [[0, 17, 4]], [[2, 50, 46]], [[0, 3, 2]], [[18, 20, 0]], [null]]}']}))

def test_modules():
    """Test importing essential scientific and data analysis modules"""
    print("\n[4] Essential Modules Test")
    
    tool = FirejailPythonCodeTool()
    tests = {
        "numpy": "import numpy as np; print('numpy imported')",
        "pandas": "import pandas as pd; print('pandas imported')",
        "sympy": "import sympy; print('sympy imported')",
        "scipy": "import scipy; print('scipy imported')",
        "sklearn": "import sklearn; print('sklearn imported')"
    }
    
    results = {}
    for name, code in tests.items():
        print(f"  - {name}:")
        action = f"<python>{code}</python>"
        observation, _, _ = tool.conduct_action(f"module-test-{name}", action, {})
        print("    Output:", observation)
        results[name] = observation
    
    return results

def test_multiprocess():
    """Test running multiple code executions in parallel to stress test the system"""
    print("\n[5] Multiprocess Stress Test")
    
    tool = FirejailPythonCodeTool()
    codes = ["import numpy as np\nprint(np.ones((1024, 1024)).sum())"] * 1024
    results = []
    
    with ThreadPoolExecutor(max_workers=max(cpu_count() // 2, 1)) as pool:
        futures = []
        for i, code in enumerate(codes):
            action = f"<python>{code}</python>"
            futures.append(pool.submit(
                tool.conduct_action, 
                f"multiprocess-test-{i}", 
                action, 
                {}
            ))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            observation, done, valid = future.result()
            assert "ERROR" not in observation and "Exception" not in observation, observation
            results.append({
                "observation": observation,
                "done": done,
                "valid": valid
            })
    
    print(f"Successfully completed {len(results)} parallel executions")
    return results
    
def _send_test_request(url, trajectory_id, action, test_name, extra_field=None):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    if extra_field is None:
        extra_field = {}
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        **extra_field,
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for error status codes
        
        result = response.json()
        logger.info(f"Response received for {test_name} test")
        
        # Print observation
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            logger.info(f"\n--- {test_name} Result ---\n{observation}\n")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}

def test_firejail_python_direct():
    """Run direct tests on the FirejailPythonCodeTool without using the server API"""
    logger.info("Testing FirejailPythonCodeTool directly...")
    
    tool = FirejailPythonCodeTool()
    test_cases = [
        {
            "name": "Simple Print",
            "action": "<python>print('Direct test')</python>",
            "extra_field": {}
        },
        {
            "name": "Basic Math",
            "action": "```python\nprint(2 + 2)\n```",
            "extra_field": {}
        },
        {
            "name": "With Input",
            "action": "<python>name = input('Name? '); print(f'Hello {name}')</python>",
            "extra_field": {"stdin": "World"}
        }
    ]
    
    results = {}
    for test_case in test_cases:
        logger.info(f"Running direct test: {test_case['name']}")
        observation, done, valid = tool.conduct_action(
            "direct-test", 
            test_case['action'], 
            test_case['extra_field']
        )
        results[test_case['name']] = {
            "observation": observation,
            "done": done,
            "valid": valid
        }
        logger.info(f"\n--- {test_case['name']} Direct Result ---\n{observation}\n")
    
    return results

def test_firejail_with_matplotlib():
    """Test Firejail Python with matplotlib plotting"""
    action = """<python>
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='sin(x)')
    plt.title('Sine Function')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    
    # Try to save the plot
    try:
        plt.savefig('sine.png')
        print("Plot saved successfully")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    print("Matplotlib test completed")
except ImportError:
    print("Matplotlib not available in this environment")
</python>"""
    
    tool = FirejailPythonCodeTool()
    observation, done, valid = tool.conduct_action("matplotlib-test", action, {})
    logger.info(f"\n--- Matplotlib Test Result ---\n{observation}\n")
    
    return {
        "observation": observation,
        "done": done,
        "valid": valid
    }

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_firejail_python_code_with_test_tool  firejail --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_firejail_python_code_with_test_tool  direct
        python -m verl_tool.servers.tests.test_firejail_python_code_with_test_tool  matplotlib
        python -m verl_tool.servers.tests.test_firejail_python_code_with_test_tool  timeout
        python -m verl_tool.servers.tests.test_firejail_python_code_with_test_tool  modules
        python -m verl_tool.servers.tests.test_firejail_python_code_with_test_tool  multiprocess
    """
    fire.Fire({
        "firejail": test_firejail_python,
        "direct": test_firejail_python_direct,
        "matplotlib": test_firejail_with_matplotlib,
        "modules": test_modules,
        "multiprocess": test_multiprocess,
    })

if __name__ == "__main__":
    main()