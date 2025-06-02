#!/usr/bin/env python
"""Test cases for the firejail_python_code_with_verl_code_test.py.

This test ensures that the new verl_code_test tool works exactly 
like the original test_tool while using the same input parameters.
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..tools.firejail_python_code_with_verl_code_test import FirejailPythonCodeWithVerlCodeTest as FirejailPythonCodeWithTestTool


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

    print("--- Test 2: Taco test cases without fn_name---")
    action = "```python\ndef sub(maxs, mins):\n\tfor i in range(len(maxs)):\n\t\tif maxs[i] != mins[i]:\n\t\t\tif i == len(maxs) - 1:\n\t\t\t\treturn int(maxs[i]) - int(mins[i])\n\t\t\tif i == len(maxs) - 2:\n\t\t\t\treturn int(maxs[i:i + 2]) - int(mins[i:i + 2])\n\t\t\treturn 10\n\treturn 0\n\ndef checkEqual(S):\n\tans = 8\n\tfor k in range(1, len(S)):\n\t\tif len(S) % k != 0:\n\t\t\tcontinue\n\t\tmins = maxs = S[0:k]\n\t\tfor s in range(0, len(S), k):\n\t\t\tmaxs = max(maxs, S[s:s + k])\n\t\t\tmins = min(mins, S[s:s + k])\n\t\tans = min(ans, sub(maxs, mins))\n\treturn ans\n\ndef check12(S):\n\tmaxv = 0\n\tminv = 10\n\tp = 0\n\twhile p < len(S):\n\t\tv = int(S[p])\n\t\tif S[p] == '1' and p + 1 < len(S):\n\t\t\tv = 10 + int(S[p + 1])\n\t\t\tp += 1\n\t\tmaxv = max(maxv, v)\n\t\tminv = min(minv, v)\n\t\tp += 1\n\treturn maxv - minv\nS = input()\nprint(min(checkEqual(S), check12(S)))\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["9714431", "16612328", "23422731", "754526", "955577", "75547", "2112", "799", "88", "32523857", "4787", "1859551", "135661", "3675", "156692", "167918384", "83994", "4837847", "14513597", "15282598", "12659326", "1468417", "6280", "115464", "52376853", "2315", "3641224", "97187", "836", "195884", "36250", "2427817", "17598762", "5744554", "9295", "129848", "3863342", "3743", "133862", "1237", "1625", "1179729", "12651", "3776912", "4829", "73", "2228", "2546", "3136", "138", "3380", "4828", "3652", "5667", "7275", "774", "9329", "279", "15119", "200", "2461", "19", "2258", "31", "1250", "1216", "1595", "271", "236", "187", "166", "123", "231272", "12342923", "16587352", "32887158", "42478456", "353843", "1884868", "148239", "54241537", "213811", "3614", "1003", "177127860", "54250", "1720310", "6415742", "12117", "1293", "5541389", "44936", "550", "43448", "664", "39426", "5003285", "73925", "4379155", "2270", "123125129", "119138", "11121314"], "outputs": ["8\\n", "7\\n", "6\\n", "5\\n", "4\\n", "3\\n", "1\\n", "2\\n", "0\\n", "6\\n", "4\\n", "8\\n", "5\\n", "4\\n", "8\\n", "8\\n", "6\\n", "5\\n", "8\\n", "8\\n", "8\\n", "7\\n", "8\\n", "5\\n", "6\\n", "4\\n", "5\\n", "8\\n", "5\\n", "8\\n", "6\\n", "7\\n", "8\\n", "3\\n", "3\\n", "8\\n", "6\\n", "4\\n", "7\\n", "6\\n", "5\\n", "8\\n", "5\\n", "8\\n", "7\\n", "4\\n", "6\\n", "4\\n", "5\\n", "5\\n", "8\\n", "6\\n", "4\\n", "2\\n", "3\\n", "3\\n", "7\\n", "7\\n", "6\\n", "2\\n", "5\\n", "8\\n", "6\\n", "2\\n", "5\\n", "4\\n", "8\\n", "6\\n", "4\\n", "7\\n", "5\\n", "2\\n", "6\\n", "8\\n", "7\\n", "7\\n", "6\\n", "5\\n", "7\\n", "8\\n", "6\\n", "7\\n", "5\\n", "3\\n", "8\\n", "5\\n", "7\\n", "6\\n", "5\\n", "8\\n", "8\\n", "6\\n", "5\\n", "5\\n", "2\\n", "7\\n", "8\\n", "7\\n", "8\\n", "7\\n", "6", "5", "3"]}']}))

    print("--- Test 3: Taco test cases without fn_name one wrong test cases---")
    action = "```python\ndef sub(maxs, mins):\n\tfor i in range(len(maxs)):\n\t\tif maxs[i] != mins[i]:\n\t\t\tif i == len(maxs) - 1:\n\t\t\t\treturn int(maxs[i]) - int(mins[i])\n\t\t\tif i == len(maxs) - 2:\n\t\t\t\treturn int(maxs[i:i + 2]) - int(mins[i:i + 2])\n\t\t\treturn 10\n\treturn 0\n\ndef checkEqual(S):\n\tans = 8\n\tfor k in range(1, len(S)):\n\t\tif len(S) % k != 0:\n\t\t\tcontinue\n\t\tmins = maxs = S[0:k]\n\t\tfor s in range(0, len(S), k):\n\t\t\tmaxs = max(maxs, S[s:s + k])\n\t\t\tmins = min(mins, S[s:s + k])\n\t\tans = min(ans, sub(maxs, mins))\n\treturn ans\n\ndef check12(S):\n\tmaxv = 0\n\tminv = 10\n\tp = 0\n\twhile p < len(S):\n\t\tv = int(S[p])\n\t\tif S[p] == '1' and p + 1 < len(S):\n\t\t\tv = 10 + int(S[p + 1])\n\t\t\tp += 1\n\t\tmaxv = max(maxv, v)\n\t\tminv = min(minv, v)\n\t\tp += 1\n\treturn maxv - minv\nS = input()\nprint(min(checkEqual(S), check12(S)))\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["9714431", "16612328", "23422731", "754526", "955577", "75547", "2112", "799", "88", "32523857", "4787", "1859551", "135661", "3675", "156692", "167918384", "83994", "4837847", "14513597", "15282598", "12659326", "1468417", "6280", "115464", "52376853", "2315", "3641224", "97187", "836", "195884", "36250", "2427817", "17598762", "5744554", "9295", "129848", "3863342", "3743", "133862", "1237", "1625", "1179729", "12651", "3776912", "4829", "73", "2228", "2546", "3136", "138", "3380", "4828", "3652", "5667", "7275", "774", "9329", "279", "15119", "200", "2461", "19", "2258", "31", "1250", "1216", "1595", "271", "236", "187", "166", "123", "231272", "12342923", "16587352", "32887158", "42478456", "353843", "1884868", "148239", "54241537", "213811", "3614", "1003", "177127860", "54250", "1720310", "6415742", "12117", "1293", "5541389", "44936", "550", "43448", "664", "39426", "5003285", "73925", "4379155", "2270", "123125129", "119138", "11121314"], "outputs": ["7\\n", "7\\n", "6\\n", "5\\n", "4\\n", "3\\n", "1\\n", "2\\n", "0\\n", "6\\n", "4\\n", "8\\n", "5\\n", "4\\n", "8\\n", "8\\n", "6\\n", "5\\n", "8\\n", "8\\n", "8\\n", "7\\n", "8\\n", "5\\n", "6\\n", "4\\n", "5\\n", "8\\n", "5\\n", "8\\n", "6\\n", "7\\n", "8\\n", "3\\n", "3\\n", "8\\n", "6\\n", "4\\n", "7\\n", "6\\n", "5\\n", "8\\n", "5\\n", "8\\n", "7\\n", "4\\n", "6\\n", "4\\n", "5\\n", "5\\n", "8\\n", "6\\n", "4\\n", "2\\n", "3\\n", "3\\n", "7\\n", "7\\n", "6\\n", "2\\n", "5\\n", "8\\n", "6\\n", "2\\n", "5\\n", "4\\n", "8\\n", "6\\n", "4\\n", "7\\n", "5\\n", "2\\n", "6\\n", "8\\n", "7\\n", "7\\n", "6\\n", "5\\n", "7\\n", "8\\n", "6\\n", "7\\n", "5\\n", "3\\n", "8\\n", "5\\n", "7\\n", "6\\n", "5\\n", "8\\n", "8\\n", "6\\n", "5\\n", "5\\n", "2\\n", "7\\n", "8\\n", "7\\n", "8\\n", "7\\n", "6", "5", "3"]}']}))

    
    
def test_modules():
    """Test importing essential scientific and data analysis modules"""
    print("\n[4] Essential Modules Test")
    
    tool = FirejailPythonCodeWithTestTool()
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
    
    tool = FirejailPythonCodeWithTestTool()
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
    
    tool = FirejailPythonCodeWithTestTool()
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
    
    tool = FirejailPythonCodeWithTestTool()
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
        python -m verl_tool.servers.tests.test_firejail_python_code_with_verl_code_test  firejail --url=http://localhost:5001/get_observation
        python -m verl_tool.servers.tests.test_firejail_python_code_with_verl_code_test  direct
        python -m verl_tool.servers.tests.test_firejail_python_code_with_verl_code_test  matplotlib
        python -m verl_tool.servers.tests.test_firejail_python_code_with_verl_code_test  timeout
        python -m verl_tool.servers.tests.test_firejail_python_code_with_verl_code_test  modules
        python -m verl_tool.servers.tests.test_firejail_python_code_with_verl_code_test  multiprocess
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