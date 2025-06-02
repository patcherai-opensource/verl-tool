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
    
    print("--- Test 1: Taco test cases ---") # should pass
    action = "```python\nimport math\n\ndef race(v1, v2, g):\n\tif v2 < v1:\n\t\treturn None\n\tseconds = 0.1\n\twhile v1 / 3600 * seconds + g >= v2 / 3600 * seconds:\n\t\tseconds += 0.05\n\thours = seconds / 3600\n\thoursRest = seconds % 3600\n\tminutes = hoursRest / 60\n\tseconds = hoursRest % 60\n\treturn [math.floor(hours), math.floor(minutes), math.floor(seconds)]\n\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"fn_name": "race", "inputs": [[720, 850, 70], [80, 91, 37], [80, 100, 40], [720, 850, 37], [720, 850, 370], [120, 850, 37], [820, 850, 550], [820, 81, 550]], "outputs": [[[0, 32, 18]], [[3, 21, 49]], [[2, 0, 0]], [[0, 17, 4]], [[2, 50, 46]], [[0, 3, 2]], [[18, 20, 0]], [null]]}']}))

    print("--- Test 2: Taco test cases without fn_name---") # should pass
    action = "```python\ndef sub(maxs, mins):\n\tfor i in range(len(maxs)):\n\t\tif maxs[i] != mins[i]:\n\t\t\tif i == len(maxs) - 1:\n\t\t\t\treturn int(maxs[i]) - int(mins[i])\n\t\t\tif i == len(maxs) - 2:\n\t\t\t\treturn int(maxs[i:i + 2]) - int(mins[i:i + 2])\n\t\t\treturn 10\n\treturn 0\n\ndef checkEqual(S):\n\tans = 8\n\tfor k in range(1, len(S)):\n\t\tif len(S) % k != 0:\n\t\t\tcontinue\n\t\tmins = maxs = S[0:k]\n\t\tfor s in range(0, len(S), k):\n\t\t\tmaxs = max(maxs, S[s:s + k])\n\t\t\tmins = min(mins, S[s:s + k])\n\t\tans = min(ans, sub(maxs, mins))\n\treturn ans\n\ndef check12(S):\n\tmaxv = 0\n\tminv = 10\n\tp = 0\n\twhile p < len(S):\n\t\tv = int(S[p])\n\t\tif S[p] == '1' and p + 1 < len(S):\n\t\t\tv = 10 + int(S[p + 1])\n\t\t\tp += 1\n\t\tmaxv = max(maxv, v)\n\t\tminv = min(minv, v)\n\t\tp += 1\n\treturn maxv - minv\nS = input()\nprint(min(checkEqual(S), check12(S)))\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["9714431", "16612328", "23422731", "754526", "955577", "75547", "2112", "799", "88", "32523857", "4787", "1859551", "135661", "3675", "156692", "167918384", "83994", "4837847", "14513597", "15282598", "12659326", "1468417", "6280", "115464", "52376853", "2315", "3641224", "97187", "836", "195884", "36250", "2427817", "17598762", "5744554", "9295", "129848", "3863342", "3743", "133862", "1237", "1625", "1179729", "12651", "3776912", "4829", "73", "2228", "2546", "3136", "138", "3380", "4828", "3652", "5667", "7275", "774", "9329", "279", "15119", "200", "2461", "19", "2258", "31", "1250", "1216", "1595", "271", "236", "187", "166", "123", "231272", "12342923", "16587352", "32887158", "42478456", "353843", "1884868", "148239", "54241537", "213811", "3614", "1003", "177127860", "54250", "1720310", "6415742", "12117", "1293", "5541389", "44936", "550", "43448", "664", "39426", "5003285", "73925", "4379155", "2270", "123125129", "119138", "11121314"], "outputs": ["8\\n", "7\\n", "6\\n", "5\\n", "4\\n", "3\\n", "1\\n", "2\\n", "0\\n", "6\\n", "4\\n", "8\\n", "5\\n", "4\\n", "8\\n", "8\\n", "6\\n", "5\\n", "8\\n", "8\\n", "8\\n", "7\\n", "8\\n", "5\\n", "6\\n", "4\\n", "5\\n", "8\\n", "5\\n", "8\\n", "6\\n", "7\\n", "8\\n", "3\\n", "3\\n", "8\\n", "6\\n", "4\\n", "7\\n", "6\\n", "5\\n", "8\\n", "5\\n", "8\\n", "7\\n", "4\\n", "6\\n", "4\\n", "5\\n", "5\\n", "8\\n", "6\\n", "4\\n", "2\\n", "3\\n", "3\\n", "7\\n", "7\\n", "6\\n", "2\\n", "5\\n", "8\\n", "6\\n", "2\\n", "5\\n", "4\\n", "8\\n", "6\\n", "4\\n", "7\\n", "5\\n", "2\\n", "6\\n", "8\\n", "7\\n", "7\\n", "6\\n", "5\\n", "7\\n", "8\\n", "6\\n", "7\\n", "5\\n", "3\\n", "8\\n", "5\\n", "7\\n", "6\\n", "5\\n", "8\\n", "8\\n", "6\\n", "5\\n", "5\\n", "2\\n", "7\\n", "8\\n", "7\\n", "8\\n", "7\\n", "6", "5", "3"]}']}))

    print("--- Test 3: Taco test cases without fn_name one wrong test cases---") # should not pass, I changed the first outputs from 8 to 7 in the expected return
    action = "```python\ndef sub(maxs, mins):\n\tfor i in range(len(maxs)):\n\t\tif maxs[i] != mins[i]:\n\t\t\tif i == len(maxs) - 1:\n\t\t\t\treturn int(maxs[i]) - int(mins[i])\n\t\t\tif i == len(maxs) - 2:\n\t\t\t\treturn int(maxs[i:i + 2]) - int(mins[i:i + 2])\n\t\t\treturn 10\n\treturn 0\n\ndef checkEqual(S):\n\tans = 8\n\tfor k in range(1, len(S)):\n\t\tif len(S) % k != 0:\n\t\t\tcontinue\n\t\tmins = maxs = S[0:k]\n\t\tfor s in range(0, len(S), k):\n\t\t\tmaxs = max(maxs, S[s:s + k])\n\t\t\tmins = min(mins, S[s:s + k])\n\t\tans = min(ans, sub(maxs, mins))\n\treturn ans\n\ndef check12(S):\n\tmaxv = 0\n\tminv = 10\n\tp = 0\n\twhile p < len(S):\n\t\tv = int(S[p])\n\t\tif S[p] == '1' and p + 1 < len(S):\n\t\t\tv = 10 + int(S[p + 1])\n\t\t\tp += 1\n\t\tmaxv = max(maxv, v)\n\t\tminv = min(minv, v)\n\t\tp += 1\n\treturn maxv - minv\nS = input()\nprint(min(checkEqual(S), check12(S)))\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["9714431", "16612328", "23422731", "754526", "955577", "75547", "2112", "799", "88", "32523857", "4787", "1859551", "135661", "3675", "156692", "167918384", "83994", "4837847", "14513597", "15282598", "12659326", "1468417", "6280", "115464", "52376853", "2315", "3641224", "97187", "836", "195884", "36250", "2427817", "17598762", "5744554", "9295", "129848", "3863342", "3743", "133862", "1237", "1625", "1179729", "12651", "3776912", "4829", "73", "2228", "2546", "3136", "138", "3380", "4828", "3652", "5667", "7275", "774", "9329", "279", "15119", "200", "2461", "19", "2258", "31", "1250", "1216", "1595", "271", "236", "187", "166", "123", "231272", "12342923", "16587352", "32887158", "42478456", "353843", "1884868", "148239", "54241537", "213811", "3614", "1003", "177127860", "54250", "1720310", "6415742", "12117", "1293", "5541389", "44936", "550", "43448", "664", "39426", "5003285", "73925", "4379155", "2270", "123125129", "119138", "11121314"], "outputs": ["7\\n", "7\\n", "6\\n", "5\\n", "4\\n", "3\\n", "1\\n", "2\\n", "0\\n", "6\\n", "4\\n", "8\\n", "5\\n", "4\\n", "8\\n", "8\\n", "6\\n", "5\\n", "8\\n", "8\\n", "8\\n", "7\\n", "8\\n", "5\\n", "6\\n", "4\\n", "5\\n", "8\\n", "5\\n", "8\\n", "6\\n", "7\\n", "8\\n", "3\\n", "3\\n", "8\\n", "6\\n", "4\\n", "7\\n", "6\\n", "5\\n", "8\\n", "5\\n", "8\\n", "7\\n", "4\\n", "6\\n", "4\\n", "5\\n", "5\\n", "8\\n", "6\\n", "4\\n", "2\\n", "3\\n", "3\\n", "7\\n", "7\\n", "6\\n", "2\\n", "5\\n", "8\\n", "6\\n", "2\\n", "5\\n", "4\\n", "8\\n", "6\\n", "4\\n", "7\\n", "5\\n", "2\\n", "6\\n", "8\\n", "7\\n", "7\\n", "6\\n", "5\\n", "7\\n", "8\\n", "6\\n", "7\\n", "5\\n", "3\\n", "8\\n", "5\\n", "7\\n", "6\\n", "5\\n", "8\\n", "8\\n", "6\\n", "5\\n", "5\\n", "2\\n", "7\\n", "8\\n", "7\\n", "8\\n", "7\\n", "6", "5", "3"]}']}))

    print("--- Test 4: Taco test cases without fn_name one wrong test cases---") # should pass
    action = "```python\nt = int(input())\nfor z in range(t):\n\tn = int(input())\n\tarr = list(map(int, input().split()))\n\tif len(set(arr)) == 1:\n\t\tprint('NO ')\n\telse:\n\t\tprint('YES ')\n\t\trep = []\n\t\tfor i in range(1, n):\n\t\t\tif arr[0] == arr[i]:\n\t\t\t\trep.append(i)\n\t\t\telse:\n\t\t\t\tprint('1', i + 1)\n\t\t\t\tk = i\n\t\tfor num in rep:\n\t\t\tprint(k + 1, num + 1)\n\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["4\\n5\\n1 2 2 1 3\\n3\\n1 1 1\\n4\\n1 1000 101 1000\\n4\\n1 2 3 4\\n", "1\\n5\\n6756657 32231 86 234 23442\\n", "1\\n2\\n7 7\\n"], "outputs": ["YES\\n1 2\\n1 3\\n1 5\\n5 4\\nNO\\nYES\\n1 2\\n1 3\\n1 4\\nYES\\n1 2\\n1 3\\n1 4\\n", "YES\\n1 2\\n1 3\\n1 4\\n1 5\\n", "NO\\n", "NO\\n", "YES\\n1 2\\n1 3\\n1 4\\n1 5\\n"]}']}))

    print("--- Test 4: Taco test cases without fn_name one wrong test cases---") # should pass
    action = "```python\nn = int(input())\na = list(map(int, input().split()))\ncnt = {}\nfor i in range(n):\n\tmn = a[i]\n\tfor j in range(i, n):\n\t\tmn = min(mn, a[j])\n\t\tif mn in cnt:\n\t\t\tcnt[mn] += 1\n\t\telse:\n\t\t\tcnt[mn] = 1\nq = int(input())\nfor i in range(q):\n\tk = int(input())\n\tif k in cnt:\n\t\tprint(cnt[k])\n\telse:\n\t\tprint(0)\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": [["5", "4 1 2 3 4", "4", "3", "4", "6", "1", "", ""], "5\\n4 0 2 3 4\\n4\\n3\\n4\\n6\\n1", "5\\n4 0 2 3 4\\n4\\n5\\n4\\n6\\n1"], "outputs": [["2", "2", "0", "8"], "2\\n2\\n0\\n0\\n", "0\\n2\\n0\\n0\\n"]}']}))

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