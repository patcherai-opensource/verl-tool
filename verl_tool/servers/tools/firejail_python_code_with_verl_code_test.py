"""
add-apt-repository ppa:deki/firejail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install firejail firejail-profiles
"""
from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import signal
import sys
import json
import uuid
import hashlib
import textwrap
from typing import Tuple, Dict, Any, Optional, Union, List

import random


from .testing_util import *

# Timeout for code execution in seconds
TIMEOUT = 5
PRE_IMPORT_LIBS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"

def check_forbidden_imports(code: str) -> bool:
    """
    Checks if the code contains imports of potentially dangerous packages.
    
    Args:
        code: Python code string to analyze
        
    Returns:
        Boolean indicating if the code contains forbidden imports
    """
    # List of potentially dangerous modules that could affect the host system
    forbidden_modules = [
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    ]
    
    # Simple string-based check for import statements
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True
    
    # Check for os.system, os.popen, and similar dangerous calls
    dangerous_patterns = [
        "os.system", "os.popen", "os.spawn", "os.fork", 
        "os.exec", "sys.exit", "os._exit", "os.kill"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    
    return False

def wrap_code_blocks(code: Union[str, List[str]]) -> str:
    """
    Wraps the provided code blocks with try-except to handle exceptions including syntax errors.
    For previous codes, redirect stdout and stderr to null and export defined functions and variables.
    
    Args:
        code: List of code strings to wrap
        
    Returns:
        Wrapped code string
    """
    wrapped_code = ""
    
    # Convert single string to list for consistent handling
    if isinstance(code, str):
        code = [code]
    
    # Import needed at the top
    wrapped_code += "import sys, os, io, ast\n\n"
    
    # Add the safe_exec_with_exports function
    wrapped_code += """
def parse_and_exec_salvageable(code_string):
    # Split the code into lines
    lines = code_string.splitlines()
    
    # Try to execute code incrementally, line by line or in blocks
    current_block = ""
    local_namespace = {}
    
    for line in lines:
        # Add the current line to our accumulating block
        if current_block:
            current_block += "\\n" + line
        else:
            current_block = line
            
        # Skip empty lines or comments
        if not line.strip() or line.strip().startswith('#'):
            continue
            
        # Try to parse the current block to check for syntax
        try:
            ast.parse(current_block)
            
            # If it parses successfully, try to execute it
            try:
                # Create a new local namespace for this execution
                exec(current_block, globals(), local_namespace)
                
                # Clear the block after successful execution
                current_block = ""
            except Exception as e:
                print(f"Runtime error in block: {e}")
                current_block = ""  # Reset the block after a runtime error
                
        except SyntaxError:
            # If we have a syntax error in the accumulated block,
            # don't reset yet - we might need more lines to complete the syntax
            pass
    
    return local_namespace
"""
    
    for i, block in enumerate(code):
        is_last_block = i == len(code) - 1
        
        # For all blocks except the last, use safe_exec_with_exports
        if not is_last_block:
            wrapped_block = (
                f"\n# Code block {i+1} (previous)\n"
                f"original_stdout, original_stderr = sys.stdout, sys.stderr\n"
                f"sys.stdout, sys.stderr = io.StringIO(), io.StringIO()\n"
                f"try:\n"
                f"    exported_vars = parse_and_exec_salvageable('''{block}''')\n"
                f"finally:\n"
                f"    sys.stdout, sys.stderr = original_stdout, original_stderr\n\n"
                f"    for name, value in exported_vars.items():\n"
                f"        globals()[name] = value\n"
            )
        else:
            # For the last (current) block, just include the code directly
            wrapped_block = f"\n# Code block {i+1} (current)\n{block}\n"
        
        wrapped_code += wrapped_block
    
    return wrapped_code
    
def execute_python_in_firejail(code: Union[str, List[str]], timeout: int=TIMEOUT, stdin: Optional[str] = None, python_path: str = None, pre_import_lib: bool = False) -> Tuple[str, bool]:
    """
    Execute Python code in a Firejail sandbox with a timeout.
    
    Args:
        code: Python code string to execute
        stdin: Optional input to provide to the executed code
        
    Returns:
        String containing execution output or error message
    """
    # Check for forbidden imports first
    if check_forbidden_imports(code):
        return "", "Execution blocked: Code contains potentially dangerous operations or imports.", True
    
    # Create a minimal environment instead of copying everything
    original_env = os.environ.copy()
    env = {}
    
    # Core system variables
    essential_vars = [
        "PATH", "HOME", "USER", "SHELL", 
        "LANG", "LC_ALL", "LC_CTYPE", "TERM",
        # Python-specific
        "PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
        # Runtime optimization
        "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        # Temp directories
        "TMPDIR", "TEMP", "TMP",
        # Display if needed
        "DISPLAY", "XAUTHORITY"
    ]
    
    # Copy only essential variables if they exist
    for var in essential_vars:
        if var in original_env:
            env[var] = original_env[var]
    
    # Explicitly set optimization variables
    env["OPENBLAS_NUM_THREADS"] = "1"
    
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]
    
    # Build the firejail command with resource limits
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",  # Limit file size
        "--rlimit-as=1096m",
    ]
    # set cwd to be a temp dir
    # cwd = os.path.join(os.getcwd(), "/tmp/firejail")
    cwd = "/tmp/firejail"
    if not os.path.exists(cwd):
        os.makedirs(cwd, exist_ok=True)
    # write code to a temp file
    # file_name = f"code_{hashlib.md5(code.encode()).hexdigest()}.py"
    file_name = f"code_{uuid.uuid4().hex}.py"
    file_path = os.path.join(cwd, file_name)
    code = wrap_code_blocks(code)
    with open(file_path, "w") as f:
        f.write(code)
    if pre_import_lib:
        code = PRE_IMPORT_LIBS + code
    # command.extend(["python3", "-c", code])
    # command.extend(["python3", file_path])
    if not python_path:
        python_path = "python3"
    else:
        assert os.path.exists(python_path), f"Python path {python_path} does not exist."
    command.extend([python_path, file_path])
    has_error = False
    try:
        # Execute the command
        result = subprocess.run(
            command,
            input=stdin if stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        
        stdout = result.stdout
        stderr = result.stderr.strip()
        if stderr:
            has_error = True
    except subprocess.TimeoutExpired:
        has_error = True
        stdout = ""
        stderr = f"Execution timed out after {timeout} seconds.\n"
    # Clean up the temporary file
    try:
        os.remove(file_path)
    except Exception as e:
        pass
    return stdout, (stderr if stderr else ""), has_error

@register_tool
class FirejailPythonCodeWithVerlCodeTest(BaseTool):
    tool_type = "firejail_python_code_with_verl_code_test"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = False
    enable_mannual_reflection = False # deprecated
    force_run_test_cases = True
    done_without_error = True # passive
    python_path = None
    pre_import_lib = False
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code securely inside a Firejail sandbox."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        # Try to find Python code in various formats
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
        
        # if not all_valid_python_code:
        #     all_valid_python_code = re.findall(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)

        if len(all_valid_python_code) == 0:
            return "", False
        
        # # Use the first code block found (we could extend this to support multiple blocks)
        # parsed_code = all_valid_python_code[0].strip()
        
        # use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
        return parsed_code, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action in a Firejail sandbox.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            code_has_error = False
            observation = ""
            # redundant variable
            execution_result = ""
            test_cases = extra_field.get("public_tests", None) if extra_field else None
            
            test_cases = json.loads(test_cases) if isinstance(test_cases, str) else test_cases
            
            if self.force_run_test_cases and test_cases is not None:
                # run code test given the extracted test cases                
                res, error_msg = run_test(in_outs=test_cases, test=parsed_action, timeout=5)
                
                # case: captured an error for a test case
                if error_msg != {}:
                    
                    
                    # if error_msg is not empty, it means there is an error in the code
                    code_has_error = True
                    metadata = {
                        "error_code": error_msg["error_code"],
                        "traceback": error_msg.get("traceback"),
                        "inputs": error_msg.get("inputs", []),
                        "output": error_msg.get("output", ""),
                        "expected": error_msg.get("expected", ""),
                    }

                    # -1: syntax error / cannot extract code
                    # -2: wrong answer
                    # -3: time limit exceeded
                    # -4: runtime error
                    # "inputs": only have if -2
                    # "output": only have if -2
                    # "expected": only have if -2, 
                                        
                    if metadata["error_code"] == -1:
                        observation = f"The above code is incorrect and got a syntax error.\nError Msg:\n{metadata['traceback']}"
                    elif metadata["error_code"] == -2:
                        observation = f"The above code is incorrect and got a wrong answer.\nInput: {metadata['inputs']}\nGenerated Output: {metadata['output']}\nExpected: {metadata['expected']}"
                    elif metadata["error_code"] == -3:
                        observation = f"The above code is incorrect and got time limit exceeded.\nError Msg:{metadata['traceback']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}"
                    elif metadata["error_code"] == -4:
                        observation = f"The above code is incorrect and got a runtime error.\nError Msg:{metadata['traceback']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n"
                    
                # case: no error for all test cases
                else:
                    observation = "All test passed"
                
            if self.done_without_error:
                if code_has_error:
                    done = False
                else:
                    done = True
            else: 
                done = False
            valid = True
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
        