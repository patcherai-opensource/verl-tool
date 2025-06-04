# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import os
import re
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
import fastapi
import numpy as np
import ray
import uvicorn
from datasets import load_dataset
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from starlette.responses import JSONResponse
from verl.utils import hf_tokenizer, hf_processor

from verl_tool.agent_workers.tool_chat_completion_scheduler import NaiveChatCompletionScheduler
from verl_tool.tests.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.trainer.main_ppo import create_rl_dataset
from verl_tool.llm_agent.config import AgentActorConfig

boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code.

    WARNING: This class is for testing purpose only, do not use it in production.
    Please use a sandbox with strong isolation and security restrictions instead.
    """

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]
        print(f"execute code:\n{code}")

        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            stdout, stderr = await process.communicate()

            return JSONResponse(content={"stdout": stdout.decode(), "stderr": stderr.decode(), "returncode": process.returncode})
        finally:
            try:
                os.unlink(temp_file)
            except:  # noqa: E722
                pass

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print("FastAPI startup")
            self.server_ready.set()
            yield

            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/code/execution", self.code_execution, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"


class ToolChatCompletionScheduler(NaiveChatCompletionScheduler):
    """This is a chat completion scheduler that supports sandbox code execution
    """

    def __init__(self, config, model_path, server_addresses, sandbox_address, agent_config, **kwargs):
        super().__init__(config, model_path, server_addresses, **kwargs)
        self.sandbox_address = sandbox_address
        self.agent_config = agent_config
        print(f"agent_config: {self.agent_config}")
    
    async def parse_code_block(self, action: str) -> str:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            parsed_code
        """
        # Try to find Python code in various formats
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
        
        if len(all_valid_python_code) == 0:
            return ""
        
        # use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
        return parsed_code
    
    async def sandbox_code_execution(self, code: str) -> Dict[str, Any]:
        """Execute python code in sandbox."""
        try:
            session = aiohttp.ClientSession()
            async with session.post(
                url=f"http://{self.sandbox_address}/code/execution",
                json={"code": code},
            ) as resp:
                return await resp.json()
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.agent_config.max_action_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            include_stop_str_in_output=True, # TODO: config
            stop=self.agent_config.action_stop_tokens, 
            # extra_body={
            # },
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[ToolChatCompletionScheduler] generate_sequences sampling params: {kwargs}")


        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            new_kwargs = {k: v for k, v in kwargs.items()}

            batch_conversations, batch_index, turn = (
                info["batch_conversations"],
                info["batch_index"],
                info["turn"],
            )
            print(f"[id={completions.id},turn={turn}] {completions}")
            role, content = completions.choices[0].message.role, completions.choices[0].message.content
            batch_conversations[batch_index].append({"role": role, "content": content})


            # STEP 0: if we reach max turns + 1 break and if we reach max turns, remove stop token
            if turn == self.agent_config.max_turns + 1:
                return
            elif turn == self.agent_config.max_turns:
                # if we reach max turns + 1, remove code block stop token
                print(f"[id={completions.id},turn={turn}] new_kwargs 1: {new_kwargs}")
                new_kwargs.pop("stop")
                print(f"[id={completions.id},turn={turn}] new_kwargs 2: {new_kwargs}")

            # STEP 1: check if we got answer
            matches = boxed_pattern.findall(content)
            if matches:
                print(f"[id={completions.id},turn={turn}] Got answer: {matches[0]}, done!")
                return

            # STEP 2: check if stop reason is 
            finish_reason = completions.choices[0].finish_reason
            stop_reason = completions.choices[0].stop_reason

            # STEP 3: execute code block in sandbox 
            if finish_reason == "stop" and stop_reason is not None: # TODO: check
                code = await self.parse_code_block(content) 
                result = await self.sandbox_code_execution(code)
                stdout, stderr = result["stdout"], result["stderr"]
                batch_conversations[batch_index].append({"role": "tool", "content": f"{stdout}{stderr}"}) # TODO: check whether works for qwen 

            # STEP 4: resubmit chat completions with code block output 
            extra_headers = {"x-request-id": completions.id}
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "batch_conversations": batch_conversations,
                    "batch_index": batch_index,
                    "turn": turn + 1,
                },
                model=self.model_name,
                messages=batch_conversations[batch_index],
                extra_headers=extra_headers,
                **new_kwargs,
            )

        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]): # TODO: include `raw_prompt` of gen_batch in `ray_trainer` 
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = list(conversation)
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "turn": 1,
                        },
                        model=self.model_name,
                        messages=batch_conversations[batch_index],
                        **kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        print("[ToolChatCompletionScheduler] generate_sequences done")

        # _postprocess assumes n>=1
        batch_conversations = [[conversation] for conversation in batch_conversations]
        return self._postprocess(batch, batch_conversations, kwargs["n"])


# system_prompt = """
# You are a helpful assistant. Let's solve math problem in following steps:
# 1. Write a python code first and return the code to user, the code must be in following format:

# <code>
# ```python
# import os

# print(...)
# ```
# </code>

# The code must explictly print necessary output to stdout. Remember stop generation at </code> immediately and return the code.
# 2. User will send the python code to a external sandbox to execute and get output from stdout.
# 3. User will send the output in format <interpreter>output</interpreter> to you, and you should use the output to answer the question.
# The answer format must be: <answer>\\boxed{'The final answer goes here.'}</answer>
# """


def test_vllm_tool_calling():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # Load config # TODO: update in .sh script expecially for agent config 
    config = OmegaConf.load("verl_tool/trainer/config/ppo_trainer.yaml")
    config.actor_rollout_ref.model.path = "/map-vepfs/models/Qwen2.5-Math-1.5B"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.chat_scheduler = "verl_tool.tests.test_async_vllm_tool_calling.ToolChatCompletionScheduler"
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 1024
    config.trainer.n_gpus_per_node=2
    config.data.train_files = "./data/tests/aime24/train.parquet"
    config.data.return_raw_chat = True

    agent_config = AgentActorConfig()
    for key in getattr(config, 'agent', {}).keys():
        if key in agent_config.__dict__.keys():
            setattr(agent_config, key, config.agent[key])
    setattr(agent_config, 'n', config.actor_rollout_ref.rollout.n)

    agent_config.max_action_length = 512 # TODO: check whther vllm padding
    agent_config.max_turns = 1
    agent_config.action_stop_tokens = ["```output"] # TODO: magic string delete after testing # TODO: use ```without python
    # if agent_config.action_stop_tokens is not None:
    #     if os.path.exists(agent_config.action_stop_tokens):
    #         with open(agent_config.action_stop_tokens, 'r') as f:
    #             agent_config.action_stop_tokens = [x for x in f.read().split(',') if x]
    #         print(f"Using action stop tokens: {agent_config.action_stop_tokens}")
    #     else:
    #         raise ValueError(f"action_stop_tokens file not found: {agent_config.action_stop_tokens}")
    # else:
    #     agent_config.action_stop_tokens = []


    # Init sandbox and async rollout manager
    sandbox = Sandbox.options(num_cpus=1).remote()
    sandbox_address = ray.get(sandbox.get_server_address.remote())
    # TODO: update `ray_trainer` init 
    async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"sandbox_address": sandbox_address, "agent_config": agent_config})

    # Build dataset
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, None) 
    raw_prompt = train_dataset.dataframe['prompt']

    gen_batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompt),
        },
    )
    result = async_rollout_manager.generate_sequences(prompts=gen_batch)
    print(result)


if __name__ == "__main__":
    test_vllm_tool_calling()

# TODO: 1. find a way to silence vllm logging
"""
# preprocess aime24 data
python examples/data_preprocess/aime24.py --local_dir ./data/tests/aime24 

# run test
python verl_tool/tests/test_async_vllm_tool_calling.py
"""
