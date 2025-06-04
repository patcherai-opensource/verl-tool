import asyncio
import re
from typing import Any, Dict, List

import aiohttp
import torch
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler


class NaiveChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}"
            conversation, batch_conversations, batch_index = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_index"],
            )

            conversations = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # TODO: use `semaphore` to control max concurrent requests
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "conversation": list(conversation),
                        },
                        model=self.model_name,
                        messages=conversation.tolist(),
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, kwargs["n"])

    def _postprocess(self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
        # TODO: check here tool tokens
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]] 

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch)
    
class ToolChatCompletionScheduler(NaiveChatCompletionScheduler):
    """This is a demo chat completion scheduler that supports sandbox code execution
    described in ReTool paper: https://arxiv.org/pdf/2504.11536
    """

	# TODO: update agent config
    def __init__(self, config, model_path, server_addresses, sandbox_address, system_prompt, **kwargs):
        super().__init__(config, model_path, server_addresses, **kwargs)
        self.sandbox_address = sandbox_address
        self.system_prompt = system_prompt

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
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            extra_body={
                "include_stop_str_in_output": True,
                "stop": ["</answer>", "</code>"], # TODO: update 
            },
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[ToolChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        max_turns = 3 # TODO: update

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            batch_conversations, batch_index, turn = (
                info["batch_conversations"],
                info["batch_index"],
                info["turn"],
            )
            role, content = completions.choices[0].message.role, completions.choices[0].message.content
            batch_conversations[batch_index].append({"role": role, "content": content})

            # STEP 0: check if we reach max turns
            if turn == max_turns:
                print(f"[id={completions.id},turn={turn}] Reach max turns {max_turns}, done!")
                return

            # STEP 1: check if we got answer
            matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL) # TODO: update extract answer pattern
            if matches:
                print(f"[id={completions.id},turn={turn}] Got answer: {matches[0]}, done!")
                return

            # STEP 2: check if we got code block
            matches = re.findall(r"<code>\s*```python(.*?)```\s*</code>", content, re.DOTALL) # TODO: update extract code block pattern
            if not matches:
                print(f"[id={completions.id},turn={turn}] No code block found, done!")
                return

            # STEP 3: execute code block in sandbox # TODO: update call tool server
            code = matches[0].strip()
            result = await self.sandbox_code_execution(code)
            stdout, stderr = result["stdout"], result["stderr"]
            batch_conversations[batch_index].append({"role": "tool", "content": f"{stdout}{stderr}"})
            print(f"[id={completions.id},turn={turn}] Code block executed, continue...")

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
                **kwargs,
            )

        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation_id in enumerate(batch.non_tensor_batch["raw_prompt_ids"]): 
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            print(f'conversation_id: {conversation_id}')
            print(f'conversation: {conversation}')
            conversation = self.tokenizer.decode(conversation_id)
            batch_conversations[batch_index] = [{"role": "system", "content": self.system_prompt}] + list(conversation)
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
