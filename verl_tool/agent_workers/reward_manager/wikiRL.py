import nltk
import json
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import os
import time
import asyncio
import regex as re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from mini_webarena.rl_utils import format_score
from mini_webarena.evaluator import metric_heuristic
# ------------------------------------------------------------------------------
# WikiQA Reward Manager
# ------------------------------------------------------------------------------
class WikiQARewardManager:
    """
    Reward Manager for the wikiQA dataset.

    This class computes a combined reward for each predicted answer by comparing it with
    the ground truth answers. The final reward is a weighted combination of a fuzzy matching
    score and a structure score.
    # """
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None) -> None:
        """
        Initialize the WikiQARewardManager.

        Parameters:
        - fuzzy_weight: The weight applied to the fuzzy matching score.
        - structure_weight: The weight applied to the structure score.
        """
        if tokenizer is None:
            # Simply use QWen2.5-7B tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.fuzzy_weight = 0.7
        self.structure_weight = 0.3

    def answer_score(self, pred, ground_truths):
        def extract_last_stop_content(input_str: str) -> str:
            matches = re.findall(r"```stop\s*\[([^\]]*)\]```", input_str)
            if matches:
                return matches[-1]
            return ""
        # First match ```stop [...]``` use regex to find the last ```stop [...]``` in the string
        pred = extract_last_stop_content(pred)
        score = metric_heuristic(ground_truths, pred)
        # print("answer score", ground_truths, pred, score)
        return score

    def format_score(self, actions):
        # Average of format_score
        scores = [format_score(action) for action in actions]
        return sum(scores) / len(scores) if scores else 0

    def __call__(self, data:DataProto):
        """
        Compute rewards for a batch of data samples.

        Expected input 'data' structure:
          - data.batch['responses']: A list of predicted answer strings.
          - For each sample i, data[i].non_tensor_batch['reward_model']['ground_truth'] should
            provide the list of ground truth answer strings.

        Parameters:
        - data: A batch object containing multiple data samples.

        Returns:
        - reward_tensor: A torch.Tensor containing the computed rewards for each sample.
        """
        # Retrieve the list of predicted responses.
        # print("")
        # print(data)
        # import pickle
        # with open("data_stub.pkl", "wb") as f:
        #     pickle.dump(data, f)

        special_token_ids = set(self.tokenizer.all_special_ids)

        actions_list = []
        observations_list = []
        response_list = []
        for i in range(len(data)):
            actions = []
            observations = []
            input_ids = data.batch["input_ids"][i].tolist()
            attention_mask = data.batch["attention_mask"][i].tolist()

            action_lengths_list = data.non_tensor_batch["action_lengths"][i]
            obs_lengths_list = data.non_tensor_batch["obs_lengths"][i]

            # 获取 response 部分的 token ids
            response_ids = input_ids[2048:] # Depends on Prompt Length, TODO
            response_mask = attention_mask[2048:]
            response_tokens = [
                tid for tid, mask in zip(response_ids, response_mask)
                if mask == 1 and tid not in special_token_ids
            ]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            response_list.append(response_text)

            # print(f"Sample {i}:")
            # print(f"  Prompt tokens (no special):   {len(input_ids[:2048])}")
            # print(f"  Response tokens (no special): {len(response_tokens)}")
            # print(f"  Total tokens (no special):    {len(input_ids[:2048]) + len(response_tokens)}")

            # 切分并解码 response_tokens
            cursor = 0
            for idx, (action_len, obs_len) in enumerate(zip(action_lengths_list, obs_lengths_list)):
                action_tokens = response_tokens[cursor:cursor + action_len - 1 ]
                cursor += action_len - 1
                obs_tokens = response_tokens[cursor:cursor + obs_len - 1]
                cursor += obs_len - 1

                action_text = self.tokenizer.decode(action_tokens, skip_special_tokens=True).strip()
                actions.append(action_text)
                obs_text = self.tokenizer.decode(obs_tokens, skip_special_tokens=True).strip()
                observations.append(obs_text)
                # print(f"[Action {idx + 1}]\n{action_text}")
                # print(f"[Obs {idx + 1}]\n{obs_text}")
                # print("&&&&&")

            if cursor < len(response_tokens):
                remaining_tokens = response_tokens[cursor:]
                remaining_text = self.tokenizer.decode(remaining_tokens, skip_special_tokens=True).strip()
                actions.append(remaining_text)
                # print("[Remaining tokens after cuts]")
                # print(remaining_text)
                # print()
            actions_list.append(actions)
            observations_list.append(observations)

        # TODO: Add save to disk function,
        # actions.jsonl: {traj_id, actions},
        # traj.jsonl: {traj_id, actions, observations}

        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]  # prompt 的长度
        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)  # shape: [batch_size]
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)

        answer_scores = []
        format_scores = []

        for i in range(len(data)):
            ground_truths = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            # 假设 response_list[i] 是当前生成的文本序列 (str)，actions_list[i] 是解码后的 token list / action list
            pred = response_list[i]
            answer_reward = self.answer_score(pred, ground_truths)
            format_reward = self.format_score(actions_list[i])

            # 最终的 scalar reward，比如 answer_reward + 0.5 * format_reward
            final_reward = answer_reward + 0.5 * format_reward

            # 只在该样本最后一个有效 token 的位置赋值
            # valid_response_length[i] - 1 因为下标从 0 开始
            reward_tensor[i, valid_response_length[i].item() - 1] = final_reward

            # 存一下分数用于查看
            answer_scores.append(answer_reward)
            format_scores.append(format_reward)

        print(f"Computed rewards for {len(data)} samples.")
        print("Answer scores:", answer_scores)
        print("Format scores:", format_scores)

        # exit(1)
        # reward_tensor = reward_tensor.mean(dim=-1)
        return reward_tensor

if __name__ == '__main__':
    import pickle

    # Load the saved data object from disk
    with open("data_stub.pkl", "rb") as f:
        dummy_data = pickle.load(f)

    # Instantiate the WikiQARewardManager (you can pass in config if needed)
    reward_manager = WikiQARewardManager()

    # Compute rewards for the loaded data
    rewards = reward_manager(dummy_data)
    print("Rewards:", rewards)


"""
(TaskRunner pid=2019847) ==== Call WikiQARewardManager ====
(TaskRunner pid=2019847) DataProto(batch=TensorDict(
(TaskRunner pid=2019847)     fields={
(TaskRunner pid=2019847)         attention_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         info_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         input_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         old_log_probs: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         position_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         prompts: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         ref_log_prob: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         responses: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         responses_with_info_mask: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False)},
(TaskRunner pid=2019847)     batch_size=torch.Size([4]),
(TaskRunner pid=2019847)     device=None,
(TaskRunner pid=2019847)     is_shared=False), non_tensor_batch={'data_source': array(['wiki_qa', 'wiki_qa', 'wiki_qa', 'wiki_qa'], dtype=object), 'ability': array(['wiki', 'wiki', 'wiki', 'wiki'], dtype=object), 'reward_model': array([{'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'}],
(TaskRunner pid=2019847)       dtype=object), 'index': array([0, 0, 0, 0], dtype=object), 'uid': array(['ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b'], dtype=object)}, meta_info={'turns_stats': [4, 4], 'active_mask': [True, True], 'valid_action_stats': [4, 4], 'global_token_num': [5541, 5541, 3697, 5542], 'temperature': 0.9})
"""
