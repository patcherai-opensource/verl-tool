import nltk
import json
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher

def clean_text(text: str) -> str:
    text = text.strip().lower()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]
    return text

def char_lcs_ratio(ref: str, pred: str) -> float:
    matcher = SequenceMatcher(None, ref, pred)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    max_len = max(len(ref), len(pred)) or 1
    return lcs_len / max_len

def token_f1(ref: str, pred: str) -> float:
    ref_tokens = set(word_tokenize(ref))
    pred_tokens = set(word_tokenize(pred))

    if not ref_tokens or not pred_tokens:
        return 0.0

    intersection = ref_tokens & pred_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(ref_tokens)

    if precision == 0 and recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def edit_distance_ratio(ref: str, pred: str) -> float:
    dp = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(pred) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    edit_dist = dp[len(ref)][len(pred)]
    max_len = max(len(ref), len(pred)) or 1
    return edit_dist / max_len

def fuzzy_match(ref: str, pred: str,
                alpha: float = 0.7,
                beta: float = 0.3,
                gamma: float = 0.1) -> float:
    print("Now is in fuzzy_match")
    print(f"ref: {ref}, pred: {pred}")

    ref = clean_text(ref)
    pred = clean_text(pred)

    char_lcs = char_lcs_ratio(ref, pred)
    tok_f1 = token_f1(ref, pred)
    dist_penalty = edit_distance_ratio(ref, pred)

    score = alpha * char_lcs + beta * tok_f1 - gamma * dist_penalty
    print("score: ", score)
    return max(0.0, min(score, 1.0))

def format_score(s: str, is_success: bool = True) -> float:
    score = 0.0

    if "<think>" not in s or "</think>" not in s:
        return 0.0
    score += 0.3

    idx_think = s.index("<think>")
    prefix = s[:idx_think].strip()
    if prefix == "":
        score += 0.1

    tail_part = s.split("</think>", maxsplit=1)
    if len(tail_part) < 2:
        if is_success:
            score += 0.2
        return round(score, 3)

    tail_content = tail_part[1].strip()

    pattern = r"```((.|\n)*?)```"
    match = re.search(pattern, tail_content)
    if match:
        score += 0.2
        action_text = match.group(1).strip()

        expected_block = f"```{action_text}```"
        if tail_content == expected_block:
            score += 0.2

    if is_success:
        score += 0.2

    return round(score, 3)

# ------------------------------------------------------------------------------
# WikiQA Reward Manager
# ------------------------------------------------------------------------------
class WikiQARewardManager:
    """
    Reward Manager for the wikiQA dataset.

    This class computes a combined reward for each predicted answer by comparing it with
    the ground truth answers. The final reward is a weighted combination of a fuzzy matching
    score and a structure score.
    """
    def __init__(self, fuzzy_weight: float = 0.7, structure_weight: float = 0.3):
        """
        Initialize the WikiQARewardManager.

        Parameters:
        - fuzzy_weight: The weight applied to the fuzzy matching score.
        - structure_weight: The weight applied to the structure score.
        """
        self.fuzzy_weight = fuzzy_weight
        self.structure_weight = structure_weight

    def compute_reward(self, pred: str, ground_truths: list) -> float:
        """
        Compute the reward for a single prediction.

        This method calculates the fuzzy match scores for all provided ground truth answers,
        selects the maximum fuzzy score, computes the structure score for the predicted answer,
        and then combines them using the defined weights.

        Parameters:
        - pred: The predicted answer string.
        - ground_truths: A list of ground truth answer strings.

        Returns:
        - A float representing the final computed reward.
        """
        # Calculate fuzzy match scores for each ground truth answer.
        fuzzy_scores = [fuzzy_match(gt, pred) for gt in ground_truths]
        # Select the best (maximum) fuzzy score.
        max_fuzzy = max(fuzzy_scores) if fuzzy_scores else 0.0
        # Calculate the structure score for the predicted answer.
        struct_score = format_score(pred)
        # Combine the two scores using the defined weights.
        final_score = self.fuzzy_weight * max_fuzzy + self.structure_weight * struct_score
        return final_score

    def __call__(self, data) -> torch.Tensor:
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
        print(data)
        exit(1)
        predictions = data.batch['responses']
        rewards = []
        # Iterate over each sample in the batch.
        for i in range(len(data)):
            # Get the predicted answer for the current sample.
            pred = predictions[i]
            # Get the list of ground truth answers for the current sample.
            ground_truths = data[i].non_tensor_batch['reward_model']['ground_truth']
            # Compute the combined reward for this sample.
            reward = self.compute_reward(pred, ground_truths)
            rewards.append(reward)
        # Convert the list of rewards into a PyTorch tensor.
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        print(f"Computed rewards for {len(rewards)} samples.")
        return reward_tensor

# ------------------------------------------------------------------------------
# Dummy Data Example for Testing
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Define a dummy data sample class to mimic the expected data structure.
    class DummyDataSample:
        def __init__(self, response, ground_truth):
            # non_tensor_batch holds metadata and ground truth information.
            self.non_tensor_batch = {'reward_model': {'ground_truth': ground_truth}}
            self.response = response

    # Define a dummy data batch class that holds multiple dummy samples.
    class DummyDataBatch:
        def __init__(self, responses, ground_truths):
            # 'batch' simulates a tensor batch containing the predicted responses.
            self.batch = {'responses': responses}
            # Create a list of DummyDataSample objects for each response and corresponding ground truth.
            self.data_samples = [DummyDataSample(resp, gt) for resp, gt in zip(responses, ground_truths)]

        def __len__(self):
            return len(self.data_samples)

        def __getitem__(self, index):
            return self.data_samples[index]

    # Create dummy responses and corresponding ground truths for testing.
    dummy_responses = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius"
    ]
    dummy_ground_truths = [
        ["Paris is the capital of France", "Paris"],
        ["100°C is the boiling point of water", "Water boils at 100°C."]
    ]
    dummy_data = DummyDataBatch(dummy_responses, dummy_ground_truths)

    # Instantiate the WikiQARewardManager with default weights.
    reward_manager = WikiQARewardManager()
    # Compute rewards for the dummy data batch.
    rewards = reward_manager(dummy_data)
    print("Rewards:", rewards)


