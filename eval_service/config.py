import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model: str
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    num_models: int = 1
@dataclass
class ToolConfig:
    tool_server_url: str = "http://localhost:30150/get_observation"
    max_turns: int = 5  # max generation turns
    truncate_obs_side: str = "left"  # "left" or "right", which side to truncate when the observation is too long
    action_stop_tokens: str = None
    max_obs_length: int = None  # maximum length of observation
    
    def post_init(self):
        """
        Post-initialization processing for ToolConfig (will not call automatically)
        """
        # action_stop_tokens can be a string or a file path
        if isinstance(self.action_stop_tokens, str):
            if os.path.exists(self.action_stop_tokens):
                with open(self.action_stop_tokens, "r") as f:
                    self.action_stop_tokens = f.read().split(',')
            else:
                self.action_stop_tokens = self.action_stop_tokens.split(',')
            self.action_stop_tokens = [token.strip('\n ') for token in self.action_stop_tokens]
            self.action_stop_tokens = [token for token in self.action_stop_tokens if token]
        else:
            self.action_stop_tokens = None
        print(f"using action_stop_tokens: {self.action_stop_tokens}")

@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 8000