from .base import BaseTool, register_tool
import regex as re
from concurrent.futures import ThreadPoolExecutor



@register_tool
class FinishTool(BaseTool):
    tool_type = "finish"
    timeout = 10
    
    def get_usage_inst(self):
        return ""
    
    def parse_action(self, action:str):
        """
        Parse the raw action string (which is the llm response) into a actual action and it's contents
        """
        return "", False # this is the end of the trajectory, not a real action
    
    def conduct_action(self, trajectory_id, action, extra_data):
        action, is_valid = self.parse_action(action)
        observation = ""
        done = True
        return observation, done, is_valid
    