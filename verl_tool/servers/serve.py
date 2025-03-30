import argparse
import re
import fire
import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .tools import get_tool_cls, ALL_TOOLS
from typing import List, Union

logger = logging.getLogger(__name__)


def parse_agent_request(data):
    trajectory_ids = data.get("trajectory_ids")
    actions = data.get("actions")
    extra_fields_keys = [key for key in data if key not in ["trajectory_ids", "actions"]]
    extra_fields = [{key: data.get(key)[i] for key in extra_fields_keys} for i in range(len(actions))]
    assert len(trajectory_ids) == len(actions), f"Number of trajectory_ids ({len(trajectory_ids)}) does not match number of actions ({len(actions)})"
    if not extra_fields:
        extra_fields = [None for _ in range(len(actions))]
    
    parsed_agent_request = {
        "trajectory_ids": trajectory_ids,
        "actions": actions,
        "extra_fields": extra_fields,
    }
    return parsed_agent_request

def get_tools_usage_instructions(tools):
    usage_instructions = {}
    for tool_type, tool in tools.items():
        usage_instructions[tool_type] = tool.get_usage_inst()
    message = "Your action did not match any of the available tools, please use one of the following tools: \n"
    message += "\n".join([f"- {tool_type}: {usage_instructions[tool_type]}" for tool_type in usage_instructions])
    return message

def get_tool_type_for_action(tools, action, extra_fields):
    finish = extra_fields.get("finish", False)
    if finish:
        # finish is a special keyword to indicate the end of the generation, if it is set to True, return "finish"
        return "finish"
    if len(tools) == 1:
        # if there is only one tool, always use that tool
        return list(tools.keys())[0]
    # if there are multiple tools, try to find the tool that matches the action
    for tool_type, tool in tools.items():
        _, valid = tool.parse_action(action)
        if valid:
            return tool_type
    return None

def get_multi_tool_observations(tools, trajectory_ids, actions, extra_fields):
    tool_type_each_action = [get_tool_type_for_action(tools, actions[i], extra_fields[i]) for i in range(len(actions))]
    # get observations for each tool
    all_tool_types = set(tool_type_each_action)
    all_observations = [None for _ in range(len(actions))]
    all_dones = [False for _ in range(len(actions))]
    all_valids = [False for _ in range(len(actions))]
    for tool_type in all_tool_types:
        tool = tools[tool_type]
        tool_indices = [i for i, t in enumerate(tool_type_each_action) if t == tool_type]
        if tool_type is None:
            # not a single tool matched the action
            observations = [get_tools_usage_instructions(tools) for _ in range(len(actions))]
            dones = [False for _ in range(len(actions))]
            valids = [False for _ in range(len(actions))]
        else:
            tool_trajectory_ids = [trajectory_ids[i] for i in tool_indices]
            tool_actions = [actions[i] for i in tool_indices]
            tool_extra_fields = [extra_fields[i] for i in tool_indices]
            observations, dones, valids = tool.get_observations(tool_trajectory_ids, tool_actions, tool_extra_fields)
        for i, idx in enumerate(tool_indices):
            all_observations[idx] = observations[i]
            all_dones[idx] = dones[i]
            all_valids[idx] = valids[i]
    return all_observations, all_dones, all_valids

def main(
    host: str = "0.0.0.0",
    port: int = 5000,
    num_workers: int = 1,
    tool_type: str = "base",
):
    """
    Start the server with the given tool type(s).
    
    Args:
        host: The host address
        port: The port number
        num_workers: The number of workers
        tool_type: The tool type(s) to use, separated by commas (default: base)

    """

    app = FastAPI()
    if isinstance(tool_type, str):
        tool_type = (tool_type,)
    if "finish" not in tool_type:
        tool_type = tool_type + ("finish",)
    
    print(f"Starting server with tools: {tool_type}")
    tools = {t_type: get_tool_cls(t_type)(num_workers=num_workers) for t_type in tool_type}
    
    # Print all registered tools
    print("Available Tools:")
    for tool in ALL_TOOLS:
        print(f"  - {tool}", " (serving)" if tool in tools else "")
    
    @app.post("/get_observation")
    async def get_observation(request: Request):
        data = await request.json()
        parsed_data = parse_agent_request(data)
        observations, dones, valids = get_multi_tool_observations(tools, **parsed_data)
        result = {"observations": observations, "dones": dones, "valids": valids}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)
    
    uvicorn.run(app, host=host, port=port, log_level="info")
    

if __name__ == "__main__":
    fire.Fire(main)