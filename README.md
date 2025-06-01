# Verl-Tool

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/imgs/logo.png">
    <img alt="VerlTool" src="assets/imgs/logo.png" width=20%>
  </picture>
</p>

<h3 align="center">
VerlTool: An unified and easy-to-extend tool-agent training framework based on verl.
</h3>

<p align="center">
| 
<a href="https://github.com/TIGER-AI-Lab/verl-tool?tab=readme-ov-file#installation"><b>Quick Start</b></a> |
  <a href="https://github.com/TIGER-AI-Lab/verl-tool/tree/main/examples/data_preprocess"><b>Data</b></a> |
  <a href="https://github.com/TIGER-AI-Lab/verl-tool/tree/main/examples/train"><b>Training Scripts</b></a> |
  <a href="https://github.com/TIGER-AI-Lab/verl-tool/tree/main/benchmarks"><b>Evaluation</b></a> 
|
</p>

---

## Table of Contents
- [News](#news)
- [Main Results](#main-results)
- [Model Checkpoints](#model-checkpoints)
- [Features](#features)
- [Installation](#installation)
- [Training](#training)
  - [Single Node Training](#single-node-training)
  - [Multi Node Training](#multi-node-training)
- [Evaluation](#evaluation)
- [ToDos](#todos)
- [Contribution](#contribute-your-own-tools)
  - [Contribution to tool libraries](#contribution-to-tool-libraries)
  - [New reward manager](#new-reward-manager)

<!-- <p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p> -->


## News
+ [2025/05/31] We release the Verl-tool training/eval code. We are working on the paper and will release it very soon.

## Main Results
### Verl-tool on Math 
**1.5B Model Performance across challenging mathematical benchmarks:**
| Model Name                                 | Tool | GSM8K | MATH 500 | Minerva Math | Olympiad Bench | AIME24 | AMC23 | Avg   |
|--------------------------------------------|-----------|--------|-----------|---------------|------------------|------------------|--------|--------|
| Qwen2.5-Math-1.5B                           | ❌        | 39.50  | 34.80     | 8.10          | 23.00            | 13.30            | 35.00  | 25.62 |
| Qwen2.5-Math-1.5B-Instruct                  | ❌        | 84.90  | 74.20     | 26.80         | 39.00            | 10.00            | 57.50  | 48.70 |
| Qwen2.5-Math-1.5B-Instruct + SimpleRL-Zoo   | ❌        | 81.90  | 70.20     | 20.60         | 33.90            | 20.00            | 55.00  | 46.90 |
| Qwen-2.5-Math-1.5B-Insturct-TIR             | ✅        | 83.70  | 76.20     | 24.30         | 41.30            | 26.70            | 55.00  | 51.20 |
| ToRL-1.5B                                   | ✅        | 85.60  | 77.80     | 29.80         | 44.00            | 26.70            | 67.50  | 55.23 |
| **Qwen-2.5-Math-1.5B + Verl-Tool**          | ✅        | **85.10** | **77.40** | **28.30**     | **44.00**        | **33.30**        | **65.00** | **55.52** |


**7B Model Performance across challenging mathematical benchmarks:**
|Model Name                                 |Tool|GSM8K|MATH 500|Minerva  Math|Olympiad  Bench|AIME24 |AMC23|Avg  |
|-------------------------------------------|---------|-----|--------|-------------|---------------|----------------|-----|-----|
|Qwen-2.5-Math-7B                           |❌        |65.50|63.60   |12.50        |25.80          |13.30           |42.50|37.20|
|Qwen2.5-Math-7B-Instruct                   |❌        |95.20|83.00   |37.10        |41.60          |16.70           |70.00|57.27|
|Qwen-2.5-Math-7B + SimpleRL-Zoo            |❌        |88.80|80.20   |26.80        |41.60          |30.00           |52.50|53.30|
|Qwen-2.5-Math-7B-Insturct-TIR              |✅        |94.60|82.40   |29.00        |50.50          |30.00           |62.50|58.17|
|TORL-7B    |✅        |92.70|82.20   |33.50        |49.90          |43.30           |65.00|61.10|
|**Qwen-2.5-Math-7B + Verl-Tool**           |✅        |**91.40**|**83.40**|**29.80**    |**50.20**      |**40.00**       |**72.50**|**61.22**|



## Model Checkpoints 
All these models are also in our [Huggingface Collection](https://huggingface.co/VerlTool). 
|Model|Link| Wandb |
|-|-|-|
|Qwen-2.5-Math-1.5B-Verl-tool|[🤗](https://huggingface.co/VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step)|[📈](https://wandb.ai/1004271927-SHU/Verl-Tool-Math?nw=nwuser1004271927)|
|Qwen-2.5-Math-7B-Verl-tool|[🤗](https://huggingface.co/VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6-310-step)|[📈](https://wandb.ai/1004271927-SHU/Verl-Tool-Math?nw=nwuser1004271927)|


## Features

- **Native RL Framework for Training Tool-Calling Agents.** `verl-tool` natively supports **multi-turn interactive loops** between agents and their environments.
- **Developer Friendly.** By integrating `verl` as a submodule, `verl-tool` abstracts away the complexity of RL training. Developers only need focus on building their tools following simple templates.
- **Fast Tool Server with Multiple Tool Types.** `verl-tool` uses [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) to asynchronously serve tools, ensuring full compatibility with verl agent training. See [./verl_tool/servers/tools](./verl_tool/servers/tools) for all available tools — each Python file represents a supported tool type.
- **Comprehensive Evaluation Suite.** See [benchmarks](benchmarks). `verl-tool` allows you to instantly evaluate your models’ capabilities. Currently, it supports benchmarks for math and code models, with more domains to be added in the future.


## Installation
We highly recommend using uv to install verl-tool. 
```bash
git submodule update --init --recursive
pip install uv # if not installed
uv sync
source .venv/bin/activate
uv pip install -e verl
uv pip install vllm==0.8.4
uv pip install flash-attn --no-build-isolation
uv pip install -e ".[acecoder,torl]"
uv pip install dill==0.4.0 fsspec==2025.3.2 protobuf==5.29.4
```
### Conda Version
```bash
git submodule update --init --recursive
conda create --name verl-tool-env python=3.10
conda activate verl-tool-env
pip install -e .
pip install -e verl
pip install vllm==0.8.4
pip install flash-attn --no-build-isolation
pip install -e ".[acecoder,torl]"
pip install dill==0.4.0
pip install fsspec==2025.3.2
pip install protobuf==5.29.4
```

## Training
We will take verl-tool-math (Tool-Integrated RL for Math) as an example. Check [examples](examples) for more training examples. 

### Data Preprocess
Prepare the data for training. You can use the provided script to preprocess the data. More examples can be found in [examples/data_preprocess](examples/data_preprocess).

```bash
python examples/data_preprocess/deep_math.py --data_source zwhe99/DeepMath-103K --local_dir data/deep_math_tool_v9 --sys_prompt_version v9 # preprocess the data and save
```

### Single Node Training
We train the **Verl-Tool-1.5B-Math** model using the command below. The training configuration can be found in [verl_tool/trainer/config/ppo_trainer.yaml](verl_tool/trainer/config/ppo_trainer.yaml) and [verl_tool/llm_agent/config.py](verl_tool/llm_agent/config.py). For other model types, examples can be found in [`examples/train`](examples/train).

```bash
bash examples/train/torl/train_qwen_1.5B_math_deep_math.sh # train the model 
```
Training tips:

1. For low VRAM GPUs, we recommend setting `do_offload=True`, `enforce_eager=True`, `tensor_parallel_size=1`, `use_dynamic_bsz=False`, and using a small `ppo_micro_batch_size_per_gpu`. For high VRAM GPUs, set `do_offload=False` and `use_dynamic_bsz=True` to speed up training.
2. If VLLM generation gets stuck, try lowering `workers_per_tool` and reducing `gpu_memory_utilization` in the script.
3. If you encounter CPU OOM issues during VLLM rollout generation, try setting `do_offload=False` and lowering `gpu_memory_utilization`.
4. See [verl performance tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for more details. 
   

### Multi Node Training
1. Head Node
```bash
ray start --head --dashboard-host=0.0.0.0 # start ray head node
bash examples/train/acecoder/train.sh # train the model
```
2. Worker Node
```bash
ray start --address='head_node_ip:6379' --block # start ray worker node
# no need to run the training script on worker node
```

### Test Servers
```bash
# Start the tool server
host=localhost
port=5000
python -m verl_tool.servers.serve --host $host --port $port --tool_type "firejail_python_code_test" --workers_per_tool 4 &
python -m verl_tool.servers.tests.test_firejail_python_code_tool firejail --url=http://localhost:5000/get_observation
```


### Logs 
The training step records are automatically saved in [`verl_step_records`](verl_step_records).

## Evaluation
**We provide comprehensive benchmarks to evaluate both math and code models in [`benchmarks`](benchmarks).**  We will add more task benchmarks in the future.

## ToDos  
- [ ] Async rollout and tool interaction for each trajectory using vllm and sglang
- [ ] integrate MCP server interface as a general tool type
- [ ] Web-browsing tool

## Contribute Your Own Tools 
### Contribution to Tool Libraries
Go to the [./verl_tool/servers/tools](./verl_tool/servers/tools) directory. Each tool has a name (e.g. `base`, `python_code`), the name of tool is exactly the name of the python file that you should create in the directory. See [./verl_tool/servers/tools/python_code.py](./verl_tool/servers/tools/python_code.py) for an example.

### New Reward Manager
Go to the [`./verl_tool/agent_workers/reward_manager`](./verl_tool/agent_workers/reward_manager) directory and add your new reward manager.  
Then, make sure update the `verl_tool/trainer/main_ppo.py` file to include your new reward manager.

