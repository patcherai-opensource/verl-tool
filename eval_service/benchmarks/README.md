# Benchmarks

Here are all the benchmarks we have tested with the evaluation service.
You need to use `verl_tool` env to launch the eval service first, and get the vt_base_url (set in the script, default is `http://0.0.0.0:5000)
```bash
bash eval_service/scripts/start_api_service.sh &
```
Then you run the eval script in different envs for each benchmark. (see instructions below)

## MathEvalHarness
### Install
```bash
cd math-evaluation-harness
uv venv --python 3.10
uv pip install -r requirements.txt
```

### Eval
```bash
bash scripts/run_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
bash scripts/run_vt_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH # set vt_base_url in the script
```

- see its `README.md` for how to modify system prompt

## BigCodeBench

### Install
```bash
git clone https://github.com/jdf-prog/bigcodebench.git
cd bigcodebench
git checkout verltool
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt
uv pip install protobuf==3.20
```
### Eval
```bash
export BIGCODEBENCH_TIMEOUT_PER_TASK=30 # originally 240
split=instruct # instruct or complete
subset=full # hard or full
bigcodebench.evaluate \
  --model VerlTool/acecoder-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-250-step \
  --execution local \
  --split $split \
  --subset $subset \
  --backend openai \
  --bs 2048 \
  --base_url http://0.0.0.0:5000 
```

- modify system prompt in `bigcodebench/gen/util/openai_request.py`

## evalplus

### Install
```bash
git clone https://github.com/jdf-prog/evalplus.git
cd evalplus
git checkout verltool
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
```

### Eval
```bash
export OPENAI_API_KEY="{KEY}" # https://platform.deepseek.com/api_keys
evalplus.evaluate --model "VerlTool/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-mtrl-sys8-110-step"              \
                  --dataset humaneval           \
                  --base-url http://0.0.0.0:5000  \
                  --backend openai --greedy

export OPENAI_API_KEY="{KEY}" # https://platform.deepseek.com/api_keys
evalplus.evaluate --model "VerlTool/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-mtrl-sys8-110-step"              \
                  --dataset mbpp           \
                  --base-url http://0.0.0.0:5000  \
                  --backend openai --greedy
```

- modify system prompt in `evalplus/gen/util/openai_request.py`

## LiveCodeBench
### Install
```bash
git clone https://github.com/jdf-prog/LiveCodeBench
cd LiveCodeBench
git checkout verltool
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### Eval
```bash
export OPENAI_API_KEY="{KEY}" # random key
export OPENAI_BASE_URL="http://0.0.0.0:5000" 
python -m lcb_runner.runner.main --model "VerlTool/acecoder-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-250-step"  --scenario codegeneration --evaluate --start_date 2023-09-01 --end_date --multiprocess 64
python -m lcb_runner.runner.main --model "VerlTool/acecoder-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-250-step"  --scenario codegeneration --evaluate  --release_version release_v4 --multiprocess 64
python -m lcb_runner.runner.main --model "VerlTool/acecoder-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-250-step"  --scenario codegeneration --evaluate  --release_version release_v4 --multiprocess 64 --n 1  --temperature 0 --max_tokens 4096 --top_p 0.95 --num_process_evaluate 32
```

- modify system prompt in `lcb_runner/runner/oai_runner.py`