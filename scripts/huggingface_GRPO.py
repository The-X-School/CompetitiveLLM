# train_grpo.py
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, get_peft_model
import multiprocessing
import psutil
import time
import builtins
import io
import sys
import re
import os
import json

def get_taco_data():
    '''if (os.path.exists('./taco_train')):
        TACO_train = load_from_disk('./taco_train')
    else:
        TACO_train = load_dataset("BAAI/TACO", split="train")
        TACO_train.save_to_disk('./taco_train')

    if (os.path.exists('./taco_valid')):
        TACO_valid = load_from_disk('./taco_valid')
    else:
        TACO_valid = load_dataset("BAAI/TACO", split="test")
        TACO_valid.save_to_disk('./taco_valid')'''

    TACO_train = load_dataset("BAAI/TACO", split="train")
    TACO_valid = load_dataset("BAAI/TACO", split="test")

    TACO_train = TACO_train \
        .rename_column('question', 'prompt') \
        .rename_column('solutions', 'completion')

    TACO_valid = TACO_valid \
        .rename_column('question', 'prompt') \
        .rename_column('solutions', 'completion')

    TACO_train = TACO_train.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <think> and </think>. Then, provide your solution between <solution> and </solution>"},
            {"role": "user",   "content": x["prompt"]},
            {"role": "assistant", "content": x["completion"]} # Include completion as a message
        ],
        "completion": [x["completion"]]
    })

    TACO_valid = TACO_valid.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <think> and </think>. Then, provide your solution between <solution> and </solution>"},
            {"role": "user",   "content": x["prompt"]},
            {"role": "assistant", "content": x["completion"]} # Include completion as a message
        ],
        "completion": [x["completion"]]
    })


    return TACO_train, TACO_valid

TACO_train, TACO_valid = get_taco_data()

#this is the code that will be used to test the code
def test_code(code, cases, ex_out, time_limit):
    score = 0
    correct_cases = 0
    time_limit = float(time_limit)

    def run_code(code, case_input, output_queue):
        # Mock input
        builtins.input = lambda: case_input

        # Capture output
        buf = io.StringIO()
        sys.stdout = buf
        try:
            exec(code)
        except Exception as e:
            output_queue.put(("ERROR\n", 0))
            return
        finally:
            sys.stdout = sys.__stdout__

        output_queue.put((buf.getvalue(), 0))  # memory will be added later

    for i in range(len(cases)):
        output_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_code, args=(code, cases[i], output_queue))
        p.start()
        proc = psutil.Process(p.pid)

        max_mem = 0
        start_time = time.time()
        while p.is_alive() and time.time() - start_time < time_limit:
            try:
                mem = proc.memory_info().rss / 1024  # in KB
                max_mem = max(max_mem, mem)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)

        if p.is_alive():
            p.terminate()
            out = "TIME LIMIT EXCEEDED\n"
            score -= 1 / len(cases) * 10
        else:
            try:
                out, _ = output_queue.get(timeout=1)
            except:
                out = "ERROR\n"
                score -= 1 / len(cases) * 50

        #print(f"Case {i+1}: Memory used ≈ {int(max_mem)} KB")

        if out == ex_out[i]:
            correct_cases += 1
    score += correct_cases / len(cases) * 100
    if correct_cases == len(cases):
        score += 25
    return score

def extract_tags(completion, tag_start, tag_end):
    return re.compile(f"{tag_start}(.*?){tag_end}").find(completion)

def reward_check(completions, input_output, time_limit, **kwargs):
    #completions is the code???
    input_output = json.loads(input_output)
    inputs, outputs = input_output['inputs'], input_output['outputs']
    rewards=[]
    for completion in completions:
        code = extract_tags(completion, "<solution>", "</solution>")
        if time_limit != None: time_limit = re.compile("(.*) second").findall(time_limit)[0]
        else: time_limit = 2
        reward = test_code(code, inputs, outputs, time_limit)
        rewards.append(reward)
    return rewards

reasoning_start = "<think>" 
reasoning_end   = "</think>"
solution_start  = "<solution>"
solution_end    = "</solution>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10, per_device_train_batch_size=24)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_check,
    args=training_args,
    train_dataset=TACO_train,
    eval_dataset=TACO_valid
)
trainer.train()