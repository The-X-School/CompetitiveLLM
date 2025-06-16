# train_grpo.py
from datasets import load_dataset
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

train = load_dataset("trl-lib/tldr", split="train")
valid = load_dataset("trl-lib/tldr", split="validation")

TACO_train = load_dataset("BAAI/TACO", split="train")
TACO_valid = load_dataset("BAAI/TACO", split="test")

print("Finished loading TACO")

TACO_train = TACO_train.rename_column('question', 'prompt')
TACO_train = TACO_train.rename_column('solutions', 'completion')

TACO_valid = TACO_valid.rename_column('question', 'prompt')
TACO_valid = TACO_valid.rename_column('solutions', 'completion')
#prompt_to_completion_valid = {TACO_valid['prompt'][i]: TACO_valid['completion'][i] for i in range(len(TACO_valid))}
prompt_to_completion_train = {TACO_train['prompt'][i]: [TACO_train['input_output'][i]['inputs'], TACO_train['input_output'][i]['outputs']] for i in range(len(TACO_train))}
prompt_to_time = {TACO_train['prompt'][i]: TACO_train['time_limit'][i] for i in range(len(TACO_train))}

#this is the code that will be used to test the code
def test_code(code, cases, ex_out, time_limit):
    score = 0
    correct_cases = 0

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
        first_mem = proc.memory_info().rss / 1024
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

def reward_check(completions, **kwargs):
    #completions is the code???
    rewards=[]
    for completion in completions:
        code = extract_tags(completion, "<solution>", "</solution>")
        time_limit = prompt_to_time[kwargs['prompt']]
        cases = prompt_to_completion_train[kwargs['prompt']][0]
        ex_out = prompt_to_completion_train[kwargs['prompt']][1]
        reward = test_code(code, cases, ex_out, time_limit)
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = get_peft_model(model, peft_config)

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=reward_check,
    args=training_args,
    train_dataset=TACO_train,
    eval_dataset=TACO_valid
)
trainer.train()