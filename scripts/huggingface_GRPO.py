# train_grpo.py
from taco_loader import get_taco_data

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
import json
import os

def get_taco_data():
    if (os.path.exists('./taco_train')):
        TACO_train = load_from_disk('./taco_train')
    else:
        TACO_train = load_dataset("BAAI/TACO", split="train")
        TACO_train.save_to_disk('./taco_train')

    if (os.path.exists('./taco_valid')):
        TACO_valid = load_from_disk('./taco_valid')
    else:
        TACO_valid = load_dataset("BAAI/TACO", split="test")
        TACO_valid.save_to_disk('./taco_valid')

    TACO_train = TACO_train.rename_column('question', 'prompt')
    TACO_valid = TACO_valid.rename_column('question', 'prompt')

    TACO_train = TACO_train.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <think> and </think>. Then, provide your solution between <solution> and </solution>"},
            {"role": "user",   "content": x["prompt"]},
        ]
    })

    TACO_valid = TACO_valid.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <think> and </think>. Then, provide your solution between <solution> and </solution>"},
            {"role": "user",   "content": x["prompt"]},
        ]
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

def perfect_format(completions, **kwargs):
    pattern = re.compile(f"{reasoning_start}.+?{reasoning_end}.*?{solution_start}.+?{solution_end}")
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format(completions, **kwargs):
    rewards = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        rewards.append(score)
    return rewards

def dummy_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        response = completions[0]['content']
        rewards.append(-abs(response.length-100))
    return rewards
        

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10, per_device_train_batch_size=24)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = get_peft_model(model, peft_config)

reasoning_start = "<think>" 
reasoning_end   = "</think>"
solution_start  = "<solution>"
solution_end    = "</solution>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

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
    processing_class=tokenizer,
    reward_funcs=[
        reward_check,
        soft_format,
        perfect_format
    ],
    args=training_args,
    train_dataset=TACO_train,
    eval_dataset=TACO_valid
)
trainer.train()

example = TACO_train[0]
prompt_messages = example["prompt"]
prompt_text = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
print(f"Input prompt:\n{prompt_text}\n")
inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated response:\n{generated_text}")

solution = extract_tags(generated_text, solution_start, solution_end)
print(f"\nExtracted solution:\n{solution}")
