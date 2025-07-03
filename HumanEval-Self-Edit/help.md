convert .txt -> .md

# Setup

```
pip install openai tiktoken tqdm datasets
cd HumanEval-Self-Edit
```

# Steps

Add things to output.json:

- model: `deepseek/deepseek-r1:free`

Generate datasets: `python CodeEditor/generate_datasets/generate_datasets_for_edit_humaneval.py`


# Files

(i might be wrong)

`CodeEditor/generate_datasets/generate_datasets/generate_datasets_for_copy.py` = template

`CodeEditor/generate_datasets/generate_datasets/generate_datasets_for_edit_humaneval.py` = generate datasets for HumanEval

`model.py` = model + change output file*

# Notes

- fix model.py, blank responses