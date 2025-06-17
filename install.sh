git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

pip install transformers
pip install peft
pip install bitsandbytes
pip install datasets
pip install trl
pip install tf-keras
pip install --upgrade pandas
pip install numpy==1.26.4
pip install gpustat
