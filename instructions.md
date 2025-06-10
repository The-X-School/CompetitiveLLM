Heres the steps for pushing/pulling/making branches for this:

To enter the enviroment, do: conda activate name


To make a branch: git checkout -b branch_name
To change branches: git checkout branch_name
Push your changes(do git commit first): git push origin branch_name 
See all branches: git branch


If you want to pull, just do : git pull
How to stage all files to include in the next commit: git add .
How to commit: git commit -s -m "your message"
How to check what's modfiyed/staged: git status
(You have to commit before you push)


Check the log: git log

Update environment from a yaml file: conda env update --file file_name.yaml

Check inside yaml file: cat file_name.yaml

To rebuild the enviroment from yaml file: 

conda env remove --name myenv
conda env create --file env_name.yaml

Notes:
We're using python 3.10
install huggingface_hub: pip install -U "huggingface_hub[cli]