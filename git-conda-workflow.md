
# 🛠️ Git & Conda Environment Workflow

## 🔁 Conda Environment

- **Activate environment**  
  ```bash
  conda activate name
  ```

- **Update environment from YAML file**  
  ```bash
  conda env update --file file_name.yaml
  ```

- **View contents of YAML file**  
  ```bash
  cat file_name.yaml
  ```

- **Rebuild environment from YAML file**  
  ```bash
  conda env remove --name myenv
  conda env create --file env_name.yaml
  ```

> ⚠️ We're using **Python 3.10**

## 🧪 Install Dependencies

- **Install `huggingface_hub` with CLI**  
  ```bash
  pip install -U "huggingface_hub[cli]"
  ```

## 🌿 Git Basics

- **Create a new branch**  
  ```bash
  git checkout -b branch_name
  ```

- **Switch branches**  
  ```bash
  git checkout branch_name
  ```

- **View all branches**  
  ```bash
  git branch
  ```

- **Stage all files for commit**  
  ```bash
  git add .
  ```

- **Check modified/staged files**  
  ```bash
  git status
  ```

- **Commit with a signed message**  
  ```bash
  git commit -s -m "your message"
  ```

- **Push your changes (after commit)**  
  ```bash
  git push origin branch_name
  ```

- **Pull latest changes**  
  ```bash
  git pull
  ```

- **Check commit history**  
  ```bash
  git log
  ```

## ✅ Final Notes

- **Always use _Squash and Merge_** when hitting the big green button on GitHub.
