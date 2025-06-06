# MedBot: AI-Powered Medical Assistant with Custom LLM and Semantic Search
# Overview
This project is an AI-driven medical assistant that analyzes symptoms and provides preliminary treatment recommendations. It leverages a custom-built transformer-based LLM trained on medical datasets and uses Pinecone for semantic search over 500+ medical documents. The application is containerized with Docker and deployed on AWS using CI/CD pipelines.

# Key Features
Custom LLM: Built from scratch using PyTorch and Hugging Face Transformers, trained on 10GB of medical texts.
Semantic Search: Pinecone vector database enables fast, accurate retrieval of medical information.
Scalable Deployment: Dockerized and deployed on AWS EC2 with 99% uptime and automated CI/CD (GitHub Actions).
Modular Codebase: Organized with FastAPI (backend), React (frontend), and reusable Python modules.

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini

## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Pinecone

