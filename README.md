# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask

# How to run?

### STEP 01 - Create and activate a conda environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### STEP 02 - Install the requirements

```bash
pip install -r requirements.txt
```

Important: run the install and app commands from the activated `medibot` environment. If you use your system `python` instead of Conda, you may see `ModuleNotFoundError` errors for packages like `flask` or `dotenv`.

### STEP 03 - Create a `.env` file in the root directory

Add your Pinecone credentials and app settings:

```ini
PINECONE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_INDEX_NAME=medical-chatbot
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
FLASK_SECRET_KEY=change-me
```

### STEP 04 - Start Ollama

Make sure Ollama is installed, then start the Ollama server:

```bash
ollama serve
```

In another terminal, pull the model if you do not already have it:

```bash
ollama pull llama3.2:3b
```

### STEP 05 - Store embeddings in Pinecone

```bash
python store_index.py
```

### STEP 06 - Run the Flask app

```bash
python app.py
```

Now open:

```bash
http://localhost:8080
```


### Techstack Used:

- Python
- LangChain
- Flask
- Ollama
- Pinecone
