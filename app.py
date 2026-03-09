from flask import Flask, render_template, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from src.prompt import *
import os
from uuid import uuid4


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
SEARCH_K = int(os.environ.get("SEARCH_K", "3"))
FETCH_K = int(os.environ.get("FETCH_K", "10"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "120"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_DB_URL = os.environ.get("CHAT_DB_URL", "sqlite:///chat_history.db")

from langchain_ollama import ChatOllama

chatModel = ChatOllama(
    model=MODEL,
    temperature=0,
    num_predict=NUM_PREDICT,
    base_url=OLLAMA_BASE_URL,
)

retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": SEARCH_K, "fetch_k": FETCH_K, "lambda_mult": 0.5}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def get_session_id() -> str:
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid4())
        session["session_id"] = session_id
    return session_id

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    chat_history = SQLChatMessageHistory(
        session_id=get_session_id(),
        connection=CHAT_DB_URL,
        table_name="message_store",
    )
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history.messages,
    })
    chat_history.add_message(HumanMessage(content=msg))
    chat_history.add_message(AIMessage(content=response["answer"]))
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
