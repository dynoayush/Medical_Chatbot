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

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

MODEL = "llama3.2:3b"   # fastest reliable option on your machine
SEARCH_K = 3
FETCH_K = 10
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80
NUM_PREDICT = 120

from langchain_ollama import ChatOllama

chatModel = ChatOllama(
    model=MODEL,
    temperature=0,
    num_predict=NUM_PREDICT,
    base_url="http://127.0.0.1:11434",
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
        connection="sqlite:///chat_history.db",
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
    app.run(host="0.0.0.0", port= 8080, debug= True)
