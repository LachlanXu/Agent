import json
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain_community.vectorstores import FAISS


# 定义文件路径
file_path = r"C:\Users\xhx20\Desktop\Agent\datasets\TeleQnA.txt"
persist_directory = r"C:\Users\xhx20\Desktop\Agent\agent_web\vectorstore"

# 读取文件内容
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)


# 提取问题及相关信息，转换为 Document 格式
documents = []
for key, value in data.items():
    content = (
        f"Question: {value['question']}\n"
        f"Explanation: {value['explanation']}\n"
        f"Category: {value['category']}"
    )
    documents.append(Document(page_content=content, metadata={"id": key}))

if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
    # 如果目录存在且不为空，加载现有的 vectorstore
    print("Loading existing vectorstore...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(
            api_key="sk-nI4Wy7Ws2Tj2piRR6a65045dEcD7446c84BfF8BeD27a7d13",
            base_url="https://free.v36.cm/v1/",
        ),
    )
else:
    # 否则，构建新的 vectorstore
    print("Creating new vectorstore...")
    vectorstore = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(
            api_key="sk-nI4Wy7Ws2Tj2piRR6a65045dEcD7446c84BfF8BeD27a7d13",
            base_url="https://free.v36.cm/v1/",
        ),
        persist_directory=persist_directory  # 可选，设置持久化路径
    )

retriever = vectorstore.as_retriever(
    search_stype="similarity",
    search_kwargs={"k":2}
)