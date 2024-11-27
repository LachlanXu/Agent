from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import tool, AgentExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import os
from pydantic import BaseModel
import json
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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


os.environ["TAVILY_API_KEY"] = "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"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a powerful assistant equipped with file management capabilities.Priority is given to finding answers from the retriever. If not found, then use tavily_search."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

search = TavilySearchResults(
    max_results=1
)


retriever_tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "Searches and returns excerpts from the 2022 State of the Union.",
)
tools = [retriever_tool, search]

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key="sk-nI4Wy7Ws2Tj2piRR6a65045dEcD7446c84BfF8BeD27a7d13",
    base_url="https://free.v36.cm/v1/",
)
model_with_tools = model.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    
    | prompt_template
    | model_with_tools
    | OpenAIToolsAgentOutputParser()
)



agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

memory = ChatMessageHistory(session_id="test-session")
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

app = FastAPI(
    title="Agent WebUI",
    version="1.0",
    description="A WebUI of AI Agent",
)


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class InputModel(BaseModel):
    input: str

class OutputModel(BaseModel):
    result: dict

# add_routes(
#    app,
#    agent_executor.with_types(input_type=InputModel, output_type=OutputModel).with_config(
#        {"run_name": "agent"}
#    ),
#    path="/agent"
# )

add_routes(
   app,
   agent_with_chat_history.with_types(input_type=InputModel, output_type=OutputModel).with_config(
       {"configurable": {"session_id": "<foo>"}}
   ),
   path="/agent"
)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)


