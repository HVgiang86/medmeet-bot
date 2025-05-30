import asyncio
import os
import textwrap
from typing import List, Tuple, TypedDict

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# Define paths for persistence
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "faiss_db")  # Changed from chroma_db
DATA_DIRECTORY = os.path.join(os.getcwd(), "data")


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def load_documents(directory=DATA_DIRECTORY):
    """Load and split documents from PDF directory"""
    console.print("[bold]Loading documents...[/bold]")
    loader = PyPDFDirectoryLoader(directory)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    console.print(f"[bold green]Loaded {len(chunks)} document chunks[/bold green]")
    return chunks


def get_vectorstore(force_reload=False):
    """Create or load vector database from document chunks"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(PERSIST_DIRECTORY) and not force_reload:
        console.print("[bold]Loading existing vector database...[/bold]")
        vectorstore = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
        console.print(f"[bold green]Loaded vector database with {vectorstore.index.ntotal} documents[/bold green]")
    else:
        console.print("[bold]Creating new vector embeddings...[/bold]")
        chunks = load_documents()
        if not chunks:
            console.print("[bold yellow]No documents loaded, cannot create vector store.[/bold yellow]")
            vectorstore = FAISS.from_documents(documents=chunks,  # This will be an empty list
                embedding=embeddings)
            console.print("[bold yellow]Created an empty vector database as no documents were found.[/bold yellow]")

        else:
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            console.print(f"[bold green]Created vector database with {len(chunks)} documents[/bold green]")

        # Ensure the directory exists before saving
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        vectorstore.save_local(PERSIST_DIRECTORY)
        console.print(f"[bold green]Saved vector database.[/bold green]")

    return vectorstore


def get_llm_gemini():
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = os.environ.get("GEMINI_MODEL")
    if model_name is None:
        model_name = "gemini-2.0-flash"

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_tokens=None, timeout=None, max_retries=2,
        google_api_key=api_key  # ,convert_system_message_to_human=True # Depending on model and Langchain version
    )

    return llm


# --- Medical Service Recommendation ---
class MedicalServicePydantic(BaseModel):
    id: str = Field(description="Unique identifier for the medical service")
    name: str = Field(description="Name of the medical service")
    description: str = Field(
        description="Detailed description of the medical service")  # Add other relevant fields like category, price_range, etc. if needed


async def fetch_medical_services() -> List[MedicalServicePydantic]:
    """
    Fetch medical services from the external API.
    Calls the actual API instead of using mocked data.
    """
    console.print("[bold]Fetching medical services from API...[/bold]")

    try:
        # Import the actual function from our services
        from backend.services.medical_service import fetch_medical_services as api_fetch_medical_services

        # Get the actual data from API - returns List[Tuple[str, str]] with (id, name)
        api_services = await api_fetch_medical_services()

        # Convert to MedicalServicePydantic objects
        # Since API only returns id and name, we'll use a generic description
        services = []
        for service_id, service_name in api_services:
            service = MedicalServicePydantic(id=service_id, name=service_name,
                description=f"Medical service: {service_name}"  # Generic description since API doesn't provide it
            )
            services.append(service)

        console.print(f"[bold green]Fetched {len(services)} medical services from API[/bold green]")
        return services

    except Exception as e:
        console.print(f"[bold red]Error fetching medical services from API: {e}[/bold red]")
        console.print("[bold yellow]Falling back to mocked data...[/bold yellow]")

        mock_services_data = [
            {
                "id": "681249263578fdf93a64a431",
                "name": "Khoa cơ xương khớp",
                "description": "Khoa cơ xương khớp"
            },
            {
                "id": "681b316cd239ea73d96e41ae",
                "name": "Khám chuyên khoa tai mũi họng",
                "description": "Khám chuyên khoa tai mũi họng"
            },
            {
                "id": "681b31d2d239ea73d96e41de",
                "name": "Khám da liễu",
                "description": "Khám da liễu"
            },
            {
                "id": "681c7a0ce776a156ff8878e8",
                "name": "Khám chuyên khoa tim mạch",
                "description": "Khám chuyên khoa tim mạch"
            },
            {
                "id": "681c8b9a8c0e58b3475f2df2",
                "name": "Khám chuyên khoa thần kinh",
                "description": "Khám chuyên khoa thần kinh"
            },
            {
                "id": "681ebfe92923d8eb1c6e486e",
                "name": "Gói khám tổng quát tim mạch",
                "description": "Gói khám tổng quát tim mạch"
            },
            {
                "id": "6820d4993d88166d7eaafb0d",
                "name": "Khám nội soi tai mũi họng",
                "description": "Khám nội soi tai mũi họng"
            },
            {
                "id": "6820d4cb3d88166d7eaafb5d",
                "name": "Gói khám kiểm tra tiêu hóa",
                "description": "Gói khám kiểm tra tiêu hóa"
            },
            {
                "id": "6820d7e13d88166d7eaafd04",
                "name": "Khám sản phụ khoa",
                "description": "Khám sản phụ khoa"
            },
            {
                "id": "6834aca92932943e34bf5d82",
                "name": "Khám chuyên khoa tai mũi họng",
                "description": "Khám chuyên khoa tai mũi họng"
            }
        ]
        services = [MedicalServicePydantic(**service) for service in mock_services_data]
        console.print(f"[bold yellow]Using {len(services)} mocked medical services[/bold yellow]")
        return services


def fetch_medical_services_sync() -> List[MedicalServicePydantic]:
    """
    Synchronous wrapper for fetch_medical_services.
    Used when async execution is not available.
    """
    try:
        return asyncio.run(fetch_medical_services())
    except Exception as e:
        console.print(f"[bold red]Error in sync fetch: {e}[/bold red]")
        # Return empty list or minimal fallback
        return []


class ServiceRecommendationOutput(BaseModel):
    """Output model for the service recommendation chain."""
    recommended_service_ids: List[str] = Field(description="List of recommended medical service IDs")


class ChatHistoryInput(TypedDict):
    chat_history: List[Tuple[str, str]]  # Or List[BaseMessage] depending on how you store it
    input: str  # Add medical_services: List[MedicalServicePydantic] if passing directly to the chain


def format_services_for_prompt(services: List[MedicalServicePydantic]) -> str:
    """Formats the list of medical services into a string for the prompt."""
    formatted_services = "".join([f"- ID: {s.id}, Name: {s.name}, Description: {s.description}" for s in services])
    return formatted_services


vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})


# --- Dedicated Service Recommendation Chain ---
def build_dedicated_service_recommend_chain():
    """
    Builds a dedicated RAG chain to recommend medical services.
    This chain uses create_history_aware_retriever for handling chat history.
    """
    llm = get_llm_gemini()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 1. Prompt to condense the user's question for service recommendation context
    CONDENSE_SERVICE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([("system",
                                                                          "You are an AI assistant. Based on the chat history and the human's follow-up input, rephrase the follow-up input to be a standalone question in English, specifically focused on what medical services might be appropriate. Output only the standalone question."),
        MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])

    # 2. History-aware retriever
    history_aware_service_retriever = create_history_aware_retriever(llm, retriever, CONDENSE_SERVICE_QUESTION_PROMPT)

    # 3. Prompt for the final LLM call to recommend services
    service_recommend_final_prompt_template = """
    # Bối cảnh
    Bạn là một trợ lý AI y tế có tên MedBot.
    Nhiệm vụ của bạn là đề xuất các dịch vụ y tế phù hợp cho người dùng.
    Bạn sẽ nhận được lịch sử trò chuyện, một câu hỏi/yêu cầu cụ thể từ người dùng, và một danh sách các dịch vụ y tế hiện có.
    Chỉ đề xuất các dịch vụ từ danh sách được cung cấp. Hãy xem xét kỹ lưỡng mô tả của từng dịch vụ để đảm bảo tính phù hợp.

    # Lịch sử trò chuyện đầy đủ (để tham khảo ngữ cảnh)
    {chat_history_str}

    # Câu hỏi/yêu cầu của người dùng
    {input}

    # Danh sách dịch vụ y tế hiện có
    {medical_services}

    # Tài liệu liên quan
    {context}

    # Yêu cầu
    Dựa trên thông tin trên, hãy chọn ra tối đa 3 ID dịch vụ y tế phù hợp nhất từ danh sách trên.
    Chỉ trả lời bằng một danh sách các ID dịch vụ, mỗi ID trên một dòng và bắt đầu baăằng ký hiệu ID_. Ví dụ:
    ID_680f4dd80158fdd3760c435a
    ID_680f4dd80158fdd3760c435a

    Nếu không có dịch vụ nào phù hợp hoặc không chắc chắn, hãy trả lời bằng một dòng trống hoặc không đưa ra ID nào.
    """
    SERVICE_RECOMMEND_FINAL_PROMPT = ChatPromptTemplate.from_template(service_recommend_final_prompt_template)

    # Helper functions
    from langchain_core.messages import HumanMessage, AIMessage
    def convert_tuples_to_lc_messages(chat_history_tuples: List[Tuple[str, str]]):
        messages = []
        for role, content in chat_history_tuples:
            if role.lower() == "user":
                messages.append(HumanMessage(content=content))
            elif role.lower() == "bot" or role.lower() == "ai" or role.lower() == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def format_history_tuples_to_string(chat_history_tuples: List[Tuple[str, str]]) -> str:
        return "\n".join([f"[{role}] {content}" for role, content in chat_history_tuples])

    def parse_llm_output_to_service_ids(llm_output_str: str) -> ServiceRecommendationOutput:
        ids = [line.strip() for line in llm_output_str.split('\n') if line.strip().startswith("ID_")]
        ids = [id_.replace("ID_", "").strip() for id_ in ids if id_.startswith("ID_")]
        return ServiceRecommendationOutput(recommended_service_ids=ids)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the chain
    chain = ({"input": lambda x: x["input"],
                 "chat_history_str": lambda x: format_history_tuples_to_string(x.get("chat_history", [])),
                 "medical_services": lambda x: format_services_for_prompt(fetch_medical_services_sync()), "context": (
                                                                                                                         lambda
                                                                                                                             x: {
                                                                                                                             "input":
                                                                                                                                 x[
                                                                                                                                     "input"],
                                                                                                                             "chat_history": convert_tuples_to_lc_messages(
                                                                                                                                 x.get(
                                                                                                                                     "chat_history",
                                                                                                                                     []))}) | history_aware_service_retriever | format_docs} | SERVICE_RECOMMEND_FINAL_PROMPT | llm | StrOutputParser() | RunnableLambda(
        parse_llm_output_to_service_ids))

    return chain


# --- END Dedicated Service Recommendation Chain ---


def build_rag_chain():
    """Build the complete RAG chain using modern LangChain patterns"""
    # Get the vectorstore

    # Initialize LLM
    llm = get_llm_gemini()
    # Query reformulation prompt
    conversational_prompt = ChatPromptTemplate.from_messages([("system", """
        Given a chat history between an AI chatbot and user
        that chatbot's message marked with [bot] prefix and user's message marked with [user] prefix,
        and given the latest user question which might reference context in the chat history,
        formulate a standalone question which can be understood without the chat history.
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        Regardless of the language input, please translate and write it in English.
        """), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])

    # Response generation prompt
    rag_prompt = ChatPromptTemplate.from_messages([("system", """
        # Bối cảnh

        Bạn là trợ lý ảo AI về sức khoẻ tên là MedBot.
        Bạn có kiến thức chuyên sâu về y khoa, y tế, chăm sóc sức khoẻ cá nhân.
        Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng về vấn đề sức khoẻ cá nhân, tư vấn thăm khám cho người dùng.
        Hãy trở lên đáng tin, tuân thủ những nội dung từ tài liệu bên dưới.
        Hãy trả lời ngắn gọn và đầy đủ. Dù đầu vào là ngôn ngữ nào, luôn trả lời bằng Tiếng Việt.
        Hãy trả lời với Markdown format.

        # Tài liệu

        Dưới đây là một số tài liệu Tiếng Anh liên quan:
        {context}
        """), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])

    recommend_prompt = ChatPromptTemplate.from_messages([("system", """
            # Bối cảnh

            Bạn là trợ lý ảo AI về sức khoẻ tên là MedBot.
            Bạn có kiến thức chuyên sâu về y khoa, y tế, chăm sóc sức khoẻ cá nhân.
            Bạn được cung cấp các tài liệu về y khoa liên quan.
            DỰa vào các tài liệu này cùng với hiểu biết của bạn. Hãy đưa ra 2 - 4 câu hỏi gợi ý liên quan đến chủ đề này cho người dùng.

            Những câu hỏi này có thể được dùng để hỏi bạn trong tương lai.
            Chỉ đưa ra câu hỏi gợi ý mà không đưa ra lời chào, lời nói hay bất kỳ thông tin nào khác. Dù đầu vào là ngôn ngữ nào, luôn trả lời bằng Tiếng Việt. . Mỗi câu hỏi gợi ý nên ngắn gọn trong khoảng 15 từ.

            Những câu gợi ý của bạn phải cách nhau bởi dấu gạch đứng "|". Ví dụ:
            Bị đau bụng nên uống thuốc gì | Bị đau bụng nên kiêng gì

            Một số ví dụ:
            1.
            Câu hỏi của người dùng: "Tôi bị đau bụng"
            Những câu hỏi gợi ý của bạn:
            - Bị đau bụng nên uống thuốc gì
            - Bị đau bụng nên kiêng gì
            - Làm thế nào để giảm đau bụng ngay
            - Nguyên nhân dẫn đến đau bụng

            2.
            Câu hỏi của người dùng: "Tôi thấy buồn nôn"
            Những câu hỏi gợi ý của bạn:
            - Làm sao để giảm buồn nôn ngay
            - Buồn nôn do đâu

            # Tài liệu

            Dưới đây là một số tài liệu Tiếng Anh liên quan:
            {context}
            """), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, conversational_prompt)

    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)

    # Combine into final chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    class RecommendationOutput(BaseModel):
        """Output model for the recommendation chain."""
        recommendations: List[str] = Field(description="Recommendation question for user")

    recommend_answer_chain = create_stuff_documents_chain(llm, recommend_prompt)
    recommend_chain = create_retrieval_chain(history_aware_retriever, recommend_answer_chain)

    return rag_chain, recommend_chain


## global
rag, recommend = build_rag_chain()

result = recommend.invoke({"input": "Toi bị đau bụng", "chat_history": [], })

print(result)
# rag = build_rag_chain() # recommend is now part of rag's output
# service_recommend = build_service_recommend_chain() # This was the previous more manual one
dedicated_service_recommend_chain = build_dedicated_service_recommend_chain()  # New dedicated chain

chat_history = [("user", "Tôi bị đau bụng")]
# Invoke the dedicated service recommendation chain
response = dedicated_service_recommend_chain.invoke(
    {"input": "Tôi bị đau bụng", "chat_history": chat_history  # List[Tuple[str, str]]
    })

# Print the recommended service IDs
print("Recommended Service IDs:")
for service_id in response.recommended_service_ids:
    print(service_id)
