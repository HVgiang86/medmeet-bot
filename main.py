import textwrap
import os
import uuid
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# Define paths for persistence
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
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
    
    # Check if the vector store already exists
    if os.path.exists(PERSIST_DIRECTORY) and not force_reload:
        console.print("[bold]Loading existing vector database...[/bold]")
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        console.print(f"[bold green]Loaded vector database with {vectorstore._collection.count()} documents[/bold green]")
    else:
        console.print("[bold]Creating new vector embeddings...[/bold]")
        chunks = load_documents()
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()
        console.print(f"[bold green]Created and saved vector database with {len(chunks)} documents[/bold green]")
    
    return vectorstore


def build_rag_chain():
    """Build the complete RAG chain using modern LangChain patterns"""
    # Get the vectorstore
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    # Initialize LLM
    llm = OllamaLLM(model="llama3.2")

    # Query reformulation prompt
    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Given a chat history between an AI chatbot and user
        that chatbot's message marked with [bot] prefix and user's message marked with [user] prefix,
        and given the latest user question which might reference context in the chat history,
        formulate a standalone question which can be understood without the chat history.
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        Regardless of the language input, please translate and write it in English.
        """), 
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}")
    ])

    # Response generation prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """
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
        """), 
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}")
    ])

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, conversational_prompt)

    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)

    # Combine into final chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def chat_loop():
    """Main chat loop function"""
    rag_chain = build_rag_chain()
    chat_history = []
    console.print("[bold blue]MedBot initialized. Ask your health-related questions (type 'exit' to quit).[/bold blue]")

    while True:
        user_input = console.input("[bold green]You: [/bold green]")

        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[bold blue]MedBot: Goodbye! Take care of your health.[/bold blue]")
            break

        try:
            response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})

            # Update chat history
            chat_history.append(("[user]", user_input))
            chat_history.append(("[bot]", response["answer"]))

            # Display response
            console.print("[bold blue]MedBot: [/bold blue]")
            console.print(to_markdown(response["answer"]))

        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")


if __name__ == "__main__":
    chat_loop()
