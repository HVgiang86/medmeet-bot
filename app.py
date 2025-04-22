import streamlit as st
import uuid
from main import build_rag_chain, to_markdown

# Page config
st.set_page_config(
    page_title="MedBot - AI Health Assistant",
    page_icon="🏥",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .user-message {
        background-color: #e6f7ff;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f0f2f6;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🏥 MedBot - AI Health Assistant")
st.markdown("Ask questions about health and medical topics in any language.")

# Session state for chat history and session tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-session-{str(uuid.uuid4())}"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    # Initialize the RAG chain once per session with LangSmith tracing
    with st.spinner("Initializing MedBot..."):
        st.session_state.rag_chain = build_rag_chain()

# Display chat history
for message in st.session_state.chat_history:
    role = message.split("]")[0] + "]"
    content = message.split("]")[1].strip()
    if role == "[user]":
        st.markdown(f'<div class="user-message">👤 You: {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🏥 MedBot: {content}</div>', unsafe_allow_html=True)

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question:", key="user_input")
    submit_button = st.form_submit_button("Send")

# Process user input
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append("[user]" + str(user_input))
    
    # Convert session_state.chat_history to the format expected by the model
    model_chat_history = []
    for msg in st.session_state.chat_history:
        model_chat_history.append(msg)
    
    # Get response from the model with LangSmith tracing
    with st.spinner("MedBot is thinking..."):
        try:
            # Create a unique query ID within this session
            query_id = f"{st.session_state.session_id}-query-{len(model_chat_history)//2}"
            
            # Invoke the chain with request ID metadata for LangSmith
            response = st.session_state.rag_chain.invoke({
                "input": user_input, 
                "chat_history": model_chat_history[:-1],  # Exclude the current message
            }, 
            {"metadata": {"request_id": query_id}})
            
            answer = response["answer"]
            
            # Add bot response to chat history
            st.session_state.chat_history.append("[bot]" + str(answer))
            
            # Rerun to display the new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add session info in sidebar
with st.sidebar:
    st.subheader("About MedBot")
    st.markdown("""
    MedBot is an AI health assistant that provides information from medical documents.
    
    **Features:**
    - Answers health-related questions
    - Supports multiple languages
    - Maintains conversation context
    - Retrieves information from medical documents
    - LangSmith monitoring for all queries
    """)
    
    st.subheader("Session Info")
    st.info(f"Session ID: {st.session_state.session_id}")
    
    # Add link to LangSmith dashboard if available
    st.markdown("[View traces in LangSmith Dashboard](https://smith.langchain.com/)", unsafe_allow_html=True)
    
    # Clear chat history button
    if st.button("Clear Conversation"):
        # Create a new session ID for the new conversation
        st.session_state.session_id = f"streamlit-session-{str(uuid.uuid4())}"
        # Initialize a new chain with the new session ID
        st.session_state.rag_chain = build_rag_chain(run_name=st.session_state.session_id)
        # Clear chat history
        st.session_state.chat_history = []
        st.rerun() 