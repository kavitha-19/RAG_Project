import streamlit as st
from rag.utilis import two_llms
from rag.chroma_db import collection
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio

st.title("RAG Application")

# Clear GPU memory cache before loading the model
torch.cuda.empty_cache()

# Create a container to hold the conversation
conversation_container = st.container()

# Initialize an empty conversation list if it's the first load
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display the conversation
with conversation_container:
    for user_query, gemma_response, openai_response in st.session_state.conversation:
        st.markdown(f"**User:** {user_query}")
        st.markdown(f"**Gemma Model 2b Response:** {gemma_response}")
        st.markdown(f"**OpenAI GPT-4 Response:** {openai_response}")

# Input box and submit button placed at the bottom using columns
input_col, button_col = st.columns([8, 2])  # 8: for input, 2: for button
query_input = input_col.text_input("Enter your query:")
submit_button = button_col.button("Submit")

# Load the model only if it hasn't been loaded yet
@st.cache_resource
def load_model():
    local_model_dir = "./local"
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    model = AutoModelForCausalLM.from_pretrained(local_model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Function to handle async query processing
async def handle_query(query, model, tokenizer, device):
    response = await two_llms(query, model, tokenizer, device, collection=collection())
    return response

if submit_button:
    if query_input:
        # Run the async function and fetch the result
        response = asyncio.run(handle_query(query_input, model, tokenizer, device))

        # Append the new conversation to the list
        st.session_state.conversation.append((query_input, response[0], response[1]))

        # Display the updated conversation immediately
        with conversation_container:
            st.markdown(f"**User:** {query_input}")
            st.markdown(f"**Gemma Model 2b Response:** {response[0]}")
            st.markdown(f"**OpenAI GPT-4 Response:** {response[1]}")
