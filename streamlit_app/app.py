import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from PIL import Image
import time
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import utils.func
import utils.constants as const
from utils.vllm_manager import VLLMManager

# Initialize vLLM manager
vllm_manager = VLLMManager()

# App title
st.set_page_config(page_title="Jetson Copilot", menu_items=None)

AVATAR_AI = Image.open("./images/jetson-soc.png")
AVATAR_USER = Image.open("./images/user-purple.png")


def find_saved_indexes():
    return utils.func.list_directories(const.INDEX_ROOT_PATH)


def load_index(index_name):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
    dir = f"{const.INDEX_ROOT_PATH}/{index_name}"
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    index = load_index_from_storage(storage_context)
    return index


# Check if default model exists
default_model = "Qwen/Qwen1.5-3B-Chat"
models = vllm_manager.list_models()
model_names = [model["name"] for model in models]

if default_model not in model_names:
    with st.spinner(f"Downloading {default_model} model..."):
        try:
            vllm_manager.download_model(default_model)
            logging.info(f"Downloaded {default_model} successfully.")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")

old_index_name = ""
# Side bar
with st.sidebar:
    st.title(":airplane: Jetson Copilot")
    st.subheader("Your local AI assistant on Jetson", divider="rainbow")

    # Refresh model list
    models = vllm_manager.list_models()
    model_names = [model["name"] for model in models]

    col3, col4 = st.columns([5, 1])
    with col3:
        selected_model = st.selectbox(
            "Choose your LLM",
            model_names,
            index=(
                model_names.index(default_model) if default_model in model_names else 0
            ),
        )
        st.session_state["model"] = selected_model
        logging.info(f'> st.session_state["model"] = {st.session_state.model}')

        # Load the selected model
        try:
            vllm_manager.load_model(selected_model)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    with col4:
        st.markdown("")
    st.page_link("pages/download_model.py", label=" Download a new LLM", icon="➕")

    use_index = st.toggle("Use RAG", value=False)
    if use_index:
        col1, col2 = st.columns([5, 1])
        saved_index_list = find_saved_indexes()
        with col1:
            index = next(
                (i for i, item in enumerate(saved_index_list) if item.startswith("_")),
                None,
            )
            index_name = st.selectbox("Index", saved_index_list, index)
            logging.info(f"> index_name = {index_name}")
        with col2:
            st.markdown("")
        if old_index_name != index_name:
            old_index_name = index_name
            logging.info(f"> old_index_name = {old_index_name}")
            if index_name != None:
                with st.spinner("Loading Index..."):
                    st.session_state.index = load_index(index_name)
                    logging.info(f" ### Loading Index '{index_name}' completed.")
        st.page_link("pages/build_index.py", label=" Build a new index", icon="➕")

        if index_name != None:
            context_prompt = st.text_area(
                "System prompt with context",
                """You are a chatbot, able to have normal interactions, as well as talk about NVIDIA Jetson embedded AI computer.
Here are the relevant documents for the context:\n
{context_str}
\nInstruction: Use the previous chat history, or the context above, to interact and help the user.""",
                height=240,
            )
            logging.info(f"> context_prompt = {context_prompt}")

            class VLLMWrapper:
                def __init__(self, manager):
                    self.manager = manager

                def complete(self, prompt):
                    return self.manager.generate(prompt)

            # Initialize chat engine with vLLM
            vllm_wrapper = VLLMWrapper(vllm_manager)
            st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                chat_mode="context",
                streaming=True,
                memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
                llm=vllm_wrapper,
                context_prompt=context_prompt,
                verbose=True,
            )

# Initialize history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me any question about NVIDIA Jetson embedded AI computer!",
            "avatar": AVATAR_AI,
        }
    ]


def model_res_generator(prompt=""):
    if use_index:
        logging.info(">>> RAG enabled:")
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        for chunk in response_stream.response_gen:
            yield chunk
    else:
        logging.info(">>> Just LLM (no RAG):")
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Generate response using vLLM
        try:
            response = vllm_manager.generate(prompt)
            yield response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            yield f"Error: {str(e)}"


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter prompt here.."):
    # Add latest message to history
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "avatar": AVATAR_USER}
    )

    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Thinking..."):
            time.sleep(1)
            message = st.write_stream(model_res_generator(prompt))
            st.session_state.messages.append(
                {"role": "assistant", "content": message, "avatar": AVATAR_AI}
            )
