import streamlit as st
import pandas as pd
import time
import os
import logging
import sys
from typing import Dict, Any
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import utils.func
import utils.constants as const
from utils.vllm_manager import VLLMManager

# Initialize vLLM manager
vllm_manager = VLLMManager()

# App title
st.set_page_config(page_title="Jetson Copilot - Download Model", menu_items=None)

st.subheader("List of Models Already Downloaded")
with st.spinner("Checking existing models..."):
    models = vllm_manager.list_models()
    models_data = []
    for model in models:
        # Get model size in MiB
        model_path = model["path"]
        size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(model_path)
            for filename in filenames
        ) / (1024 * 1024)

        models_data.append(
            (
                model["name"],
                size,
                "HuggingFace",  # Format is always HuggingFace for vLLM
                "Unknown",  # Family info not directly available
                "Unknown",  # Parameter size not directly available
                "Original",  # No quantization by default
            )
        )

    logging.info(f"{len(models)} models found!")
    df = pd.DataFrame(
        models_data,
        columns=["Name", "Size(MiB)", "Format", "Family", "Parameter", "Quantization"],
    )
    if len(models) != 0:
        st.dataframe(df.style.format({"Size(MiB)": "{:,.1f}"}))


def on_newmodel_name_change():
    logging.info("on_newmodel_name_change()")
    newmodel_name = st.session_state.my_newmodel_name
    if newmodel_name.strip():
        logging.info("Name supplied")
        st.session_state.download_model_disabled = False
    else:
        logging.info("Name NOT supplied")
        st.session_state.download_model_disabled = True


def download_model():
    logging.info("download_model()")
    newmodel_name = st.session_state.my_newmodel_name
    with container_status:
        try:
            my_bar = st.progress(0, text="Initializing download...")
            # Download model using vLLM manager
            # Since HuggingFace download doesn't provide progress, we'll update based on steps
            my_bar.progress(25, text="Downloading model files...")
            vllm_manager.download_model(newmodel_name)
            my_bar.progress(100, text="Download completed!")
            st.success(f"Successfully downloaded model: {newmodel_name}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            st.error(f"Error downloading model: {str(e)}", icon="🚨")


st.subheader("Download a New Model")
st.info(
    "Enter the Hugging Face model name (e.g., 'meta-llama/Llama-2-7b-chat-hf'). "
    "Check available models on [Hugging Face Hub](https://huggingface.co/models).",
    icon="ℹ️",
)

model_name = st.text_input(
    "Name of model to download",
    key="my_newmodel_name",
    on_change=on_newmodel_name_change,
)
st.button(
    "Download Model",
    key="my_button",
    on_click=download_model,
    disabled=st.session_state.get("download_model_disabled", True),
)
container_status = st.container()

st.page_link("app.py", label="Back to home", icon="🏠")
