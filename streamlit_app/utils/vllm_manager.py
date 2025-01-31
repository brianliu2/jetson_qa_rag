from vllm import LLM, SamplingParams
import logging
import sys
import os
from typing import List, Dict, Optional

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class VLLMManager:
    def __init__(self):
        self.llm = None
        self.model_name = None
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
        )
        os.makedirs(self.models_dir, exist_ok=True)

    def load_model(self, model_name: str) -> None:
        """
        Load a model using vLLM
        """
        try:
            self.llm = LLM(model=model_name)
            self.model_name = model_name
            logging.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def list_models(self) -> List[Dict]:
        """
        List all available models in the models directory
        """
        models = []
        if os.path.exists(self.models_dir):
            for model_dir in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, model_dir)
                if os.path.isdir(model_path):
                    config_path = os.path.join(model_path, "config.json")
                    if os.path.exists(config_path):
                        models.append({"name": model_dir, "path": model_path})
        return models

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate text using the loaded model
        """
        if not self.llm:
            raise RuntimeError("No model loaded. Please load a model first.")

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text

    def download_model(self, model_name: str) -> None:
        """
        Download a model from Hugging Face Hub
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Download model and tokenizer
            save_path = os.path.join(self.models_dir, model_name)

            logging.info(f"Downloading model {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Save model and tokenizer
            logging.info(f"Saving model to {save_path}...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            logging.info(f"Successfully downloaded model: {model_name}")
        except Exception as e:
            logging.error(f"Error downloading model {model_name}: {str(e)}")
            raise
