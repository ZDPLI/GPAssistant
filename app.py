import os
from llama_cpp import Llama

MODEL_PATH = os.environ.get("MODEL_PATH", "Lingshu-7B.Q4_K_M.gguf")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Download the model from https://huggingface.co/mradermacher/Lingshu-7B-GGUF "
        "and set MODEL_PATH to its location."
    )

llm = Llama(model_path=MODEL_PATH)

# Example usage (will only work if the model exists)
if __name__ == "__main__":
    print(llm("Hello"))
