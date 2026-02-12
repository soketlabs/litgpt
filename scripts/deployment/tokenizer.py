from transformers import AutoTokenizer
from huggingface_hub import HfApi

# ================= CONFIG =================
LOCAL_TOKENIZER_PATH = "/projects/data/significant_ckpt/tokenizer_hf_normalized/"   # path to your tokenizer folder
HF_USERNAME = "SayantanJoker"          # replace with your HF username
REPO_NAME = "Eka_tokenizer"                # repo name on HF
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
# =========================================


def main():
    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)

    # Push tokenizer to Hugging Face (public repository)
    tokenizer.push_to_hub(
        repo_id=REPO_ID,
        private=False
    )

    print(f"✅ Tokenizer uploaded successfully!")
    print(f"🔗 https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
