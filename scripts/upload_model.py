from huggingface_hub import HfApi
from pathlib import Path


LOCAL_MODEL_DIR = Path(
"/projects/data/teams/tts_team/agri_training/finetuned_checkpoints_parquet/gemma3-27b-it-lora-2700/step-005625/lora_2700_2ndCrash/step-008375/hf_converted"
)


REPO_ID = "SayantanJoker/agri_model_8375_v2"

PRIVATE = False  




def main():
    api = HfApi()

    # Create repo if not exists
    print("Creating repo (if not exists)...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True
    )

    print(" Uploading folder to Hugging Face...")
    api.upload_folder(
        folder_path=str(LOCAL_MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload merged Gemma-3-27B model",
        ignore_patterns=["*.tmp", "*.log"]
    )

    print("✅ Upload completed successfully!")


if __name__ == "__main__":
    main()
