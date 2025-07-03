from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
import os

# --- 1. Set up your repository information ---
os.environ["HF_TOKEN"] = "<>"

# The name you want for your new repository on the Hub
repo_name = "sarvam/edge_canary_transcrive_v7"

# Path to your local .nemo file
local_nemo_file_path = "/data/mayur_sarvam_ai/nemo_checkpoint/canary-flash-transcribe-v7/checkpoints/canary-flash-transcribe-v7.nemo"

api = HfApi()

new_repo_id = repo_name

# --- 2. Create the repository on the Hugging Face Hub ---
repo_url = api.create_repo(
    repo_id=f"{repo_name}",
    private=True,  # Set the repository to private
    repo_type="model",
    exist_ok=True # To avoid errors if the repo already exists
)
print(f"Repository created or already exists: {repo_url}")

# --- 4. Upload the files ---
api = HfApi()

# Upload the .nemo model file
api.upload_file(
    path_or_fileobj=local_nemo_file_path,
    path_in_repo=f"{repo_name}.nemo",  # The name of the file in the repository
    repo_id=f"{repo_name}",
    repo_type="model",
)
print(f"Uploaded {local_nemo_file_path} to the repository.")

print("\n🚀 Successfully pushed your NeMo model to the Hugging Face Hub!")