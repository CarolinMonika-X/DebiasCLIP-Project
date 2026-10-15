from huggingface_hub import snapshot_download
import os

print("Downloading actual image files (.npz). Idhu konjam neram aagum...")

# Idhu dataset repo la irukkura ellaa files-ayum download pannum
snapshot_download(
    repo_id="harvardairobotics/FairVLMed",
    repo_type="dataset",
    local_dir="./data/FairVLMed", # Unga project folder kulla save aagum
    local_dir_use_symlinks=False
)

print("Download complete! Files are in ./data/FairVLMed")