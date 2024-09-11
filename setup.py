import os
import subprocess
import sys

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode())
        return False
    return True

def setup_project():
    # Create project structure
    os.makedirs("Florence-2-large-ft/checkpoints", exist_ok=True)
    os.makedirs("Florence-2-large-ft/configs", exist_ok=True)
    os.makedirs("florence-sam-masking/utils", exist_ok=True)

    # Clone repositories if they don't exist
    if not os.path.exists("Florence-2-large-ft"):
        if not run_command("git clone https://huggingface.co/microsoft/Florence-2-large-ft"):
            print("Failed to clone Florence-2-large-ft. Continuing with setup...")
    
    if not os.path.exists("florence-sam-masking"):
        if not run_command("git clone https://huggingface.co/spaces/jiuface/florence-sam-masking"):
            print("Failed to clone florence-sam-masking. Continuing with setup...")

    # Download large model files if they don't exist
    if not os.path.exists("Florence-2-large-ft/pytorch_model.bin"):
        run_command("wget https://huggingface.co/microsoft/Florence-2-large-ft/resolve/main/pytorch_model.bin -O Florence-2-large-ft/pytorch_model.bin")
    
    if not os.path.exists("Florence-2-large-ft/checkpoints/sam2_hiera_large.pt"):
        run_command("wget https://huggingface.co/jiuface/florence-sam-masking/resolve/main/checkpoints/sam2_hiera_large.pt -O Florence-2-large-ft/checkpoints/sam2_hiera_large.pt")

    # Install dependencies
    run_command("pip install torch torchvision torchaudio transformers supervision opencv-python einops timm matplotlib hydra-core omegaconf samv2")

    print("Setup completed successfully!")

if __name__ == "__main__":
    setup_project()