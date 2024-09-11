import os
import sys
import numpy as np
import torch
import supervision as sv
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoImageProcessor
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Add the necessary directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
florence_model_dir = os.path.join(current_dir, 'Florence-2-large-ft')
sam_model_dir = os.path.join(current_dir, 'florence-sam-masking')
sys.path.extend([florence_model_dir, sam_model_dir])

# Import the custom Florence modules
from configuration_florence2 import Florence2Config
from modeling_florence2 import Florence2ForConditionalGeneration
from processing_florence2 import Florence2Processor

# Import SAM modules
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Florence model setup
FLORENCE_MODEL_PATH = florence_model_dir

# SAM model setup
SAM_CHECKPOINT = os.path.join(florence_model_dir, "checkpoints", "sam2_hiera_large.pt")
SAM_CONFIG_DIR = os.path.join(sam_model_dir, "configs")

def load_florence_model(device: torch.device, model_path: str):
    config = Florence2Config.from_pretrained(model_path)
    model = Florence2ForConditionalGeneration.from_pretrained(model_path, config=config).to(device).eval()
    return model, config

def load_sam_image_model(device: torch.device, config_dir: str, checkpoint: str = SAM_CHECKPOINT):
    GlobalHydra.instance().clear()
    initialize(config_path=config_dir, version_base=None)
    cfg = compose(config_name="sam2_hiera_l")
    
    model = SAM2Base(**cfg.model)
    model.to(device)
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    return SAM2ImagePredictor(sam_model=model)

def run_sam_inference(model, image: np.ndarray, detections: sv.Detections):
    model.set_image(image)
    bboxes = sorted(detections.xyxy, key=lambda bbox: bbox[0])
    mask, score, _ = model.predict(box=bboxes, multimask_output=False)
    if len(mask.shape) == 4:
        mask = np.squeeze(mask)
    detections.mask = mask.astype(bool)
    return detections

def process_image(
    image_path,
    task_prompt: str,
    text_prompt: str = None,
    dilate: int = 0,
    merge_masks: bool = False,
    return_rectangles: bool = False,
    invert_mask: bool = False
):
    # Load models
    florence_model, florence_config = load_florence_model(DEVICE, FLORENCE_MODEL_PATH)
    sam_model = load_sam_image_model(DEVICE, config_dir=SAM_CONFIG_DIR)

    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(FLORENCE_MODEL_PATH)
    image_processor = AutoImageProcessor.from_pretrained(FLORENCE_MODEL_PATH)

    # Create processor instance
    processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

    # Load image
    image = Image.open(image_path)

    # Prepare inputs
    inputs = processor(text=[task_prompt, text_prompt], images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output = florence_model.generate(**inputs, max_length=512)

    # Post-process output
    result = processor.post_process_generation(output[0], task=task_prompt, image_size=image.size)

    # Create detections
    detections = sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2,
        result=result,
        resolution_wh=image.size
    )

    # Run SAM inference
    detections = run_sam_inference(sam_model, np.array(image.convert("RGB")), detections)

    # Process masks
    masks = []
    if return_rectangles:
        for bbox in detections.xyxy:
            mask = np.zeros(image.size[::-1], dtype=np.uint8)
            cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 255, thickness=cv2.FILLED)
            masks.append(mask)
    else:
        for i in range(len(detections.mask)):
            mask = detections.mask[i].astype(np.uint8) * 255
            if dilate > 0:
                kernel = np.ones((dilate, dilate), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            masks.append(mask)

    if merge_masks:
        merged_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            merged_mask = cv2.bitwise_or(merged_mask, mask)
        masks = [merged_mask]

    if invert_mask:
        masks = [cv2.bitwise_not(mask) for mask in masks]

    return masks, result

def display_results(image_path, masks, parsed_result):
    if masks:
        plt.figure(figsize=(15, 5))
        for i, mask in enumerate(masks):
            plt.subplot(1, len(masks), i+1)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Mask {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Overlay mask on original image
        original_image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(original_image)
        for mask in masks:
            plt.imshow(mask, alpha=0.5, cmap='jet')
        plt.axis('off')
        plt.title("Masks overlaid on original image")
        plt.show()

        print("Parsed Result:", parsed_result)
    else:
        print("No masks were generated.")


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_prompt = "person"

    result_masks, parsed_result = process_image(
        image_path=image_path,
        task_prompt=task_prompt,
        text_prompt=text_prompt,
        dilate=10,
        merge_masks=False,
        return_rectangles=False,
        invert_mask=False
    )

    display_results(image_path, result_masks, parsed_result)