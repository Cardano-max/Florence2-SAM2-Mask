import os
import sys
import numpy as np
import torch
import supervision as sv
from PIL import Image
import requests
from io import BytesIO
import cv2
import time
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
from Florence-2-large-ft.configuration_florence2 import Florence2Config
from Florence-2-large-ft.modeling_florence2 import Florence2ForConditionalGeneration
from Florence-2-large-ft.processing_florence2 import Florence2Processor\

# Import SAM modules
from florence-sam-masking.sam2.modeling.sam2_base import SAM2Base
from florence-sam-masking.sam2.sam2_image_predictor import SAM2ImagePredictor

# Import utility functions
from florence_sam_masking.utils.florence import load_florence_model, run_florence_inference
from florence_sam_masking.utils.sam import load_sam_image_model, run_sam_inference

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Florence model setup
FLORENCE_MODEL_PATH = florence_model_dir

# SAM model setup
SAM_CHECKPOINT = os.path.join(florence_model_dir, "checkpoints", "sam2_hiera_large.pt")
SAM_CONFIG_DIR = os.path.join(sam_model_dir, "configs")

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE, config=os.path.join(SAM_CONFIG_DIR, "sam2_hiera_l.yaml"), checkpoint=SAM_CHECKPOINT)

def fetch_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        return None

class calculateDuration:
    def __init__(self, activity_name=""):
        self.activity_name = activity_name

    def __enter__(self):
        self.start_time = time.time()
        self.start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))
        print(f"Activity: {self.activity_name}, Start time: {self.start_time_formatted}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time))
        
        if self.activity_name:
            print(f"Elapsed time for {self.activity_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")
        
        print(f"Activity: {self.activity_name}, End time: {self.start_time_formatted}")

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(image_input, image_url, task_prompt, text_prompt=None, dilate=0, merge_masks=False, return_rectangles=False, invert_mask=False):
    
    if not image_input and not image_url:
        print("Please provide an image or image URL.")
        return None
    
    if not task_prompt:
        print("Please enter a task prompt.")
        return None
   
    if image_url:
        with calculateDuration("Download Image"):
            print("start to fetch image from url", image_url)
            image_input = fetch_image_from_url(image_url)
            print("fetch image success")
    elif isinstance(image_input, str):
        image_input = Image.open(image_input)

    # start to parse prompt
    with calculateDuration("FLORENCE"):
        print(task_prompt, text_prompt)
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=task_prompt,
            text=text_prompt
        )
    with calculateDuration("sv.Detections"):
        # start to detect
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
    
    images = []
    if return_rectangles:
        with calculateDuration("generate rectangle mask"):
            # create mask in rectangle
            (image_width, image_height) = image_input.size
            bboxes = detections.xyxy
            merge_mask_image = np.zeros((image_height, image_width), dtype=np.uint8)
            # sort from left to right
            bboxes = sorted(bboxes, key=lambda bbox: bbox[0])
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(merge_mask_image, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
                clip_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                cv2.rectangle(clip_mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
                images.append(clip_mask)
            if merge_masks:
                images = [merge_mask_image] + images
    else:
        with calculateDuration("generate segment mask"):
            # using sam generate segments images        
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            if len(detections) == 0:
                print("No objects detected.")
                return None
            print("mask generated:", len(detections.mask))
            kernel_size = dilate
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            for i in range(len(detections.mask)):
                mask = detections.mask[i].astype(np.uint8) * 255
                if dilate > 0:
                    mask = cv2.dilate(mask, kernel, iterations=1)
                images.append(mask)

            if merge_masks:
                merged_mask = np.zeros_like(images[0], dtype=np.uint8)
                for mask in images:
                    merged_mask = cv2.bitwise_or(merged_mask, mask)
                images = [merged_mask]
    if invert_mask:
        with calculateDuration("invert mask colors"):
            images = [cv2.bitwise_not(mask) for mask in images]

    return images, result

if __name__ == "__main__":
    # Example usage
    image_path = "Jimin.jpeg"  # Replace with your image path
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_prompt = "person"

    result_masks, parsed_result = process_image(
        image_input=image_path,
        image_url=None,
        task_prompt=task_prompt,
        text_prompt=text_prompt,
        dilate=10,
        merge_masks=False,
        return_rectangles=False,
        invert_mask=False
    )

    if result_masks:
        print(f"Number of masks generated: {len(result_masks)}")
        print("Parsed Result:", parsed_result)
    else:
        print("No masks were generated.")