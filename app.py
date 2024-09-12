from typing import Optional
import numpy as np
import gradio as gr
import spaces
import supervision as sv
import torch
from PIL import Image
from io import BytesIO
import PIL.Image
import requests
import cv2
import json
import time
import os
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Optional
from typing import List, Dict, Optional
from PIL import Image
from utils.sam import load_sam_model, run_sam_inference


from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.sam import load_sam_image_model, run_sam_inference
SAM_IMAGE_MODEL = load_sam_model(device=DEVICE)

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

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


@spaces.GPU()
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(image_input, image_url, task_prompt, text_prompt=None, dilate=0, merge_masks=False, return_rectangles=False, invert_mask=False) -> Optional[List[Image.Image]]:
    
    if not image_input:
        gr.Info("Please upload an image.")
        return None
    
    if not task_prompt:
        gr.Info("Please enter a task prompt.")
        return None
   
    if image_url:
        with calculateDuration("Download Image"):
            print("start to fetch image from url", image_url)
            image_input = fetch_image_from_url(image_url)
            print("fetch image success")

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
    
    # Create detections manually
    detections = []
    if task_prompt in result and 'bboxes' in result[task_prompt]:
        bboxes = result[task_prompt]['bboxes']
        labels = result[task_prompt]['labels']
        for bbox, label in zip(bboxes, labels):
            detections.append({
                'bbox': bbox,
                'label': label
            })
    
    images = []
    if return_rectangles:
        with calculateDuration("generate rectangle mask"):
            # create mask in rectangle
            (image_width, image_height) = image_input.size
            merge_mask_image = np.zeros((image_height, image_width), dtype=np.uint8)
            # sort from left to right
            detections = sorted(detections, key=lambda x: x['bbox'][0])
            for detection in detections:
                bbox = detection['bbox']
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
            sam_results = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            if len(sam_results) == 0:
                gr.Info("No objects detected.")
                return None
            print("mask generated:", len(sam_results))
            kernel_size = dilate
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            for mask in sam_results:
                mask = mask.astype(np.uint8) * 255
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

    # Convert numpy arrays to PIL Images
    pil_images = []
    for mask in images:
        if len(mask.shape) == 2:
            pil_images.append(Image.fromarray(mask))
        elif len(mask.shape) == 3:
            pil_images.append(Image.fromarray(mask[:, :, 0]))  # Take the first channel if it's a 3D array
        else:
            print(f"Unexpected mask shape: {mask.shape}")

    return pil_images


def update_task_info(task_prompt):
    task_info = {
        '<OD>': "Object Detection: Detect objects in the image.",
        '<CAPTION_TO_PHRASE_GROUNDING>': "Phrase Grounding: Link phrases in captions to corresponding regions in the image.",
        '<DENSE_REGION_CAPTION>': "Dense Region Captioning: Generate captions for different regions in the image.",
        '<REGION_PROPOSAL>': "Region Proposal: Propose potential regions of interest in the image.",
        '<OCR_WITH_REGION>': "OCR with Region: Extract text and its bounding regions from the image.",
        '<REFERRING_EXPRESSION_SEGMENTATION>': "Referring Expression Segmentation: Segment the region referred to by a natural language expression.",
        '<REGION_TO_SEGMENTATION>': "Region to Segmentation: Convert region proposals into detailed segmentations.",
        '<OPEN_VOCABULARY_DETECTION>': "Open Vocabulary Detection: Detect objects based on open vocabulary concepts.",
        '<REGION_TO_CATEGORY>': "Region to Category: Assign categories to proposed regions.",
        '<REGION_TO_DESCRIPTION>': "Region to Description: Generate descriptive text for specified regions."
    }
    return task_info.get(task_prompt, "Select a task to see its description.")



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil', label='Upload image')
            image_url =  gr.Textbox(label='Image url', placeholder='Enter text prompts (Optional)', info="The image_url parameter allows you to input a URL pointing to an image.")
            task_prompt = gr.Dropdown(['<OD>', '<CAPTION_TO_PHRASE_GROUNDING>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<OCR_WITH_REGION>', '<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>', '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>'], value="<CAPTION_TO_PHRASE_GROUNDING>", label="Task Prompt", info="check doc at [Florence](https://huggingface.co/microsoft/Florence-2-large)")
            text_prompt = gr.Textbox(label='Text prompt', placeholder='Enter text prompts')
            submit_button = gr.Button(value='Submit', variant='primary')

            with gr.Accordion("Advance Settings", open=False):
                dilate = gr.Slider(label="dilate mask", minimum=0, maximum=50, value=10, step=1, info="The dilate parameter controls the expansion of the mask's white areas by a specified number of pixels. Increasing this value will enlarge the white regions, which can help in smoothing out the mask's edges or covering more area in the segmentation.")
                merge_masks = gr.Checkbox(label="Merge masks", value=False, info="The merge_masks parameter combines all the individual masks into a single mask. When enabled, the separate masks generated for different objects or regions will be merged into one unified mask, which can simplify further processing or visualization.")
                return_rectangles = gr.Checkbox(label="Return Rectangles", value=False, info="The return_rectangles parameter, when enabled, generates masks as filled white rectangles corresponding to the bounding boxes of detected objects, rather than detailed contours or segments. This option is useful for simpler, box-based visualizations.")
                invert_mask = gr.Checkbox(label="invert mask", value=False, info="The invert_mask option allows you to reverse the colors of the generated mask, changing black areas to white and white areas to black. This can be useful for visualizing or processing the mask in a different context.")
            
        with gr.Column():
            image_gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto")
            # json_result = gr.Code(label="JSON Result", language="json")
    
   
    image_url.change(
        fn=fetch_image_from_url,
        inputs=[image_url],
        outputs=[image]
    )
    
    submit_button.click(
        fn=process_image,
        inputs=[image, image_url, task_prompt, text_prompt, dilate, merge_masks, return_rectangles, invert_mask],
        outputs=[image_gallery],
        show_api=False
    )

demo.queue()
demo.launch(debug=True, show_error=True, share=True)