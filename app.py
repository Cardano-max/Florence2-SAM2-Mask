from typing import Optional
import numpy as np
import gradio as gr
import spaces
import supervision as sv
import torch
from PIL import Image
from io import BytesIO
import PIL.Image
import cv2
import json
import time
import os
from diffusers.utils import load_image
import json
from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.sam import load_sam_image_model, run_sam_inference
import copy
import random
import time
import boto3
from datetime import datetime

DEVICE = torch.device("cuda")
MAX_SEED = np.iinfo(np.int32).max

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)


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

def upload_image_to_r2(image, account_id, access_key, secret_key, bucket_name):
    print("upload_image_to_r2", account_id, access_key, secret_key, bucket_name)
    connectionUrl = f"https://{account_id}.r2.cloudflarestorage.com"
    img = Image.fromarray(image)
    s3 = boto3.client(
        's3',
        endpoint_url=connectionUrl,
        region_name='auto',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    current_time = datetime.now().strftime("%Y/%m/%d/%H%M%S")
    image_file = f"generated_images/{current_time}_{random.randint(0, MAX_SEED)}.png"
    buffer = BytesIO()
    img.save(buffer, "PNG")
    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket_name, image_file)
    print("upload finish", image_file)
    return image_file

@spaces.GPU()
@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(image_url, task_prompt, text_prompt=None, dilate=0, merge_masks=False, return_rectangles=False, invert_mask=False, upload_to_r2=False, account_id="", bucket="", access_key="", secret_key="") -> Optional[Image.Image]:
    
    if not task_prompt or not image_url:
        gr.Info("Please enter a image url or task prompt.")
        return None, json.dumps({"status": "failed", "message": "invalid parameters"})
   
    with calculateDuration("Download Image"):
        print("start to fetch image from url", image_url)
        image_input = load_image(image_url)
        if not image_input:
            return None, json.dumps({"status": "failed", "message": "invalid image"})


    # start to parse prompt
    with calculateDuration("run_florence_inference"):
        print(task_prompt, text_prompt)
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=task_prompt,
            text=text_prompt
        )
    with calculateDuration("run_detections"):
        # start to dectect
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
    # json_result = json.dumps([])
    # print(detections)
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
        with calculateDuration("generate segmenet mask"):
            # using sam generate segments images        
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            if len(detections) == 0:
                gr.Info("No objects detected.")
                return None, json.dumps({"status": "failed", "message": "no object tetected"})
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

    # return results
    json_result = {"status": "success", "message": "", "image_urls": []}
    if upload_to_r2:
        with calculateDuration("upload to r2"):
            image_urls = []
            for image in images:
                url = upload_image_to_r2(image, account_id, access_key, secret_key, bucket)
                image_urls.append(url)
            json_result["image_urls"] = image_urls
            json_result["message"] = "upload to r2 success"
    else:
        json_result["message"] = "not upload"
        
    return images, json.dumps(json_result)


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
            image_url =  gr.Textbox(label='Image url', placeholder='Enter text prompts (Optional)', info="The image_url parameter allows you to input a URL pointing to an image.")
            task_prompt = gr.Dropdown(['<OD>', '<CAPTION_TO_PHRASE_GROUNDING>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<OCR_WITH_REGION>', '<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>', '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>'], value="<CAPTION_TO_PHRASE_GROUNDING>", label="Task Prompt", info="check doc at [Florence](https://huggingface.co/microsoft/Florence-2-large)")
            text_prompt = gr.Textbox(label='Text prompt', placeholder='Enter text prompts')
            submit_button = gr.Button(value='Submit', variant='primary')

            with gr.Accordion("Advance Settings", open=False):
                dilate = gr.Slider(label="dilate mask", minimum=0, maximum=50, value=10, step=1, info="The dilate parameter controls the expansion of the mask's white areas by a specified number of pixels. Increasing this value will enlarge the white regions, which can help in smoothing out the mask's edges or covering more area in the segmentation.")
                merge_masks = gr.Checkbox(label="Merge masks", value=False, info="The merge_masks parameter combines all the individual masks into a single mask. When enabled, the separate masks generated for different objects or regions will be merged into one unified mask, which can simplify further processing or visualization.")
                return_rectangles = gr.Checkbox(label="Return Rectangles", value=False, info="The return_rectangles parameter, when enabled, generates masks as filled white rectangles corresponding to the bounding boxes of detected objects, rather than detailed contours or segments. This option is useful for simpler, box-based visualizations.")
                invert_mask = gr.Checkbox(label="invert mask", value=False, info="The invert_mask option allows you to reverse the colors of the generated mask, changing black areas to white and white areas to black. This can be useful for visualizing or processing the mask in a different context.")
            
            with gr.Accordion("R2 Settings", open=False):
                upload_to_r2 = gr.Checkbox(label="Upload to R2", value=False)
                with gr.Row():
                    account_id = gr.Textbox(label="Account Id", placeholder="Enter R2 account id")
                    bucket = gr.Textbox(label="Bucket Name", placeholder="Enter R2 bucket name here")

                with gr.Row():
                    access_key = gr.Textbox(label="Access Key", placeholder="Enter R2 access key here")
                    secret_key = gr.Textbox(label="Secret Key", placeholder="Enter R2 secret key here")
                
        with gr.Column():
            image_gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto")
            json_result = gr.Code(label="JSON Result", language="json")
    
    submit_button.click(
        fn=process_image,
        inputs=[image_url, task_prompt, text_prompt, dilate, merge_masks, return_rectangles, invert_mask, upload_to_r2, account_id, bucket, access_key, secret_key],
        outputs=[image_gallery, json_result],
        show_api=False
    )

demo.queue()
demo.launch(debug=True, show_error=True, share=True)