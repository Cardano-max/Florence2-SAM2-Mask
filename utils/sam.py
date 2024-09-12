from typing import Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List, Dict
from typing import List, Dict, Optional
from PIL import Image

# SAM_CHECKPOINT = "checkpoints/sam2_hiera_small.pt"
# SAM_CONFIG = "sam2_hiera_s.yaml"
SAM_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"


def run_sam_inference(
    model: Any,
    image: Image,
    detections: List[Dict]
) -> List[np.ndarray]:
    image = np.array(image.convert("RGB"))
    model.set_image(image)
    # from left to right
    bboxes = [detection['bbox'] for detection in detections]
    bboxes = sorted(bboxes, key=lambda bbox: bbox[0])
    masks = []
    for bbox in bboxes:
        mask, _, _ = model.predict(box=bbox, multimask_output=False)
        # ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[0]
        elif len(mask.shape) == 4:
            mask = mask[0, 0]
        masks.append(mask)
    return masks


def load_sam_video_model(
    device: torch.device,
    config: str = SAM_CONFIG,
    checkpoint: str = SAM_CHECKPOINT
) -> Any:
    return build_sam2_video_predictor(config, checkpoint, device=device)


def run_sam_inference(
    model: Any,
    image: Image,
    detections: List[Dict]
) -> List[np.ndarray]:
    image = np.array(image.convert("RGB"))
    model.set_image(image)
    # from left to right
    bboxes = [detection['bbox'] for detection in detections]
    bboxes = sorted(bboxes, key=lambda bbox: bbox[0])
    masks = []
    for bbox in bboxes:
        mask, score, _ = model.predict(box=bbox, multimask_output=False)
        # dirty fix; remove this later
        if len(mask.shape) == 4:
            mask = np.squeeze(mask)
        masks.append(mask)
    return masks
