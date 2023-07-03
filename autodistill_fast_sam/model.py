import os
from dataclasses import dataclass

import torch
import sys
import numpy as np

import supervision as sv
import subprocess
import cv2
from autodistill.detection import CaptionOntology, DetectionBaseModel
from helpers import combine_detections

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

installation_instructions = [
    f"cd {HOME}/.cache/autodistill/ && git clone https://github.com/CASIA-IVA-Lab/FastSAM",
    f"cd {HOME}/.cache/autodistill/FastSAM/ && pip install -r requirements.txt",
    f"mkdir -p {HOME}/.cache/autodistill/FastSAM/weights/",
    f"wget -P {HOME}/.cache/autodistill/FastSAM/weights/ https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt",
    f"wget -P {HOME}/.cache/autodistill/FastSAM/weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
]


@dataclass
class FastSAM(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        if not os.path.exists(f"{HOME}/.cache/autodistill/FastSAM/"):
            for cmd in installation_instructions:
                subprocess.run(cmd, shell=True)

        sys.path.insert(0, f"{HOME}/.cache/autodistill/FastSAM/")

        from fastsam import FastSAM, FastSAMPrompt

        # add weights
        self.model = FastSAM(f"{HOME}/.cache/autodistill/FastSAM/weights/FastSAM.pt")
        self.prompter = FastSAMPrompt
        self.ontology = ontology

        pass

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        img = cv2.imread(input)

        imgsz = img.shape[0]

        results = self.model(
            input,
            device=DEVICE,
            retina_masks=True,
            imgsz=imgsz,
            conf=confidence,
            iou=0.9
        )

        prompt_process = self.prompter(input, results, device=DEVICE)

        results = []
        class_ids = []

        for i, prompt in enumerate(self.ontology.prompts()):
            ann = prompt_process.text_prompt(text=prompt)
            print(len(ann))
            for mask in ann:
                results.append(
                    sv.Detections(
                        mask=np.array([mask]),
                        xyxy=sv.mask_to_xyxy(np.array([mask])),
                        class_id=np.array([0]),
                        confidence=np.array([1]),
                    )
                )
                class_ids.append(i)

        return combine_detections(results, overwrite_class_ids=class_ids)
