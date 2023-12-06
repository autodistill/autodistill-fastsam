import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
import torch

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

from .helpers import combine_detections

HOME = os.path.expanduser("~")
AUTODISTILL_DIR = os.path.join(HOME, ".cache", "autodistill")
FASTSAM_DIR = os.path.join(AUTODISTILL_DIR, "FastSAM")
FASTSAM_WEIGHTS_DIR = os.path.join(FASTSAM_DIR, "weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_command(cmd, directory=None):
    result = subprocess.run(
        cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise ValueError(
            f"Command '{' '.join(cmd)}' failed to run. Stdout: {result.stdout}, Stderr: {result.stderr}"
        )


def install_fastsam_dependencies():
    commands = [
        (["git", "clone", "https://github.com/CASIA-IVA-Lab/FastSAM"], AUTODISTILL_DIR),
        (["pip", "install", "--quiet", "-r", "requirements.txt"], FASTSAM_DIR),
        (["mkdir", "-p", FASTSAM_WEIGHTS_DIR], None),
        (
            [
                "wget",
                "-q",
                "-P",
                FASTSAM_WEIGHTS_DIR,
                "https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt",
            ],
            None,
        ),
        (
            [
                "wget",
                "-q",
                "-P",
                FASTSAM_WEIGHTS_DIR,
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            ],
            None,
        ),
        (
            ["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"],
            AUTODISTILL_DIR,
        ),
        (
            ["pip", "install", "--quiet", "-e", "."],
            os.path.join(AUTODISTILL_DIR, "GroundingDINO"),
        ),
    ]

    for cmd, dir in commands:
        run_command(cmd, dir)


@dataclass
class FastSAM(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        if not os.path.exists(FASTSAM_DIR):
            install_fastsam_dependencies()

        sys.path.insert(0, FASTSAM_DIR)
        from fastsam import FastSAM, FastSAMPrompt

        self.model = FastSAM(os.path.join(FASTSAM_WEIGHTS_DIR, "FastSAM.pt"))
        self.prompter = FastSAMPrompt
        self.ontology = ontology

    def predict(self, input: Any, confidence: int = 0.1) -> sv.Detections:
        import cv2
        img = load_image(input, return_format="cv2")
        imgsz = img.shape[0]

        print("Running FastSAM...")
        print(imgsz)

        results = self.model(
            input,
            device=DEVICE,
            retina_masks=True,
            imgsz=imgsz,
            conf=confidence,
            iou=0.6
        )

        prompt_process = self.prompter(input, results, device=DEVICE)

        results = []
        class_ids = []

        for i, prompt in enumerate(self.ontology.prompts()):
            ann = prompt_process.text_prompt(text=prompt)
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
