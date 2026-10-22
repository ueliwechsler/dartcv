"""Optional, local, text-guided whole-dart detection using pretrained OWLv2.

Construction and imports do not download weights. The first call loads the model
from the Hugging Face cache (downloading unless local_files_only=True). Boxes are
experimental observations, not tips or evidence of calibrated accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .recognition import DartBox, validate_boxes

MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


class Owlv2DartDetector:
    """Callable BGR uint8 image -> whole-dart boxes in original-image pixels.

    Scores filter text/region matches, not calibrated probabilities. NMS can
    remove genuinely overlapping darts. IDs are local to each image, not tracks.
    CPU is the default; device may explicitly select another PyTorch device.
    """

    def __init__(
        self,
        *,
        prompts: Sequence[str] = ("a dart",),
        threshold: float = 0.1,
        nms_iou: float = 0.5,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        if (
            isinstance(prompts, str)
            or not prompts
            or any(not prompt.strip() for prompt in prompts)
        ):
            raise ValueError("prompts must be a nonempty sequence of nonempty strings")
        if not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1:
            raise ValueError("threshold and nms_iou must be in [0, 1]")
        self.prompts = tuple(prompts)
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only
        self.resolved_revision = revision
        self._model = None
        self._processor = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise ImportError(
                "OWLv2 requires the optional runtime: uv sync --extra dart-detection --group dev"
            ) from exc
        options = {"revision": self.revision, "local_files_only": self.local_files_only}
        processor = Owlv2Processor.from_pretrained(
            self.model_id, use_fast=False, **options
        )
        model = Owlv2ForObjectDetection.from_pretrained(
            self.model_id, use_safetensors=True, **options
        )
        model = model.to(self.device).eval()
        self.resolved_revision = (
            getattr(model.config, "_commit_hash", None) or self.revision
        )
        self._processor = processor
        self._torch = torch
        self._model = model

    def _predict(self, rgb: np.ndarray) -> dict:
        self._load()
        inputs = self._processor(
            images=rgb, text=[list(self.prompts)], return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.image_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=self._torch.tensor([rgb.shape[:2]], device=self.device),
            threshold=self.threshold,
        )[0]
        return {
            name: result[name].detach().cpu().numpy()
            for name in ("boxes", "scores", "labels")
        }

    def __call__(self, image: np.ndarray) -> tuple[DartBox, ...]:
        """Infer once, map/filter predictions, suppress duplicates, validate boxes.

        Model loading and inference errors propagate to the caller; run_stage()
        can convert them to a failed stage without partial observations.
        """
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) < 2
        ):
            raise ValueError("expected a nonempty H x W x 3 BGR uint8 image")
        result = self._predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = np.asarray(result["boxes"], dtype=float)
        scores = np.asarray(result["scores"], dtype=float)
        labels = np.asarray(result["labels"])
        if (
            boxes.shape != (len(scores), 4)
            or scores.ndim != 1
            or labels.shape != scores.shape
        ):
            raise ValueError("invalid OWLv2 prediction shapes")
        if (
            not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
            or not np.isfinite(labels).all()
            or not (
                (labels >= 0)
                & (labels < len(self.prompts))
                & (labels == np.floor(labels))
            ).all()
        ):
            raise ValueError("invalid OWLv2 coordinates, scores, or prompt labels")
        height, width = image.shape[:2]
        boxes = np.clip(boxes, 0, [width - 1, height - 1, width - 1, height - 1])
        valid = (
            (scores >= self.threshold)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        if not len(boxes):
            return ()
        order = np.argsort(-scores, kind="stable")
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        kept = np.asarray(
            cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, self.nms_iou)
        ).reshape(-1)
        observations = tuple(
            DartBox(
                id=index + 1,
                xyxy=tuple(float(value) for value in boxes[prediction]),
                confidence=float(scores[prediction]),
                provenance=f"{self.model_id}@{self.resolved_revision}: {self.prompts[int(labels[prediction])]}",
            )
            for index, prediction in enumerate(kept)
        )
        validate_boxes(image, observations)
        return observations
