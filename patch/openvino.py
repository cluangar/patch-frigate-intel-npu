import logging
import os
import numpy as np
import openvino as ov
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.util.model import (
    post_process_dfine,
    post_process_rfdetr,
    post_process_yolo,
)

from frigate.util.builtin import load_labels

logger = logging.getLogger(__name__)

DETECTOR_KEY = "openvino"


class OvDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class OvDetector(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [
        ModelTypeEnum.dfine,
        ModelTypeEnum.rfdetr,
        ModelTypeEnum.ssd,
        ModelTypeEnum.yolonas,
        ModelTypeEnum.yologeneric,
        ModelTypeEnum.yolox,
    ]

    def __init__(self, detector_config: OvDetectorConfig):
        super().__init__(detector_config)
        self.ov_core = ov.Core()
        self.ov_model_type = detector_config.model.model_type

        self.h = detector_config.model.height
        self.w = detector_config.model.width
#        self.thresh = 0.3  # Default threshold for filtering detections

        self.labelmap_path = detector_config.model.labelmap_path
        self.num_label = len(load_labels(self.labelmap_path, "utf-8", 0) )

        if not os.path.isfile(detector_config.model.path):
            logger.error(f"OpenVino model file {detector_config.model.path} not found.")
            raise FileNotFoundError

        self.interpreter = self.ov_core.compile_model(
            model=detector_config.model.path, device_name=detector_config.device
        )

        self.model_invalid = False

        if self.ov_model_type not in self.supported_models:
            logger.error(f"OpenVino detector does not support {self.ov_model_type} models.")
            self.model_invalid = True

        if self.ov_model_type == ModelTypeEnum.ssd:
            model_inputs = self.interpreter.inputs
            model_outputs = self.interpreter.outputs
            if len(model_inputs) != 1 or len(model_outputs) != 1:
                logger.error("SSD models must have exactly 1 input and 1 output.")
                self.model_invalid = True
            else:
                output_shape = model_outputs[0].get_shape()
                if output_shape[0] != 1 or output_shape[1] != 1 or output_shape[3] != 7:
                    logger.error(f"SSD model output shape mismatch: {output_shape}")
                    self.model_invalid = True

        if self.ov_model_type == ModelTypeEnum.yolonas:
            model_inputs = self.interpreter.inputs
            model_outputs = self.interpreter.outputs
            if len(model_inputs) != 1:
                logger.error(f"YOLO‑NAS models must have exactly 1 input. Found {len(model_inputs)}.")
                self.model_invalid = True
            self.raw_yolonas_mode = False
            if len(model_outputs) == 2:
                self.raw_yolonas_mode = True
                logger.info("Detected YOLO‑NAS raw mode (boxes + scores).")
            elif len(model_outputs) == 1:
                logger.info("Detected YOLO‑NAS flat mode.")
                output_shape = model_outputs[0].partial_shape
                if output_shape[-1] != 7:
                    logger.error(f"YOLO‑NAS flat model output shape mismatch: {output_shape}")
                    self.model_invalid = True
            else:
                logger.error(f"Unexpected number of YOLO‑NAS outputs: {len(model_outputs)}.")
                self.model_invalid = True

        if self.ov_model_type == ModelTypeEnum.yolox:
            self.output_indexes = 0
            while True:
                try:
                    tensor_shape = self.interpreter.output(self.output_indexes).shape
                    logger.info(f"Model Output-{self.output_indexes} Shape: {tensor_shape}")
                    self.output_indexes += 1
                except Exception:
                    logger.info(f"Model has {self.output_indexes} Output Tensors")
                    break
            self.num_classes = tensor_shape[2] - 5
            logger.info(f"YOLOX model has {self.num_classes} classes")
            self.calculate_grids_strides()

    def process_yolo(self, class_id, conf, pos):
        return [
            class_id,
            conf,
            (pos[1] - (pos[3] / 2)) / self.h,
            (pos[0] - (pos[2] / 2)) / self.w,
            (pos[1] + (pos[3] / 2)) / self.h,
            (pos[0] + (pos[2] / 2)) / self.w,
        ]

    def post_process_yolonas_raw(self, output: list[np.ndarray]):
        """RKNN-style YOLO‑NAS raw post‑process adapted for OpenVINO."""
        N = output[0].shape[1]

        boxes = output[0].reshape(N, 4)
        scores = output[1].reshape(N, self.num_label)

        class_ids = np.argmax(scores, axis=1)
        scores = scores[np.arange(N), class_ids]

        args_best = np.argwhere(scores > self.thresh)[:, 0]

        num_matches = len(args_best)
        if num_matches == 0:
            return np.zeros((20, 6), np.float32)
        elif num_matches > 20:
            args_best20 = np.argpartition(scores[args_best], -20)[-20:]
            args_best = args_best[args_best20]

        boxes = boxes[args_best]
        class_ids = class_ids[args_best]
        scores = scores[args_best]

        boxes = np.transpose(
            np.vstack(
                (
                    boxes[:, 1] / self.h,
                    boxes[:, 0] / self.w,
                    boxes[:, 3] / self.h,
                    boxes[:, 2] / self.w,
                )
            )
        )

        results = np.hstack(
            (class_ids[..., np.newaxis], scores[..., np.newaxis], boxes)
        )

        # Pad to (20, 6)
        out = np.zeros((20, 6), np.float32)
        out[:results.shape[0], :] = results
        return out

    def detect_raw(self, tensor_input):
        infer_request = self.interpreter.create_infer_request()
        expected_type = self.interpreter.inputs[0].get_element_type().to_string()
        if expected_type in ("u8", "uint8"):
            tensor_input = tensor_input.astype(np.uint8)
        else:
            tensor_input = tensor_input.astype(np.float32) / 255.0
        input_tensor = ov.Tensor(array=tensor_input)        

        if self.ov_model_type == ModelTypeEnum.dfine:
            infer_request.set_tensor("images", input_tensor)
            target_sizes_tensor = ov.Tensor(np.array([[self.h, self.w]], dtype=np.int64))
            infer_request.set_tensor("orig_target_sizes", target_sizes_tensor)
            infer_request.infer()
            tensor_output = (
                infer_request.get_output_tensor(0).data,
                infer_request.get_output_tensor(1).data,
                infer_request.get_output_tensor(2).data,
            )
            return post_process_dfine(tensor_output, self.w, self.h)

        infer_request.infer(input_tensor)
        detections = np.zeros((20, 6), np.float32)

        if self.model_invalid:
            return detections
        elif self.ov_model_type == ModelTypeEnum.rfdetr:
            return post_process_rfdetr([
                infer_request.get_output_tensor(0).data,
                infer_request.get_output_tensor(1).data,
            ])
        elif self.ov_model_type == ModelTypeEnum.ssd:
            results = infer_request.get_output_tensor(0).data[0][0]
            for i, (_, class_id, score, xmin, ymin, xmax, ymax) in enumerate(results):
                if i == 20:
                    break
                detections[i] = [class_id, float(score), ymin, xmin, ymax, xmax]
            return detections
        elif self.ov_model_type == ModelTypeEnum.yolonas:
            if self.raw_yolonas_mode:
                boxes = infer_request.get_output_tensor(0).data
                scores = infer_request.get_output_tensor(1).data
                return self.post_process_yolonas_raw([boxes, scores])
            else:
                predictions = infer_request.get_output_tensor(0).data
                for i, prediction in enumerate(predictions):
                    if i == 20:
                        break
                    (_, x_min, y_min, x_max, y_max, confidence, class_id) = prediction
                    if class_id < 0:
                        break
                    detections[i] = [
                        class_id,
                        confidence,
                        y_min / self.h,
                        x_min / self.w,
                        y_max / self.h,
                        x_max / self.w,
                    ]
                return detections
        elif self.ov_model_type == ModelTypeEnum.yologeneric:
            out_tensor = [item.data for item in infer_request.output_tensors]
            return post_process_yolo(out_tensor, self.w, self.h)
        elif self.ov_model_type == ModelTypeEnum.yolox:
            out_tensor = infer_request.get_output_tensor()
            results = out_tensor.data
            results[..., :2] = (results[..., :2] + self.grids) * self.expanded_strides
            results[..., 2:4] = np.exp(results[..., 2:4]) * self.expanded_strides
            image_pred = results[0, ...]
            class_conf = np.max(image_pred[:, 5:5 + self.num_classes], axis=1, keepdims=True)
            class_pred = np.argmax(image_pred[:, 5:5 + self.num_classes], axis=1)
            class_pred = np.expand_dims(class_pred, axis=1)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= 0.3).squeeze()
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
            detections = detections[conf_mask]
            ordered = detections[detections[:, 5].argsort()[::-1]][:20]
            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(object_detected[6], object_detected[5], object_detected[:4])
            return detections

