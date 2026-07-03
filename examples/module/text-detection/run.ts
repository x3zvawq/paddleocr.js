import * as ort from "onnxruntime-node";
import { DetectionService, getTextDetectionPresetOptions } from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/general_ocr_001.png";
const model = modelPath("ppocr_v6_small", "PP-OCRv6_small_det_infer.onnx");

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the PP-OCRv6 small detection model first."),
]);
const detector = new DetectionService(
    ort,
    session,
    getTextDetectionPresetOptions("PP-OCRv6_small_det")
);
const boxes = await detector.run(image);

console.dir(
    {
        imagePath,
        model,
        detected: boxes.length,
        boxes: boxes.slice(0, 10).map((box) => ({
            x: Math.round(box.x),
            y: Math.round(box.y),
            width: Math.round(box.width),
            height: Math.round(box.height),
        })),
    },
    { depth: null }
);
