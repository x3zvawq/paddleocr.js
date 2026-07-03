import * as ort from "onnxruntime-node";
import { DetectionService, getTextDetectionPresetOptions } from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/seal_text_det.png";
const model = modelPath("ppocr_v4_mobile_seal_det", "PP-OCRv4_mobile_seal_det_infer.onnx");

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the PP-OCRv4 seal detection model first."),
]);
const detector = new DetectionService(
    ort,
    session,
    getTextDetectionPresetOptions("PP-OCRv4_mobile_seal_det")
);
const boxes = await detector.run(image);

console.dir(
    {
        imagePath,
        model,
        detected: boxes.length,
        polygons: boxes.slice(0, 5).map((box) => box.polygon ?? box.points),
    },
    { depth: null }
);
