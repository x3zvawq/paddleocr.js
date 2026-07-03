import * as ort from "onnxruntime-node";
import { getObjectDetectionPresetOptions, ObjectDetectionService } from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/layout.png";
const model = modelPath("pp_docblocklayout", "PP-DocBlockLayout_infer.onnx");

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the PP-DocBlockLayout model first."),
]);
const detector = new ObjectDetectionService(
    ort,
    session,
    getObjectDetectionPresetOptions("PP-DocBlockLayout")
);
const boxes = await detector.run(image);

console.dir(
    {
        imagePath,
        model,
        detected: boxes.length,
        boxes: boxes.map((box) => ({
            label: box.label,
            score: Number(box.score.toFixed(4)),
            coordinate: box.coordinate.map((value) => Number(value.toFixed(1))),
        })),
    },
    { depth: null }
);
