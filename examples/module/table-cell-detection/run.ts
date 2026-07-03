import * as ort from "onnxruntime-node";
import { getObjectDetectionPresetOptions, ObjectDetectionService } from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/table_recognition.png";
const model = modelPath(
    "rt_detr_wired_table_cell_det",
    "RT-DETR-L_wired_table_cell_det_infer.onnx"
);

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the wired table cell detection model first."),
]);
const detector = new ObjectDetectionService(
    ort,
    session,
    getObjectDetectionPresetOptions("RT-DETR-L_wired_table_cell_det")
);
const boxes = await detector.run(image);

console.dir(
    {
        imagePath,
        model,
        detected: boxes.length,
        firstCells: boxes.slice(0, 8).map((box) => ({
            label: box.label,
            score: Number(box.score.toFixed(4)),
            coordinate: box.coordinate.map((value) => Number(value.toFixed(1))),
        })),
    },
    { depth: null }
);
