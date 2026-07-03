import * as ort from "onnxruntime-node";
import {
    getObjectDetectionPresetOptions,
    type ObjectDetectionPresetName,
    ObjectDetectionService,
} from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/layout.png";
const layoutPreset = (process.env.PADDLEOCR_LAYOUT_PRESET ??
    "PP-DocLayout_plus-L") as ObjectDetectionPresetName;
const layoutModels: Partial<Record<ObjectDetectionPresetName, string>> = {
    "PP-DocLayout_plus-L": modelPath("pp_doclayout_plus_l", "PP-DocLayout_plus-L_infer.onnx"),
    "PP-DocLayout-L": modelPath("pp_doclayout_l", "PP-DocLayout-L_infer.onnx"),
    "PP-DocLayout-M": modelPath("pp_doclayout_m", "PP-DocLayout-M_infer.onnx"),
    "PP-DocLayout-S": modelPath("pp_doclayout_s", "PP-DocLayout-S_infer.onnx"),
};
const model = layoutModels[layoutPreset];
if (!model) {
    throw new Error(`Unsupported PADDLEOCR_LAYOUT_PRESET: ${layoutPreset}.`);
}

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, `Download the ${layoutPreset} model first.`),
]);
const detector = new ObjectDetectionService(
    ort,
    session,
    getObjectDetectionPresetOptions(layoutPreset)
);
const boxes = await detector.run(image);

console.dir(
    {
        imagePath,
        layoutPreset,
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
