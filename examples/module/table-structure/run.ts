import * as ort from "onnxruntime-node";
import {
    getTableStructureRecognitionPresetOptions,
    TableStructureRecognitionService,
} from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const preset = process.env.PADDLEOCR_TABLE_STRUCTURE_PRESET ?? "SLANet";
const tableStructureModels = {
    SLANet: modelPath("slanet", "SLANet_infer.onnx"),
    SLANeXt_wired: modelPath("slanext_wired", "SLANeXt_wired_infer.onnx"),
    SLANeXt_wireless: modelPath("slanext_wireless", "SLANeXt_wireless_infer.onnx"),
} as const;
const model = tableStructureModels[preset as keyof typeof tableStructureModels];
if (!model) {
    throw new Error(`Unsupported PADDLEOCR_TABLE_STRUCTURE_PRESET: ${preset}.`);
}

const imagePath =
    process.env.PADDLEOCR_TABLE_STRUCTURE_IMAGE ??
    (preset === "SLANeXt_wireless"
        ? "examples/input/table_wireless.png"
        : "examples/input/table_recognition.png");
const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the table structure model first."),
]);
const recognizer = new TableStructureRecognitionService(
    ort,
    session,
    getTableStructureRecognitionPresetOptions(preset as keyof typeof tableStructureModels)
);
const result = await recognizer.run(image);

console.dir(
    {
        imagePath,
        preset,
        model,
        structureLength: result.structure.length,
        bboxCount: result.bbox.length,
        htmlPreview: result.html.slice(0, 240),
    },
    { depth: null }
);
