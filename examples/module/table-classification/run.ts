import * as ort from "onnxruntime-node";
import {
    getImageClassificationPresetOptions,
    ImageClassificationService,
} from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/table_wireless.png";
const model = modelPath("pp_lcnet_x1_0_table_cls", "PP-LCNet_x1_0_table_cls_infer.onnx");

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the table classification model first."),
]);
const classifier = new ImageClassificationService(
    ort,
    session,
    getImageClassificationPresetOptions("PP-LCNet_x1_0_table_cls")
);

console.dir(
    {
        imagePath,
        model,
        results: await classifier.run(image),
    },
    { depth: null }
);
