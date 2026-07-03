import * as ort from "onnxruntime-node";
import {
    getImageClassificationPresetOptions,
    ImageClassificationService,
} from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/doc_test_rot90ccw.png";
const model = modelPath("pp_lcnet_x1_0_doc_ori", "PP-LCNet_x1_0_doc_ori_infer.onnx");

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the document orientation model first."),
]);
const classifier = new ImageClassificationService(
    ort,
    session,
    getImageClassificationPresetOptions("PP-LCNet_x1_0_doc_ori")
);

console.dir(
    {
        imagePath,
        model,
        results: await classifier.run(image),
    },
    { depth: null }
);
