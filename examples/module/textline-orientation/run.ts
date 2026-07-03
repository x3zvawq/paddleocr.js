import * as ort from "onnxruntime-node";
import {
    getImageClassificationPresetOptions,
    ImageClassificationService,
} from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/textline_rot180_demo.png";
const model = modelPath("pp_lcnet_x0_25_textline_ori", "PP-LCNet_x0_25_textline_ori_infer.onnx");

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the textline orientation model first."),
]);
const classifier = new ImageClassificationService(
    ort,
    session,
    getImageClassificationPresetOptions("PP-LCNet_x0_25_textline_ori")
);

console.dir(
    {
        imagePath,
        model,
        results: await classifier.run(image),
    },
    { depth: null }
);
