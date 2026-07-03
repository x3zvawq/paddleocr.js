import * as ort from "onnxruntime-node";
import {
    getTextImageUnwarpingPresetOptions,
    TextImageUnwarpingService,
} from "../../../src/index.ts";
import { createSession, loadPngImage, modelPath } from "../../_shared.ts";

const imagePath = "examples/input/distorted_document.png";
const model = modelPath("uvdoc", "UVDoc_infer.onnx");

const [image, session] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the UVDoc model first."),
]);
const unwarper = new TextImageUnwarpingService(
    ort,
    session,
    getTextImageUnwarpingPresetOptions("UVDoc")
);
const result = await unwarper.run(image);

console.dir(
    {
        imagePath,
        model,
        input: { width: image.width, height: image.height },
        doctrImage: { width: result.doctrImage.width, height: result.doctrImage.height },
    },
    { depth: null }
);
