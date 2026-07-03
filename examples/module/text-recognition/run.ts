import * as ort from "onnxruntime-node";
import {
    type Box,
    getTextRecognitionPresetOptions,
    type PaddleOcrModelPresetName,
    RecognitionService,
} from "../../../src/index.ts";
import { createSession, loadPngImage, loadPresetDictionary, modelPath } from "../../_shared.ts";

const modelPreset: PaddleOcrModelPresetName = "PP-OCRv6_small";
const imagePath = "examples/input/general_ocr_rec_001.png";
const model = modelPath("ppocr_v6_small", "PP-OCRv6_small_rec_infer.onnx");
const dictionaryPath = modelPath("ppocr_v6_small", "ppocrv6_dict.txt");

const [image, session, charactersDictionary] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, "Download the PP-OCRv6 small recognition model first."),
    loadPresetDictionary(dictionaryPath, modelPreset),
]);
const recognizer = new RecognitionService(ort, session, {
    ...getTextRecognitionPresetOptions("PP-OCRv6_small_rec"),
    charactersDictionary,
});
const fullImageBox: Box = { x: 0, y: 0, width: image.width, height: image.height };

console.dir(
    {
        imagePath,
        model,
        results: await recognizer.run(image, [fullImageBox]),
    },
    { depth: null }
);
