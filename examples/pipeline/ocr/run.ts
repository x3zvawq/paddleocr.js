import * as ort from "onnxruntime-node";
import { type PaddleOcrModelPresetName, PaddleOcrService } from "../../../src/index.ts";
import {
    loadPngImage,
    loadPresetDictionary,
    modelPath,
    readRequiredFile,
    toArrayBuffer,
} from "../../_shared.ts";

const modelPreset: PaddleOcrModelPresetName =
    (process.env.PADDLEOCR_OCR_PRESET as PaddleOcrModelPresetName | undefined) ?? "PP-OCRv6_small";
const ocrAssets: Partial<
    Record<
        PaddleOcrModelPresetName,
        {
            dir: string;
            det: string;
            rec: string;
            dict: string;
        }
    >
> = {
    "PP-OCRv5_mobile": {
        dir: "ppocr_v5_mobile",
        det: "PP-OCRv5_mobile_det_infer.onnx",
        rec: "PP-OCRv5_mobile_rec_infer.onnx",
        dict: "ppocrv5_dict.txt",
    },
    "PP-OCRv6_tiny": {
        dir: "ppocr_v6_tiny",
        det: "PP-OCRv6_tiny_det_infer.onnx",
        rec: "PP-OCRv6_tiny_rec_infer.onnx",
        dict: "ppocrv6_tiny_dict.txt",
    },
    "PP-OCRv6_small": {
        dir: "ppocr_v6_small",
        det: "PP-OCRv6_small_det_infer.onnx",
        rec: "PP-OCRv6_small_rec_infer.onnx",
        dict: "ppocrv6_dict.txt",
    },
};
const asset = ocrAssets[modelPreset];
if (!asset) {
    throw new Error(`The OCR example has no asset mapping for ${modelPreset}.`);
}
const detectionModel = modelPath(asset.dir, asset.det);
const recognitionModel = modelPath(asset.dir, asset.rec);
const dictionaryPath = modelPath(asset.dir, asset.dict);
const textlineOrientationModel = modelPath(
    "pp_lcnet_x0_25_textline_ori",
    "PP-LCNet_x0_25_textline_ori_infer.onnx"
);
const imagePath = "examples/input/street_net_bar.png";

const [
    image,
    detectionModelBuffer,
    recognitionModelBuffer,
    textlineOrientationModelBuffer,
    charactersDictionary,
] = await Promise.all([
    loadPngImage(imagePath),
    readRequiredFile(detectionModel, `Download the ${modelPreset} detection model first.`),
    readRequiredFile(recognitionModel, `Download the ${modelPreset} recognition model first.`),
    readRequiredFile(textlineOrientationModel, "Download the textline orientation model first."),
    loadPresetDictionary(dictionaryPath, modelPreset),
]);
const ocr = await PaddleOcrService.createInstance({
    ort,
    modelPreset,
    detection: {
        modelBuffer: toArrayBuffer(detectionModelBuffer),
    },
    recognition: {
        modelBuffer: toArrayBuffer(recognitionModelBuffer),
        charactersDictionary,
    },
    textlineOrientation: {
        modelBuffer: toArrayBuffer(textlineOrientationModelBuffer),
        threshold: 0.9,
    },
});
const results = await ocr.recognize(image);
const processed = ocr.processRecognition(results);

console.dir(
    {
        imagePath,
        modelPreset,
        detected: results.length,
        confidence: Number(processed.confidence.toFixed(4)),
        textPreview: processed.text.slice(0, 500),
        firstResults: results.slice(0, 8).map((result) => ({
            text: result.text,
            confidence: Number(result.confidence.toFixed(4)),
            textlineOrientation: result.textlineOrientation,
        })),
    },
    { depth: null }
);
