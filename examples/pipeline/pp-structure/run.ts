import * as ort from "onnxruntime-node";
import {
    createFormulaTokenizerVocabulary,
    type ObjectDetectionPresetName,
    type PaddleOcrModelPresetName,
    PaddleStructureService,
} from "../../../src/index.ts";
import {
    formulaTokenizerPath,
    loadJson,
    loadPngImage,
    loadPresetDictionary,
    modelPath,
    readRequiredFile,
    toArrayBuffer,
} from "../../_shared.ts";

const modelPreset: PaddleOcrModelPresetName = "PP-OCRv6_small";
const layoutPreset = (process.env.PADDLEOCR_STRUCTURE_LAYOUT_PRESET ??
    "PP-DocLayout_plus-L") as ObjectDetectionPresetName;
const layoutModels: Partial<Record<ObjectDetectionPresetName, string>> = {
    "PP-DocLayout_plus-L": modelPath("pp_doclayout_plus_l", "PP-DocLayout_plus-L_infer.onnx"),
    "PP-DocLayout-L": modelPath("pp_doclayout_l", "PP-DocLayout-L_infer.onnx"),
    "PP-DocLayout-M": modelPath("pp_doclayout_m", "PP-DocLayout-M_infer.onnx"),
    "PP-DocLayout-S": modelPath("pp_doclayout_s", "PP-DocLayout-S_infer.onnx"),
};
const layoutModel = layoutModels[layoutPreset];
if (!layoutModel) {
    throw new Error(`Unsupported semantic layout preset for this example: ${layoutPreset}.`);
}

const imagePath = "examples/input/layout.png";
const detectionModel = modelPath("ppocr_v6_small", "PP-OCRv6_small_det_infer.onnx");
const recognitionModel = modelPath("ppocr_v6_small", "PP-OCRv6_small_rec_infer.onnx");
const textlineOrientationModel = modelPath(
    "pp_lcnet_x0_25_textline_ori",
    "PP-LCNet_x0_25_textline_ori_infer.onnx"
);
const dictionaryPath = modelPath("ppocr_v6_small", "ppocrv6_dict.txt");

const [
    image,
    documentOrientationModel,
    unwarpingModel,
    regionDetectionModel,
    layoutModelBuffer,
    detectionModelBuffer,
    recognitionModelBuffer,
    textlineOrientationModelBuffer,
    charactersDictionary,
    tableStructureModel,
    formulaModel,
    tokenizerJson,
] = await Promise.all([
    loadPngImage(imagePath),
    readRequiredFile(
        modelPath("pp_lcnet_x1_0_doc_ori", "PP-LCNet_x1_0_doc_ori_infer.onnx"),
        "Download the document orientation model first."
    ),
    readRequiredFile(modelPath("uvdoc", "UVDoc_infer.onnx"), "Download the UVDoc model first."),
    readRequiredFile(
        modelPath("pp_docblocklayout", "PP-DocBlockLayout_infer.onnx"),
        "Download the PP-DocBlockLayout region model first."
    ),
    readRequiredFile(layoutModel, `Download the ${layoutPreset} layout model first.`),
    readRequiredFile(detectionModel, "Download the PP-OCRv6 detection model first."),
    readRequiredFile(recognitionModel, "Download the PP-OCRv6 recognition model first."),
    readRequiredFile(textlineOrientationModel, "Download the textline orientation model first."),
    loadPresetDictionary(dictionaryPath, modelPreset),
    readRequiredFile(modelPath("slanet", "SLANet_infer.onnx"), "Download SLANet first."),
    readRequiredFile(
        modelPath("pp_formulanet_plus_m", "PP-FormulaNet_plus-M_infer.onnx"),
        "Download PP-FormulaNet_plus-M first."
    ),
    loadJson(formulaTokenizerPath, "Prepare the official UniMERNet tokenizer JSON first."),
]);

const structure = await PaddleStructureService.createInstance({
    ort,
    documentOrientation: {
        modelBuffer: toArrayBuffer(documentOrientationModel),
    },
    textImageUnwarping: {
        modelBuffer: toArrayBuffer(unwarpingModel),
    },
    regionDetection: {
        modelBuffer: toArrayBuffer(regionDetectionModel),
    },
    layout: {
        modelBuffer: toArrayBuffer(layoutModelBuffer),
        preset: layoutPreset,
    },
    ocr: {
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
    },
    tableStructure: {
        modelBuffer: toArrayBuffer(tableStructureModel),
    },
    formulaRecognition: {
        modelBuffer: toArrayBuffer(formulaModel),
        tokenizerVocabulary: createFormulaTokenizerVocabulary(tokenizerJson),
    },
});
const result = await structure.run(image, {
    layout: {
        fallbackRegionType: false,
    },
});

console.dir(
    {
        imagePath,
        layoutPreset,
        finalImage: { width: result.image.width, height: result.image.height },
        stages: {
            documentOrientation: result.stages.documentOrientation.status,
            textImageUnwarping: result.stages.textImageUnwarping.status,
            regionDetection: result.stages.regionDetection.status,
            layout: result.stages.layout.status,
            ocr: result.stages.ocr.status,
        },
        regionDetections: result.regionDetections.map((region) => ({
            type: region.type,
            score: Number(region.score.toFixed(4)),
            bbox: region.bbox,
        })),
        regions: result.regions.map((region) => ({
            type: region.type,
            score: Number(region.score.toFixed(4)),
            bbox: region.bbox,
            status: region.status,
            ocrPreview: region.ocr
                ?.map((item) => item.text)
                .join(" ")
                .slice(0, 160),
            tableHtmlPreview:
                region.table?.matched?.html.slice(0, 160) ??
                region.table?.structure?.html.slice(0, 160),
            formulaPreview: region.formula?.formula.slice(0, 160),
        })),
    },
    { depth: null }
);
