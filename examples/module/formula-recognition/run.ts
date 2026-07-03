import * as ort from "onnxruntime-node";
import {
    createFormulaTokenizerVocabulary,
    type FormulaRecognitionPresetName,
    FormulaRecognitionService,
    getFormulaRecognitionPresetOptions,
} from "../../../src/index.ts";
import {
    createSession,
    formulaTokenizerPath,
    loadJson,
    loadPngImage,
    modelPath,
} from "../../_shared.ts";

const preset = (process.env.PADDLEOCR_FORMULA_PRESET ??
    "PP-FormulaNet_plus-M") as FormulaRecognitionPresetName;
const formulaModels: Record<FormulaRecognitionPresetName, string> = {
    "PP-FormulaNet-S": modelPath("PP_FormulaNet_S", "PP-FormulaNet-S_infer.onnx"),
    "PP-FormulaNet-L": modelPath("PP_FormulaNet_L", "PP-FormulaNet-L_infer.onnx"),
    "PP-FormulaNet_plus-S": modelPath("PP_FormulaNet_plus_S", "PP-FormulaNet_plus-S_infer.onnx"),
    "PP-FormulaNet_plus-M": modelPath("pp_formulanet_plus_m", "PP-FormulaNet_plus-M_infer.onnx"),
    "PP-FormulaNet_plus-L": modelPath("PP_FormulaNet_plus_L", "PP-FormulaNet_plus-L_infer.onnx"),
};
const model = formulaModels[preset];
const imagePath = "examples/input/general_formula_rec_001.png";

const [image, session, tokenizerJson] = await Promise.all([
    loadPngImage(imagePath),
    createSession(ort, model, `Download the ${preset} formula model first.`),
    loadJson(formulaTokenizerPath, "Prepare the official UniMERNet tokenizer JSON first."),
]);
const recognizer = new FormulaRecognitionService(ort, session, {
    ...getFormulaRecognitionPresetOptions(preset),
    tokenizerVocabulary: createFormulaTokenizerVocabulary(tokenizerJson),
});
const result = await recognizer.run(image);

console.dir(
    {
        imagePath,
        preset,
        model,
        formulaPreview: result.formula.slice(0, 240),
        tokenCount: result.tokens.length,
    },
    { depth: null }
);
