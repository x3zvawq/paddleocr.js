import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { decode } from "fast-png";
import {
    getModelPreset,
    Image,
    type OrtInferenceSession,
    type OrtModule,
    type PaddleOcrModelPresetName,
} from "../src/index.ts";

export const modelRoot = process.env.PADDLEOCR_JS_ONNX_DIR ?? "paddleocr-js-onnx";

export const formulaTokenizerPath =
    process.env.PADDLEOCR_FORMULA_TOKENIZER_JSON ??
    "/Users/x3zvawq/workspace/PaddleOCR/ppocr/utils/dict/unimernet_tokenizer/tokenizer.json";

export function modelPath(...parts: string[]): string {
    return join(modelRoot, ...parts);
}

export function toArrayBuffer(buffer: Buffer): ArrayBuffer {
    return buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength
    ) as ArrayBuffer;
}

export async function readRequiredFile(path: string, message: string): Promise<Buffer> {
    try {
        return await readFile(path);
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") {
            throw new Error(`${message} Missing ${path}.`);
        }
        throw error;
    }
}

export async function loadPngImage(path: string): Promise<Image> {
    const imageFile = await readRequiredFile(path, "Prepare the example input image first.");
    const image = decode(toArrayBuffer(imageFile));
    const channels = image.channels ?? Math.round(image.data.length / (image.width * image.height));

    return new Image(image.width, image.height, channels, image.data as Uint8Array);
}

export async function loadJson(path: string, message: string): Promise<unknown> {
    const buffer = await readRequiredFile(path, message);
    return JSON.parse(buffer.toString("utf-8"));
}

export async function createSession(
    ort: OrtModule,
    path: string,
    message: string
): Promise<OrtInferenceSession> {
    const buffer = await readRequiredFile(path, message);
    return ort.InferenceSession.create(toArrayBuffer(buffer));
}

export async function loadPresetDictionary(
    path: string,
    presetName: PaddleOcrModelPresetName
): Promise<string[]> {
    const preset = getModelPreset(presetName);
    const text = await readRequiredFile(path, `Prepare the ${presetName} dictionary first.`).then(
        (buffer) => buffer.toString("utf-8")
    );
    const dictionary = text.split(/\r?\n/);
    if (dictionary[dictionary.length - 1] === "") {
        dictionary.pop();
    }

    if (dictionary.length !== preset.dictionary.dictionaryLength) {
        throw new Error(
            `${presetName} dictionary length mismatch: expected ${preset.dictionary.dictionaryLength}, got ${dictionary.length}.`
        );
    }

    return preset.dictionary.useSpaceChar ? [...dictionary, " "] : dictionary;
}
