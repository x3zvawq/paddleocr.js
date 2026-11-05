import type {
    DetectionRuntimeOptions,
    PaddleOcrModelPresetName,
    RecognitionRuntimeOptions,
} from "../interface.ts";

import { getTextDetectionPresetOptions } from "../modules/text-detection/preset.ts";
import { getTextRecognitionPreset } from "../modules/text-recognition/preset.ts";
import type {
    PaddleOcrDictionaryRequirement,
    TextRecognitionPresetName,
} from "../types/text-recognition.ts";

export type { PaddleOcrDictionaryRequirement } from "../types/text-recognition.ts";

export interface PaddleOcrModelPreset {
    name: PaddleOcrModelPresetName;
    detection: Partial<DetectionRuntimeOptions>;
    recognition: Partial<RecognitionRuntimeOptions>;
    dictionary: PaddleOcrDictionaryRequirement;
}

export interface ModelPresetInferenceInput {
    modelName?: string;
    fileName?: string;
    detectionModelFileName?: string;
    recognitionModelFileName?: string;
    dictionaryName?: string;
    dictionaryFileName?: string;
    recognitionOutputClasses?: number;
    recognitionOutputShape?: readonly (number | string | null | undefined)[];
    dictionaryLength?: number;
}

export interface ModelPresetInferenceResult {
    name: PaddleOcrModelPresetName;
    confidence: "high" | "medium";
    signals: string[];
}

const PPOCRV5_DETECTION: Partial<DetectionRuntimeOptions> = {
    channelOrder: "bgr",
    maxSideLength: 960,
    limitType: "max",
    maxSideLimit: 4000,
    textPixelThreshold: 0.3,
    boxScoreThreshold: 0.6,
    maxCandidates: 1000,
    unclipRatio: 1.5,
};

function createModelPreset(
    name: PaddleOcrModelPresetName,
    detection: Partial<DetectionRuntimeOptions>,
    recognitionName: TextRecognitionPresetName
): PaddleOcrModelPreset {
    const recognition = getTextRecognitionPreset(recognitionName);
    return {
        name,
        detection,
        recognition: { ...recognition.options },
        dictionary: recognition.dictionary,
    };
}

const v5 = getTextRecognitionPreset("PP-OCRv5_mobile_rec").dictionary;
const v6 = getTextRecognitionPreset("PP-OCRv6_small_rec").dictionary;
const v6Tiny = getTextRecognitionPreset("PP-OCRv6_tiny_rec").dictionary;
const PPOCRV5_RECOGNITION_OUTPUT_CLASSES = v5.recognitionOutputClasses;
const PPOCRV5_DICTIONARY_LENGTH = v5.dictionaryLength;
const PPOCRV6_RECOGNITION_OUTPUT_CLASSES = v6.recognitionOutputClasses;
const PPOCRV6_DICTIONARY_LENGTH = v6.dictionaryLength;
const PPOCRV6_TINY_RECOGNITION_OUTPUT_CLASSES = v6Tiny.recognitionOutputClasses;
const PPOCRV6_TINY_DICTIONARY_LENGTH = v6Tiny.dictionaryLength;

export const MODEL_PRESETS: Record<PaddleOcrModelPresetName, PaddleOcrModelPreset> = {
    "PP-OCRv5": createModelPreset("PP-OCRv5", PPOCRV5_DETECTION, "PP-OCRv5_mobile_rec"),
    "PP-OCRv5_mobile": createModelPreset(
        "PP-OCRv5_mobile",
        PPOCRV5_DETECTION,
        "PP-OCRv5_mobile_rec"
    ),
    "PP-OCRv5_server": createModelPreset(
        "PP-OCRv5_server",
        PPOCRV5_DETECTION,
        "PP-OCRv5_server_rec"
    ),
    "PP-OCRv6": createModelPreset(
        "PP-OCRv6",
        getTextDetectionPresetOptions("PP-OCRv6_small_det"),
        "PP-OCRv6_small_rec"
    ),
    "PP-OCRv6_tiny": createModelPreset(
        "PP-OCRv6_tiny",
        getTextDetectionPresetOptions("PP-OCRv6_tiny_det"),
        "PP-OCRv6_tiny_rec"
    ),
    "PP-OCRv6_small": createModelPreset(
        "PP-OCRv6_small",
        getTextDetectionPresetOptions("PP-OCRv6_small_det"),
        "PP-OCRv6_small_rec"
    ),
    "PP-OCRv6_medium": createModelPreset(
        "PP-OCRv6_medium",
        getTextDetectionPresetOptions("PP-OCRv6_medium_det"),
        "PP-OCRv6_medium_rec"
    ),
};

export function getModelPreset(name: PaddleOcrModelPresetName): PaddleOcrModelPreset {
    const preset = MODEL_PRESETS[name];
    if (!preset) {
        throw new Error(`Unsupported PaddleOCR model preset: ${name}`);
    }
    return preset;
}

export function getModelPresetOptions(
    name?: PaddleOcrModelPresetName
): Pick<PaddleOcrModelPreset, "detection" | "recognition"> {
    if (!name) {
        return {
            detection: {},
            recognition: {},
        };
    }
    const preset = getModelPreset(name);
    return {
        detection: { ...preset.detection },
        recognition: { ...preset.recognition },
    };
}

export function inferModelPreset(
    input: ModelPresetInferenceInput
): ModelPresetInferenceResult | undefined {
    const signals = collectInferenceSignals(input);
    const normalized = signals.map((signal) => normalizeInferenceSignal(signal));

    if (normalized.some((signal) => signal.includes("ppocrv6tiny"))) {
        return createInferenceResult("PP-OCRv6_tiny", "high", signals);
    }
    if (normalized.some((signal) => signal.includes("ppocrv6small"))) {
        return createInferenceResult("PP-OCRv6_small", "high", signals);
    }
    if (normalized.some((signal) => signal.includes("ppocrv6medium"))) {
        return createInferenceResult("PP-OCRv6_medium", "high", signals);
    }
    if (normalized.some((signal) => signal.includes("ppocrv6"))) {
        return createInferenceResult("PP-OCRv6", "medium", signals);
    }
    if (normalized.some((signal) => signal.includes("ppocrv5server"))) {
        return createInferenceResult("PP-OCRv5_server", "high", signals);
    }
    if (normalized.some((signal) => signal.includes("ppocrv5mobile"))) {
        return createInferenceResult("PP-OCRv5_mobile", "high", signals);
    }
    if (normalized.some((signal) => signal.includes("ppocrv5"))) {
        return createInferenceResult("PP-OCRv5", "medium", signals);
    }

    return inferModelPresetFromMetadata(input);
}

function inferModelPresetFromMetadata(
    input: ModelPresetInferenceInput
): ModelPresetInferenceResult | undefined {
    const signals = new Set<string>();
    const recognitionOutputClasses =
        input.recognitionOutputClasses ??
        getLastNumericShapeDimension(input.recognitionOutputShape);

    if (recognitionOutputClasses === PPOCRV6_TINY_RECOGNITION_OUTPUT_CLASSES) {
        signals.add(`recognitionOutputClasses:${PPOCRV6_TINY_RECOGNITION_OUTPUT_CLASSES}`);
    }
    if (input.dictionaryLength === PPOCRV6_TINY_DICTIONARY_LENGTH) {
        signals.add(`dictionaryLength:${PPOCRV6_TINY_DICTIONARY_LENGTH}`);
    }
    if (signals.size > 0) {
        return createInferenceResult("PP-OCRv6_tiny", "high", Array.from(signals));
    }

    if (recognitionOutputClasses === PPOCRV6_RECOGNITION_OUTPUT_CLASSES) {
        signals.add(`recognitionOutputClasses:${PPOCRV6_RECOGNITION_OUTPUT_CLASSES}`);
    }
    if (input.dictionaryLength === PPOCRV6_DICTIONARY_LENGTH) {
        signals.add(`dictionaryLength:${PPOCRV6_DICTIONARY_LENGTH}`);
    }
    if (signals.size > 0) {
        return createInferenceResult("PP-OCRv6_small", "medium", Array.from(signals));
    }

    if (recognitionOutputClasses === PPOCRV5_RECOGNITION_OUTPUT_CLASSES) {
        signals.add(`recognitionOutputClasses:${PPOCRV5_RECOGNITION_OUTPUT_CLASSES}`);
    }
    if (input.dictionaryLength === PPOCRV5_DICTIONARY_LENGTH) {
        signals.add(`dictionaryLength:${PPOCRV5_DICTIONARY_LENGTH}`);
    }

    if (signals.size > 0) {
        return createInferenceResult("PP-OCRv5", "medium", Array.from(signals));
    }
    return undefined;
}

function collectInferenceSignals(input: ModelPresetInferenceInput): string[] {
    return [
        input.modelName,
        input.fileName,
        input.detectionModelFileName,
        input.recognitionModelFileName,
        input.dictionaryName,
        input.dictionaryFileName,
    ].filter((signal): signal is string => Boolean(signal));
}

function getLastNumericShapeDimension(
    shape?: readonly (number | string | null | undefined)[]
): number | undefined {
    if (!shape) {
        return undefined;
    }
    for (let index = shape.length - 1; index >= 0; index -= 1) {
        const dimension = shape[index];
        if (typeof dimension === "number" && Number.isFinite(dimension) && dimension > 0) {
            return dimension;
        }
    }
    return undefined;
}

function normalizeInferenceSignal(signal: string): string {
    return signal.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function createInferenceResult(
    name: PaddleOcrModelPresetName,
    confidence: ModelPresetInferenceResult["confidence"],
    signals: string[]
): ModelPresetInferenceResult {
    return {
        name,
        confidence,
        signals: [...signals],
    };
}
