import {
    DEFAULT_RECOGNITION_OPTIONS,
    DEFAULT_RECOGNITION_ORDERING_OPTIONS,
} from "../../constants.ts";
import type {
    RecognitionOrderingOptions,
    RecognitionRuntimeOptions,
} from "../../types/text-recognition.ts";

export type ResolvedRecognitionOptions = Required<RecognitionRuntimeOptions>;
export type ResolvedRecognitionOrderingOptions = Required<
    Pick<RecognitionOrderingOptions, "sortByReadingOrder" | "sameLinePixelThreshold">
> &
    Pick<RecognitionOrderingOptions, "sameLineThresholdRatio">;

export function resolveRecognitionOptions(
    options: RecognitionRuntimeOptions,
    overrides: RecognitionRuntimeOptions = {}
): ResolvedRecognitionOptions {
    const resolved = { ...DEFAULT_RECOGNITION_OPTIONS, ...options, ...overrides };
    for (const name of ["imageHeight", "imageWidth"] as const) {
        if (!Number.isInteger(resolved[name]) || resolved[name] <= 0) {
            throw new Error(
                `Invalid recognition ${name}: ${resolved[name]}. Expected a positive integer.`
            );
        }
    }
    for (const name of ["mean", "stdDeviation"] as const) {
        const values = resolved[name];
        if (values?.length !== 3 || values.some((value) => !Number.isFinite(value))) {
            throw new Error(`Invalid recognition ${name}. Expected three finite numbers.`);
        }
    }
    return resolved;
}

export function resolveRecognitionOrderingOptions(
    options: RecognitionOrderingOptions = {}
): ResolvedRecognitionOrderingOptions {
    return { ...DEFAULT_RECOGNITION_ORDERING_OPTIONS, ...options };
}
