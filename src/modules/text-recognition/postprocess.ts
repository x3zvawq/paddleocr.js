import type { OrtTensor } from "../../types/ort.ts";
import type { ResolvedRecognitionOptions } from "./options.ts";

type RecognitionDecodeOptions = Pick<
    ResolvedRecognitionOptions,
    "charactersDictionary" | "useSpaceChar" | "reverseText"
>;

/** Remove only the file terminator; leading CTC blank and literal spaces are meaningful. */
export function parseRecognitionDictionary(text: string): string[] {
    const entries = text.split(/\r?\n/);
    if (entries[entries.length - 1] === "") entries.pop();
    return entries;
}

function prepareDictionary(
    options: RecognitionDecodeOptions,
    numClasses: number
): readonly string[] {
    const supplied = options.charactersDictionary;
    if (!supplied?.length) throw new Error("Recognition charactersDictionary is required.");
    const includesBlank = supplied[0] === "" || supplied[0] === "blank";
    const requiredLength = numClasses - (includesBlank ? 0 : 1);
    const missingSpace =
        options.useSpaceChar &&
        supplied.length === requiredLength - 1 &&
        supplied[supplied.length - 1] !== " ";
    const entries = missingSpace ? [...supplied, " "] : supplied;
    if (entries.length !== requiredLength) {
        throw new Error(
            `Recognition charactersDictionary length ${supplied.length} does not match model output classes ${numClasses}. Expected ${requiredLength}${includesBlank ? " including the CTC blank entry" : " without the CTC blank entry"}.`
        );
    }
    if (entries.some((entry, index) => entry === "" && !(includesBlank && index === 0))) {
        throw new Error(
            "Recognition dictionary contains an empty character. Parse dictionary text with parseRecognitionDictionary() to remove its trailing newline."
        );
    }
    return includesBlank ? entries : ["", ...entries];
}

/** Each run prepares its class mapping once, after the first model output establishes C. */
export function createCtcDecoder(options: RecognitionDecodeOptions, charWhiteSet?: Set<string>) {
    let dictionary: readonly string[] | undefined;
    return (output: OrtTensor): { text: string; confidence: number } => {
        const { data: logits, dims } = output;
        if (
            !(logits instanceof Float32Array) ||
            dims.length !== 3 ||
            dims[0] !== 1 ||
            !Number.isInteger(dims[1]) ||
            dims[1] <= 0 ||
            !Number.isInteger(dims[2]) ||
            dims[2] < 2 ||
            logits.length !== dims[1] * dims[2]
        ) {
            throw new Error(
                `Recognition CTC output must be a float32 tensor shaped [1, T, C]; got [${dims.join(", ")}].`
            );
        }
        const [, sequenceLength, numClasses] = dims;
        dictionary ??= prepareDictionary(options, numClasses);
        if (dictionary.length !== numClasses)
            throw new Error("Recognition output class count changed within one run.");
        let text = "";
        let scoreSum = 0;
        let scoreCount = 0;
        let lastIndex = -1;
        for (let t = 0; t < sequenceLength; t++) {
            let maxScore = -Infinity;
            let maxIndex = 0;
            const offset = t * numClasses;
            for (let i = 0; i < numClasses; i++) {
                const score = logits[offset + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxIndex = i;
                }
            }
            if (maxIndex === lastIndex) continue;
            lastIndex = maxIndex;
            if (maxIndex === 0) continue;
            const char = dictionary[maxIndex];
            if (charWhiteSet && !charWhiteSet.has(char) && char !== " ") continue;
            text += char;
            scoreSum += maxScore;
            scoreCount++;
        }
        return {
            text: options.reverseText ? reverseTextLikePaddleOcr(text) : text,
            confidence: scoreCount > 0 ? scoreSum / scoreCount : 0,
        };
    };
}

function reverseTextLikePaddleOcr(text: string): string {
    const groups: string[] = [];
    let current = "";
    for (const char of text) {
        if (!/[a-zA-Z0-9 :*./%+-]/.test(char)) {
            if (current) {
                groups.push(current);
            }
            groups.push(char);
            current = "";
            continue;
        }
        current += char;
    }
    if (current) {
        groups.push(current);
    }
    return groups.reverse().join("");
}
