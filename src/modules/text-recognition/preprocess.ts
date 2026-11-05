import type { Image } from "../../core/image.ts";
import type { ResolvedRecognitionOptions } from "./options.ts";

export function preprocessRecognition(
    crop: Image,
    options: ResolvedRecognitionOptions,
    maxWhRatio = 0,
    fixedInputWidth?: number
): { data: Float32Array; dims: readonly [number, number, number, number] } {
    const proportionalWidth = Math.ceil(options.imageHeight * (crop.width / crop.height));
    const targetWidth =
        fixedInputWidth ??
        Math.max(
            options.imageWidth,
            proportionalWidth,
            Math.ceil(options.imageHeight * maxWhRatio)
        );
    const resizedWidth = Math.min(proportionalWidth, targetWidth);
    const resized = crop.resize({ width: resizedWidth, height: options.imageHeight });
    const tensor = resized.tensor({
        mean_values: options.mean,
        norm_values: options.stdDeviation,
        channel_order: options.channelOrder,
    });
    return {
        data: padRecognitionTensor(tensor, resizedWidth, options.imageHeight, targetWidth),
        dims: [1, 3, options.imageHeight, targetWidth],
    };
}

function padRecognitionTensor(
    source: Float32Array,
    sourceWidth: number,
    height: number,
    targetWidth: number
): Float32Array {
    if (sourceWidth === targetWidth) {
        return source;
    }

    const channels = 3;
    const padded = new Float32Array(channels * height * targetWidth);
    for (let channel = 0; channel < channels; channel++) {
        for (let y = 0; y < height; y++) {
            const sourceStart = channel * height * sourceWidth + y * sourceWidth;
            const targetStart = channel * height * targetWidth + y * targetWidth;
            padded.set(source.subarray(sourceStart, sourceStart + sourceWidth), targetStart);
        }
    }
    return padded;
}
