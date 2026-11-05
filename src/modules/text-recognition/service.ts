import type { Image } from "../../core/image.ts";
import { createInputFeeds, getFixedInputShape } from "../../core/onnx.ts";
import type {
    Box,
    OcrProgress,
    OrtInferenceSession,
    OrtModule,
    OrtTensor,
    RecognitionOptions,
    RecognitionResult,
    RecognitionRuntimeOptions,
    TextLineOrientationClassifier,
    TextLineOrientationResult,
    TextLineOrientationRuntimeOptions,
} from "../../interface.ts";

import {
    type ResolvedRecognitionOptions,
    type ResolvedRecognitionOrderingOptions,
    resolveRecognitionOptions,
    resolveRecognitionOrderingOptions,
} from "./options.ts";
import { createCtcDecoder } from "./postprocess.ts";
import { preprocessRecognition } from "./preprocess.ts";

export type { RecognitionResult } from "../../types/text-recognition.ts";

export interface SingleRecognitionTask {
    index: number;
    image: Image;
    box: Box;
    maxWhRatio?: number;
    charWhiteSet?: Set<string>;
    textlineOrientation?: Partial<TextLineOrientationRuntimeOptions>;
    textlineOrientationClassifier?: TextLineOrientationClassifier;
}

/**
 * Service for detecting and recognizing text in images
 */
export class RecognitionService {
    private readonly options: RecognitionRuntimeOptions;
    private readonly session: OrtInferenceSession;
    private readonly ortModule: OrtModule;

    constructor(
        ortModule: OrtModule,
        session: OrtInferenceSession,
        options: Partial<RecognitionRuntimeOptions> = {}
    ) {
        this.session = session;
        this.ortModule = ortModule;

        this.options = { ...options };
    }

    /**
     * Main method to run text recognition on an image with detected regions
     * @param image The original image buffer or image in Canvas
     * @param detection Array of bounding boxes from text detection
     * @returns Array of recognition results with text and bounding box, sorted in reading order
     */
    async run(
        image: Image,
        detection: Box[],
        options?: RecognitionOptions
    ): Promise<RecognitionResult[]> {
        const recognitionOptions = resolveRecognitionOptions(this.options, options?.recognition);
        const orderingOptions = resolveRecognitionOrderingOptions(options?.ordering);
        const validBoxes = this.sortBoxesByReadingOrder(
            detection.filter((box) => box.width > 0 && box.height > 0),
            orderingOptions
        );
        const maxWhRatio = this.calculateBatchMaxWhRatio(validBoxes, recognitionOptions);
        const results: RecognitionResult[] = [];
        const charWhiteListSet = options?.charWhiteList?.length
            ? new Set(options.charWhiteList)
            : undefined;
        const decode = createCtcDecoder(recognitionOptions, charWhiteListSet);
        const total = validBoxes.length;
        const onProgress = options?.onProgress;

        onProgress?.({
            type: "rec",
            stage: "start",
            progress: this.createProgress(0, total),
        });

        for (const [i, box] of validBoxes.entries()) {
            const result = await this.processBox(
                {
                    image: image,
                    index: i,
                    box: box,
                    maxWhRatio,
                    charWhiteSet: charWhiteListSet,
                    textlineOrientation: options?.textlineOrientation,
                    textlineOrientationClassifier: options?.textlineOrientationClassifier,
                },
                recognitionOptions,
                decode
            );
            results.push(result);
            onProgress?.({
                type: "rec",
                stage: "item",
                progress: this.createProgress(i + 1, total),
                index: i,
                box,
                result,
                textlineOrientation: result.textlineOrientation,
            });
        }

        onProgress?.({
            type: "rec",
            stage: "complete",
            progress: this.createProgress(total, total),
        });

        return results;
    }

    /**
     * Process a single text box
     */
    private async processBox(
        task: SingleRecognitionTask,
        runtimeOptions: ResolvedRecognitionOptions,
        decode: ReturnType<typeof createCtcDecoder>
    ): Promise<RecognitionResult> {
        const { image, box } = task;

        let crop = box.points ? image.cropRotated(box.points) : image.crop(box);
        const textlineOrientation = await this.correctTextlineOrientation(crop, task);
        if (textlineOrientation?.rotated) {
            crop = crop.rotate180();
        }
        const preprocessed = preprocessRecognition(
            crop,
            runtimeOptions,
            task.maxWhRatio,
            getFixedInputShape(this.session).width
        );
        const inputTensor = new this.ortModule.Tensor(
            "float32",
            preprocessed.data,
            preprocessed.dims
        );
        const output = await this.runInference(inputTensor, runtimeOptions);
        const { text: recognizedText, confidence } = decode(output);

        return { text: recognizedText, box, confidence, textlineOrientation };
    }

    private async correctTextlineOrientation(
        crop: Image,
        task: SingleRecognitionTask
    ): Promise<TextLineOrientationResult | undefined> {
        const classifier = task.textlineOrientationClassifier;
        if (!classifier) {
            return undefined;
        }

        const runtimeOptions = {
            enabled: true,
            threshold: 0.9,
            ...task.textlineOrientation,
        };
        if (!runtimeOptions.enabled) {
            return undefined;
        }

        const results = await classifier.run(crop, runtimeOptions);
        const topResult = results[0];
        if (!topResult) {
            return undefined;
        }

        return {
            classId: topResult.classId,
            label: topResult.label,
            score: topResult.score,
            rotated: topResult.label.includes("180") && topResult.score > runtimeOptions.threshold,
        };
    }

    private calculateBatchMaxWhRatio(
        boxes: Box[],
        runtimeOptions: ResolvedRecognitionOptions
    ): number {
        let maxWhRatio = runtimeOptions.imageWidth / runtimeOptions.imageHeight;
        for (const box of boxes) {
            maxWhRatio = Math.max(maxWhRatio, this.calculateBoxWhRatio(box));
        }
        return maxWhRatio;
    }

    private calculateBoxWhRatio(box: Box): number {
        if (!box.points) {
            return box.width / box.height;
        }

        const cropWidth = Math.max(
            this.distance(box.points[0], box.points[1]),
            this.distance(box.points[2], box.points[3]),
            1
        );
        const cropHeight = Math.max(
            this.distance(box.points[0], box.points[3]),
            this.distance(box.points[1], box.points[2]),
            1
        );

        if (cropHeight / cropWidth >= 1.5) {
            return cropHeight / cropWidth;
        }
        return cropWidth / cropHeight;
    }

    private distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    /**
     * Sort recognition results by reading order (top to bottom, left to right)
     */
    private sortBoxesByReadingOrder(
        boxes: Box[],
        orderingOptions: ResolvedRecognitionOrderingOptions
    ): Box[] {
        if (!orderingOptions.sortByReadingOrder) {
            return [...boxes];
        }

        const sortedBoxes = [...boxes].sort((boxA, boxB) => {
            const pointA = this.getBoxTopLeft(boxA);
            const pointB = this.getBoxTopLeft(boxB);
            if (pointA.y !== pointB.y) {
                return pointA.y - pointB.y;
            }
            return pointA.x - pointB.x;
        });

        for (let i = 0; i < sortedBoxes.length - 1; i++) {
            for (let j = i; j >= 0; j--) {
                const current = sortedBoxes[j];
                const next = sortedBoxes[j + 1];
                if (!current || !next) {
                    break;
                }

                const currentPoint = this.getBoxTopLeft(current);
                const nextPoint = this.getBoxTopLeft(next);
                const sameLineThreshold = this.resolveSameLineThreshold(
                    current,
                    next,
                    orderingOptions
                );
                if (
                    Math.abs(nextPoint.y - currentPoint.y) < sameLineThreshold &&
                    nextPoint.x < currentPoint.x
                ) {
                    sortedBoxes[j] = next;
                    sortedBoxes[j + 1] = current;
                    continue;
                }
                break;
            }
        }

        return sortedBoxes;
    }

    private getBoxTopLeft(box: Box) {
        return box.points?.[0] ?? box;
    }

    private resolveSameLineThreshold(
        boxA: Box,
        boxB: Box,
        orderingOptions: ResolvedRecognitionOrderingOptions
    ): number {
        if (orderingOptions.sameLineThresholdRatio !== undefined) {
            return (boxA.height + boxB.height) * orderingOptions.sameLineThresholdRatio;
        }

        return orderingOptions.sameLinePixelThreshold;
    }

    private createProgress(current: number, total: number): OcrProgress {
        return {
            current,
            remain: total - current,
            total,
        };
    }

    /**
     * Runs the ONNX inference session with the prepared tensor
     */
    private async runInference(
        inputTensor: OrtTensor,
        runtimeOptions: ResolvedRecognitionOptions
    ): Promise<OrtTensor> {
        const results = await this.session.run(createInputFeeds(this.session, inputTensor));
        const outputTensor = this.selectOutputTensor(results, runtimeOptions);

        if (!outputTensor) {
            throw new Error(
                `Recognition output tensor not found. Available keys: ${Object.keys(results).join(", ")}`
            );
        }

        return outputTensor;
    }

    private selectOutputTensor(
        results: Record<string, OrtTensor>,
        runtimeOptions: ResolvedRecognitionOptions
    ): OrtTensor | undefined {
        if (runtimeOptions.outputSelectionStrategy !== "ctc-logits") {
            const outputNodeName = Object.keys(results)[0];
            return outputNodeName ? results[outputNodeName] : undefined;
        }

        let ctcOutputTensor: OrtTensor | undefined;
        for (const outputName of Object.keys(results)) {
            const outputTensor = results[outputName];
            const dims = outputTensor?.dims;
            if (
                outputTensor?.data instanceof Float32Array &&
                dims?.length === 3 &&
                (dims[0] === 1 || dims[0] === -1) &&
                (dims[1] ?? 0) > 0 &&
                (dims[2] ?? 0) > 1
            ) {
                ctcOutputTensor = outputTensor;
            }
        }
        if (ctcOutputTensor) {
            return ctcOutputTensor;
        }

        throw new Error(
            `Recognition CTC logits output not found. Available outputs: ${Object.entries(results)
                .map(([name, tensor]) => `${name}[${tensor.dims.join(",")}]`)
                .join(", ")}`
        );
    }
}
