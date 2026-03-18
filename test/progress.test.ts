import assert from "node:assert/strict";
import test from "node:test";
import {
    DEFAULT_DETECTION_OPTIONS,
    type DetectionRuntimeOptions,
    DetectionService,
    type OrtInferenceSession,
    type OrtModule,
    PaddleOcrService,
    type PreprocessDetectionResult,
    type RecognitionOptions,
    type RecognitionOrderingOptions,
    type RecognitionResult,
    type RecognitionRuntimeOptions,
    RecognitionService,
} from "../src/index.ts";

type DetectionServiceInternals = DetectionService & {
    applyPaddingToRect: (
        rect: { x: number; y: number; width: number; height: number },
        maxWidth: number,
        maxHeight: number,
        runtimeOptions: DetectionRuntimeOptions
    ) => { x: number; y: number; width: number; height: number };
    postprocessDetection: (
        detection: Float32Array,
        input: PreprocessDetectionResult,
        runtimeOptions: DetectionRuntimeOptions
    ) => Array<{
        x: number;
        y: number;
        width: number;
        height: number;
    }>;
    preprocessDetection: (
        image: Parameters<DetectionService["run"]>[0],
        runtimeOptions: DetectionRuntimeOptions
    ) => Promise<PreprocessDetectionResult>;
    runInference: (
        tensor: Float32Array,
        input: PreprocessDetectionResult["resizeParams"]
    ) => Promise<Float32Array | null>;
};

type RecognitionServiceInternals = RecognitionService & {
    processBox: (
        task: {
            box: { x: number; y: number; width: number; height: number };
            charWhiteSet?: Set<string>;
            image: Parameters<RecognitionService["run"]>[0];
            index: number;
        },
        runtimeOptions: RecognitionRuntimeOptions
    ) => Promise<RecognitionResult | null>;
    sortBoxesByReadingOrder: (
        boxes: Array<{ x: number; y: number; width: number; height: number }>,
        orderingOptions: RecognitionOrderingOptions
    ) => Array<{ x: number; y: number; width: number; height: number }>;
};

function createFakeInput(width = 1, height = 1) {
    return {
        width,
        height,
        data: new Uint8Array(width * height * 4).fill(255),
    };
}

test("PaddleOcrService.recognize merges instance defaults with per-call overrides and keeps calls isolated", async () => {
    const detections = [{ x: 0, y: 0, width: 1, height: 1 }];
    const results = [{ text: "A", box: detections[0], confidence: 0.99 }];
    const service = new PaddleOcrService({
        ort: {} as unknown as OrtModule,
        detection: {
            padding: 2,
            minimumAreaThreshold: 10,
            paddingBoxVertical: 0.4,
            paddingBoxHorizontal: 0.6,
        },
        recognition: {
            charactersDictionary: ["", "A"],
            imageHeight: 48,
        },
    });
    const events: unknown[] = [];
    const detectionCalls: Array<{ width: number; options: Record<string, unknown> }> = [];
    const recognitionCalls: RecognitionOptions[] = [];

    service.detectionService = {
        run: async (image, options = {}) => {
            detectionCalls.push({
                width: image.width,
                options: options as unknown as Record<string, unknown>,
            });
            options.onProgress?.({
                type: "det",
                stage: "preprocess",
                progress: { current: 1, remain: 2, total: 3 },
            });
            return detections;
        },
    } as unknown as DetectionService;

    service.recognitionService = {
        run: async (_image, detection, options) => {
            recognitionCalls.push(options ?? {});
            assert.deepEqual(detection, detections);
            options?.onProgress?.({
                type: "rec",
                stage: "complete",
                progress: { current: detection.length, remain: 0, total: detection.length },
            });
            return results;
        },
    } as unknown as RecognitionService;

    const first = await service.recognize(createFakeInput(), {
        charWhiteList: ["A"],
        detection: {
            padding: 0,
            minimumAreaThreshold: 25,
            paddingBoxVertical: 0,
            paddingBoxHorizontal: 0,
            dilationKernelSize: 3,
        },
        recognition: {
            charactersDictionary: ["", "B"],
            imageHeight: 64,
        },
        ordering: {
            sortByReadingOrder: false,
            sameLineThresholdRatio: 0.1,
        },
        onProgress: (event) => events.push(event),
    });
    const second = await service.recognize(createFakeInput(), {
        onProgress: (event) => events.push(event),
    });

    assert.deepEqual(first, results);
    assert.deepEqual(second, results);
    assert.equal(detectionCalls[0]?.width, 1);
    assert.equal(detectionCalls[1]?.width, 5);
    assert.deepEqual(detectionCalls[0]?.options, {
        ...DEFAULT_DETECTION_OPTIONS,
        padding: 0,
        minimumAreaThreshold: 25,
        paddingBoxVertical: 0,
        paddingBoxHorizontal: 0,
        dilationKernelSize: 3,
        onProgress: recognitionCalls[0]?.onProgress,
    });
    assert.deepEqual(detectionCalls[1]?.options, {
        ...DEFAULT_DETECTION_OPTIONS,
        padding: 2,
        minimumAreaThreshold: 10,
        paddingBoxVertical: 0.4,
        paddingBoxHorizontal: 0.6,
        onProgress: recognitionCalls[1]?.onProgress,
    });
    assert.deepEqual(recognitionCalls[0]?.recognition, {
        charactersDictionary: ["", "B"],
        imageHeight: 64,
        mean: [127.5, 127.5, 127.5],
        stdDeviation: [1 / 127.5, 1 / 127.5, 1 / 127.5],
    });
    assert.deepEqual(recognitionCalls[0]?.ordering, {
        sortByReadingOrder: false,
        sameLineThresholdRatio: 0.1,
    });
    assert.deepEqual(recognitionCalls[0]?.charWhiteList, ["A"]);
    assert.deepEqual(recognitionCalls[1]?.recognition, {
        charactersDictionary: ["", "A"],
        imageHeight: 48,
        mean: [127.5, 127.5, 127.5],
        stdDeviation: [1 / 127.5, 1 / 127.5, 1 / 127.5],
    });
    assert.deepEqual(recognitionCalls[1]?.ordering, {
        sortByReadingOrder: true,
        sameLineThresholdRatio: 0.25,
    });
    assert.deepEqual(events, [
        {
            type: "det",
            stage: "preprocess",
            progress: { current: 1, remain: 2, total: 3 },
        },
        {
            type: "rec",
            stage: "complete",
            progress: { current: 1, remain: 0, total: 1 },
        },
        {
            type: "det",
            stage: "preprocess",
            progress: { current: 1, remain: 2, total: 3 },
        },
        {
            type: "rec",
            stage: "complete",
            progress: { current: 1, remain: 0, total: 1 },
        },
    ]);
});

test("PaddleOcrService.recognize accepts per-call dictionaries and rejects missing dictionaries", async () => {
    const service = new PaddleOcrService({
        ort: {} as unknown as OrtModule,
    });

    service.detectionService = {
        run: async () => [{ x: 0, y: 0, width: 1, height: 1 }],
    } as unknown as DetectionService;

    service.recognitionService = {
        run: async (_image, _detection, options) => {
            assert.deepEqual(options?.recognition?.charactersDictionary, ["", "X"]);
            return [{ text: "X", box: { x: 0, y: 0, width: 1, height: 1 }, confidence: 1 }];
        },
    } as unknown as RecognitionService;

    const result = await service.recognize(createFakeInput(), {
        recognition: {
            charactersDictionary: ["", "X"],
        },
    });

    assert.equal(result[0]?.text, "X");
    await assert.rejects(
        () => service.recognize(createFakeInput()),
        /Recognition charactersDictionary is required/
    );
});

test("DetectionService emits progress and forwards merged runtime options including zero-valued overrides", async () => {
    const detectionService = new DetectionService(
        { Tensor: class Tensor {} } as unknown as OrtModule,
        { outputNames: [], run: async () => ({}) } as unknown as OrtInferenceSession,
        { minimumAreaThreshold: 20 }
    );
    const detectionServiceInternals = detectionService as unknown as DetectionServiceInternals;
    const events: unknown[] = [];
    const seenOptions: DetectionRuntimeOptions[] = [];
    const detectedBoxes = [{ x: 1, y: 2, width: 3, height: 4 }];

    detectionServiceInternals.preprocessDetection = async (_image, runtimeOptions) => {
        seenOptions.push(runtimeOptions);
        return {
            tensor: new Float32Array([1]),
            resizeParams: {
                srcWidth: 1,
                srcHeight: 1,
                dstWidth: 1,
                dstHeight: 1,
                scaleWidth: 1,
                scaleHeight: 1,
            },
        };
    };
    detectionServiceInternals.runInference = async () => new Float32Array([1]);
    detectionServiceInternals.postprocessDetection = (_detection, _input, runtimeOptions) => {
        seenOptions.push(runtimeOptions);
        return detectedBoxes;
    };

    const result = await detectionService.run(
        {} as unknown as Parameters<DetectionService["run"]>[0],
        {
            minimumAreaThreshold: 9,
            paddingBoxVertical: 0,
            paddingBoxHorizontal: 0,
            dilationKernelSize: 3,
            onProgress: (event) => events.push(event),
        }
    );

    assert.deepEqual(result, detectedBoxes);
    assert.deepEqual(seenOptions, [
        {
            ...DEFAULT_DETECTION_OPTIONS,
            minimumAreaThreshold: 9,
            paddingBoxVertical: 0,
            paddingBoxHorizontal: 0,
            dilationKernelSize: 3,
        },
        {
            ...DEFAULT_DETECTION_OPTIONS,
            minimumAreaThreshold: 9,
            paddingBoxVertical: 0,
            paddingBoxHorizontal: 0,
            dilationKernelSize: 3,
        },
    ]);
    assert.deepEqual(events, [
        {
            type: "det",
            stage: "preprocess",
            progress: { current: 1, remain: 2, total: 3 },
        },
        {
            type: "det",
            stage: "infer",
            progress: { current: 2, remain: 1, total: 3 },
        },
        {
            type: "det",
            stage: "postprocess",
            progress: { current: 3, remain: 0, total: 3 },
            detectedCount: 1,
        },
    ]);
});

test("DetectionService allows zero padding ratios when applying padding to boxes", () => {
    const detectionService = new DetectionService(
        { Tensor: class Tensor {} } as unknown as OrtModule,
        { outputNames: [], run: async () => ({}) } as unknown as OrtInferenceSession,
        {}
    );
    const detectionServiceInternals = detectionService as unknown as DetectionServiceInternals;

    const padded = detectionServiceInternals.applyPaddingToRect(
        { x: 10, y: 15, width: 20, height: 8 },
        100,
        100,
        {
            ...DEFAULT_DETECTION_OPTIONS,
            paddingBoxVertical: 0,
            paddingBoxHorizontal: 0,
        }
    );

    assert.deepEqual(padded, { x: 10, y: 15, width: 20, height: 8 });
});

test("RecognitionService honors recognition overrides, ordering overrides, and progress events", async () => {
    const recognitionService = new RecognitionService(
        { Tensor: class Tensor {} } as unknown as OrtModule,
        { run: async () => ({}) } as unknown as OrtInferenceSession,
        { charactersDictionary: ["", "A"], imageHeight: 48 }
    );
    const recognitionServiceInternals =
        recognitionService as unknown as RecognitionServiceInternals;
    const events: unknown[] = [];
    const runtimeOptionsSeen: RecognitionRuntimeOptions[] = [];
    const boxes = [
        { x: 30, y: 0, width: 10, height: 10 },
        { x: 5, y: 2, width: 10, height: 10 },
        { x: 20, y: 20, width: 10, height: 10 },
    ];

    recognitionServiceInternals.processBox = async (task, runtimeOptions) => {
        runtimeOptionsSeen.push(runtimeOptions);
        assert.ok(task.charWhiteSet?.has("A"));

        return {
            text: `box-${task.index}`,
            box: task.box,
            confidence: 1,
        };
    };

    const results = await recognitionService.run(
        {} as unknown as Parameters<RecognitionService["run"]>[0],
        boxes,
        {
            charWhiteList: ["A"],
            recognition: {
                charactersDictionary: ["", "B"],
                imageHeight: 64,
            },
            ordering: {
                sameLineThresholdRatio: 0,
            },
            onProgress: (event) => events.push(event),
        }
    );

    assert.deepEqual(
        runtimeOptionsSeen,
        Array.from({ length: 3 }, () => ({
            charactersDictionary: ["", "B"],
            imageHeight: 64,
            mean: [127.5, 127.5, 127.5],
            stdDeviation: [1 / 127.5, 1 / 127.5, 1 / 127.5],
        }))
    );
    assert.deepEqual(
        results.map((result) => result.box),
        [boxes[0], boxes[1], boxes[2]]
    );
    assert.deepEqual(
        events.map((event) => {
            const progressEvent = event as {
                stage: string;
                progress: { current: number; total: number };
            };
            return `${progressEvent.stage}:${progressEvent.progress.current}/${progressEvent.progress.total}`;
        }),
        ["start:0/3", "item:1/3", "item:2/3", "item:3/3", "complete:3/3"]
    );
});

test("RecognitionService can keep original detection order or sort by reading order", () => {
    const recognitionService = new RecognitionService(
        { Tensor: class Tensor {} } as unknown as OrtModule,
        { run: async () => ({}) } as unknown as OrtInferenceSession,
        {}
    );
    const recognitionServiceInternals =
        recognitionService as unknown as RecognitionServiceInternals;
    const boxes = [
        { x: 30, y: 0, width: 10, height: 10 },
        { x: 5, y: 2, width: 10, height: 10 },
    ];

    const sorted = recognitionServiceInternals.sortBoxesByReadingOrder(boxes, {
        sortByReadingOrder: true,
        sameLineThresholdRatio: 0.25,
    });
    const original = recognitionServiceInternals.sortBoxesByReadingOrder(boxes, {
        sortByReadingOrder: false,
        sameLineThresholdRatio: 0.25,
    });

    assert.deepEqual(sorted, [boxes[1], boxes[0]]);
    assert.deepEqual(original, boxes);
});

test("PaddleOcrService.processRecognition supports configurable line merge thresholds", () => {
    const service = new PaddleOcrService({ ort: {} as unknown as OrtModule });
    const recognition = [
        {
            text: "A",
            box: { x: 0, y: 0, width: 10, height: 10 },
            confidence: 0.9,
        },
        {
            text: "B",
            box: { x: 20, y: 6, width: 10, height: 10 },
            confidence: 0.8,
        },
    ];

    const defaultProcessed = service.processRecognition(recognition);
    const mergedProcessed = service.processRecognition(recognition, {
        lineMergeThresholdRatio: 1,
    });

    assert.equal(defaultProcessed.lines.length, 2);
    assert.equal(defaultProcessed.text, "A\nB");
    assert.equal(mergedProcessed.lines.length, 1);
    assert.equal(mergedProcessed.text, "A B");
});
