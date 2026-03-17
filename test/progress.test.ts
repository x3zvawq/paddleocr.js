import assert from "node:assert/strict";
import test from "node:test";
import {
    DetectionService,
    type OrtInferenceSession,
    type OrtModule,
    PaddleOcrService,
    type PreprocessDetectionResult,
    type RecognitionResult,
    RecognitionService,
} from "../src/index.ts";

type DetectionServiceInternals = DetectionService & {
    postprocessDetection: (
        detection: Float32Array,
        input: PreprocessDetectionResult
    ) => Array<{
        x: number;
        y: number;
        width: number;
        height: number;
    }>;
    preprocessDetection: (
        image: Parameters<DetectionService["run"]>[0]
    ) => Promise<PreprocessDetectionResult>;
    runInference: (
        tensor: Float32Array,
        input: PreprocessDetectionResult["resizeParams"]
    ) => Promise<Float32Array | null>;
};

type RecognitionServiceInternals = RecognitionService & {
    processBox: (task: {
        box: { x: number; y: number; width: number; height: number };
        charWhiteSet?: Set<string>;
        image: Parameters<RecognitionService["run"]>[0];
        index: number;
    }) => Promise<RecognitionResult | null>;
};

function createFakeInput() {
    return {
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
    };
}

test("PaddleOcrService.recognize forwards onProgress without changing the final result", async () => {
    const detections = [{ x: 0, y: 0, width: 1, height: 1 }];
    const expected = [{ text: "A", box: detections[0], confidence: 0.99 }];
    const service = new PaddleOcrService({ ort: {} as unknown as OrtModule });
    const events: unknown[] = [];

    service.detectionService = {
        run: async (_image, onProgress) => {
            onProgress?.({
                type: "det",
                stage: "preprocess",
                progress: { current: 1, remain: 2, total: 3 },
            });
            return detections;
        },
    } as unknown as DetectionService;

    service.recognitionService = {
        run: async (_image, detection, options) => {
            assert.deepEqual(detection, detections);
            assert.deepEqual(options?.charWhiteList, ["A"]);

            options?.onProgress?.({
                type: "rec",
                stage: "complete",
                progress: { current: 1, remain: 0, total: 1 },
            });

            return expected;
        },
    } as unknown as RecognitionService;

    const result = await service.recognize(createFakeInput(), {
        charWhiteList: ["A"],
        onProgress: (event) => events.push(event),
    });

    assert.deepEqual(result, expected);
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
    ]);
});

test("DetectionService emits preprocess, infer, and postprocess progress in order", async () => {
    const detectionService = new DetectionService(
        { Tensor: class Tensor {} } as unknown as OrtModule,
        { outputNames: [], run: async () => ({}) } as unknown as OrtInferenceSession,
        {}
    );
    const detectionServiceInternals = detectionService as unknown as DetectionServiceInternals;
    const events: unknown[] = [];
    const detectedBoxes = [{ x: 1, y: 2, width: 3, height: 4 }];

    detectionServiceInternals.preprocessDetection = async () => ({
        tensor: new Float32Array([1]),
        resizeParams: {
            srcWidth: 1,
            srcHeight: 1,
            dstWidth: 1,
            dstHeight: 1,
            scaleWidth: 1,
            scaleHeight: 1,
        },
    });
    detectionServiceInternals.runInference = async () => new Float32Array([1]);
    detectionServiceInternals.postprocessDetection = () => detectedBoxes;

    const result = await detectionService.run(
        {} as unknown as Parameters<DetectionService["run"]>[0],
        (event) => events.push(event)
    );

    assert.deepEqual(result, detectedBoxes);
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

test("RecognitionService emits ordered progress events and keeps item events aligned with final results", async () => {
    const recognitionService = new RecognitionService(
        { Tensor: class Tensor {} } as unknown as OrtModule,
        { run: async () => ({}) } as unknown as OrtInferenceSession,
        { charactersDictionary: ["", "A"] }
    );
    const recognitionServiceInternals =
        recognitionService as unknown as RecognitionServiceInternals;
    const events: unknown[] = [];
    const boxes = [
        { x: 30, y: 30, width: 10, height: 10 },
        { x: 5, y: 0, width: 10, height: 10 },
        { x: 20, y: 0, width: 10, height: 10 },
    ];

    recognitionServiceInternals.processBox = async (task) => {
        assert.ok(task.charWhiteSet.has("A"));

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
            onProgress: (event) => events.push(event),
        }
    );

    const itemEvents = events.filter(
        (
            event
        ): event is (typeof events)[number] & { stage: "item"; box: unknown; result: unknown } =>
            (event as { stage?: string }).stage === "item"
    );

    assert.deepEqual(
        results.map((result) => result.box),
        [boxes[1], boxes[2], boxes[0]]
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
    assert.deepEqual(
        itemEvents.map((event) => event.box),
        results.map((result) => result.box)
    );
    assert.deepEqual(
        itemEvents.map((event) => event.result),
        results
    );
});

test("RecognitionService emits start and complete when there are no valid boxes", async () => {
    const recognitionService = new RecognitionService(
        { Tensor: class Tensor {} } as unknown as OrtModule,
        { run: async () => ({}) } as unknown as OrtInferenceSession,
        { charactersDictionary: ["", "A"] }
    );
    const events: unknown[] = [];

    const results = await recognitionService.run(
        {} as unknown as Parameters<RecognitionService["run"]>[0],
        [],
        {
            onProgress: (event) => events.push(event),
        }
    );

    assert.deepEqual(results, []);
    assert.deepEqual(events, [
        {
            type: "rec",
            stage: "start",
            progress: { current: 0, remain: 0, total: 0 },
        },
        {
            type: "rec",
            stage: "complete",
            progress: { current: 0, remain: 0, total: 0 },
        },
    ]);
});
