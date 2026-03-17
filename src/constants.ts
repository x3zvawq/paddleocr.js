import type {
    DetectionServiceOptions,
    PaddleOptions,
    RecognitionServiceOptions,
} from "./interface.ts";

type DetectionDefaults = Required<Omit<DetectionServiceOptions, "modelBuffer">>;
type RecognitionDefaults = Required<Omit<RecognitionServiceOptions, "modelBuffer">>;

export const DEFAULT_DETECTION_OPTIONS: DetectionDefaults = {
    padding: 0,
    mean: [0.485 * 255, 0.456 * 255, 0.406 * 255],
    stdDeviation: [1 / 0.229 / 255, 1 / 0.224 / 255, 1 / 0.255 / 255],
    maxSideLength: 960,
    textPixelThreshold: 0.5,
    minimumAreaThreshold: 20,
    paddingBoxVertical: 0.4,
    paddingBoxHorizontal: 0.6,
};

export const DEFAULT_RECOGNITION_OPTIONS: RecognitionDefaults = {
    mean: [127.5, 127.5, 127.5],
    stdDeviation: [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5],
    imageHeight: 48,
    charactersDictionary: [],
};

export const DEFAULT_PADDLE_OPTIONS: Partial<PaddleOptions> = {
    detection: DEFAULT_DETECTION_OPTIONS,
    recognition: DEFAULT_RECOGNITION_OPTIONS,
};
