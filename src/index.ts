export {
    DEFAULT_DETECTION_OPTIONS,
    DEFAULT_PADDLE_OPTIONS,
    DEFAULT_RECOGNITION_OPTIONS,
} from "./constants.ts";

export type {
    Box,
    DetectionServiceOptions,
    OcrProgress,
    OrtInferenceSession,
    OrtModule,
    OrtTensor,
    PaddleOcrProgressEvent,
    PaddleOptions,
    RecognitionOptions,
    RecognitionServiceOptions,
} from "./interface.ts";

export { DetectionService, type PreprocessDetectionResult } from "./processor/detection.ts";
export {
    type FlattenedPaddleOcrResult,
    type PaddleOcrResult,
    PaddleOcrService,
} from "./processor/paddle-ocr.ts";
export { type RecognitionResult, RecognitionService } from "./processor/recognition.ts";
