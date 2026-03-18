# PaddleOcr

[中文文档 (Chinese Documentation)](./README_zh.md)

A lightweight, type-safe, dependency-free JavaScript/TypeScript library for PaddleOCR, supporting both Node.js and browser environments.

## Features

- **Cross-platform**: Works in Node.js, Bun, and browser environments.
- **Type-safe**: Written in TypeScript with full type definitions.
- **No dependencies**: Minimal footprint, no heavy image processing libraries included.
- **Flexible model loading**: Accepts model files as ArrayBuffer, allowing custom loading strategies (e.g., fetch, fs.readFileSync).
- **ONNX Runtime support**: Compatible with both `onnxruntime-web` and `onnxruntime-node`.
- **Customizable dictionary**: Pass your own character dictionary for recognition.
- **Modern API**: Simple, promise-based API for easy integration.

## Installation

```bash
npm install paddleocr
# or
yarn add paddleocr
# or
pnpm add paddleocr
```

## Usage

### 1. Prepare ONNX Runtime and Model Files

- In browser:
    ```js
    import * as ort from "onnxruntime-web";
    ```
- In Node.js or Bun:
    ```js
    import * as ort from "onnxruntime-node";
    ```

### 2. Load Model Files and Dictionary

You can use `fetch`, `fs.readFileSync`, or any other method to load your ONNX model files and dictionary as ArrayBuffer and string array, respectively.

### 3. Initialize the Service

```js
import { PaddleOcrService } from "paddleocr";

const paddleOcrService = await PaddleOcrService.createInstance({
    ort,
    detection: {
        modelBuffer: detectOnnx,
        minimumAreaThreshold: 24,
        textPixelThreshold: 0.55,
        paddingBoxVertical: 0.3,
        paddingBoxHorizontal: 0.5,
    },
    recognition: {
        modelBuffer: recOnnx,
        charactersDictionary: dict,
        imageHeight: 48,
    },
});
```

The `detection` and `recognition` objects above act as instance defaults. They are applied to every `recognize()` call unless you override them per request.

### 4. Prepare Image Data

The `recognize` method expects an object with `width`, `height`, and `data` (Uint8Array of RGB(A) values). Use your preferred image decoding library (e.g., `fast-png`, `image-js`).

```js
import { decode } from "fast-png";
const imageFile = await readFile("tests/image.png");
const buffer = imageFile.buffer.slice(imageFile.byteOffset, imageFile.byteOffset + imageFile.byteLength);
const image = decode(buffer);
const input = {
    data: image.data,
    width: image.width,
    height: image.height,
};
```

### 5. Run OCR

```js
const result = await paddleOcrService.recognize(input);
console.log(result);
```

### 6. Track Progress

You can pass `onProgress` to `recognize` to receive detection and recognition updates in real time.

```js
const result = await paddleOcrService.recognize(input, {
    onProgress(event) {
        console.log(event.type, event.stage, event.progress);

        if (event.type === "rec" && event.stage === "item") {
            console.log("Partial result:", event.result?.text, event.box);
        }
    },
});
```

Event contract:

- `det` emits `preprocess`, `infer`, and `postprocess` with fixed progress `1/3`, `2/3`, `3/3`
- `rec` emits `start`, one `item` per detected text box, then `complete`
- `rec/item` includes the current `result` and `box`
- `det/postprocess` includes `detectedCount`

### 7. Tune Detection, Recognition, and Ordering

You can set defaults when creating the instance, then override them for a single image.

```js
const strictResult = await paddleOcrService.recognize(invoiceInput, {
    detection: {
        minimumAreaThreshold: 40,
        textPixelThreshold: 0.65,
        paddingBoxVertical: 0,
        paddingBoxHorizontal: 0,
        dilationKernelSize: 3,
    },
    recognition: {
        imageHeight: 64,
        charactersDictionary: digitsOnlyDict,
    },
    ordering: {
        sortByReadingOrder: true,
        sameLineThresholdRatio: 0.15,
    },
});

const looseResult = await paddleOcrService.recognize(noteInput, {
    detection: {
        minimumAreaThreshold: 8,
        textPixelThreshold: 0.45,
    },
});
```

Supported per-call override groups:

- `detection`: `padding`, `mean`, `stdDeviation`, `maxSideLength`, `paddingBoxVertical`, `paddingBoxHorizontal`, `minimumAreaThreshold`, `textPixelThreshold`, `dilationKernelSize`
- `recognition`: `imageHeight`, `mean`, `stdDeviation`, `charactersDictionary`
- `ordering`: `sortByReadingOrder`, `sameLineThresholdRatio`

`charactersDictionary` can be provided either when creating the instance or per call. If neither is provided, `recognize()` throws.

### 8. Control Line Merging in `processRecognition`

```js
const rawRecognition = await paddleOcrService.recognize(input);
const processed = paddleOcrService.processRecognition(rawRecognition, {
    lineMergeThresholdRatio: 0.8,
});

console.log(processed.text);
console.log(processed.lines);
```

## Model Files

You can find sample models in the `assets/` directory:

- `PP-OCRv5_mobile_det_infer.onnx`
- `PP-OCRv5_mobile_rec_infer.onnx`
- `ppocrv5_dict.txt`

## Examples

See the `examples/` directory for usage samples.
About browser usage with Vite, check out [paddleocr-vite-example](https://github.com/X3ZvaWQ/paddleocr-vite-example)

## Contributing

Contributions are welcome! Feel free to submit a PR or open an issue.

## License

MIT
