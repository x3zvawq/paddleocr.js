import { readFile } from "node:fs/promises";
import { decode } from "fast-png";
import * as ort from "onnxruntime-node";
import { PaddleOcrService } from "../src/index.ts";

const imageFile = await readFile("examples/image.png");
const buffer = imageFile.buffer.slice(
    imageFile.byteOffset,
    imageFile.byteOffset + imageFile.byteLength
);
const image = decode(buffer);
const input = {
    data: image.data as Uint8Array,
    width: image.width,
    height: image.height,
};

const detectOnnx = await readFile("assets/PP-OCRv5_mobile_det_infer.onnx");
const recOnnx = await readFile("assets/PP-OCRv5_mobile_rec_infer.onnx");
const dict = await readFile("assets/ppocrv5_dict.txt", "utf-8").then((res) => res.split("\n"));

const paddleOcrService = await PaddleOcrService.createInstance({
    ort,
    detection: {
        modelBuffer: detectOnnx.buffer as ArrayBuffer,
    },
    recognition: {
        modelBuffer: recOnnx.buffer as ArrayBuffer,
        charactersDictionary: [...dict, " "],
    },
});

const r = await paddleOcrService.recognize(input, {
    charWhiteList: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
});
console.log(r);
