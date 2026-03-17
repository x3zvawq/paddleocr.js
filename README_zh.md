# Paddleocr 中文文档

[English Documentation](./README.md)

一个轻量级、类型安全、无依赖的 PaddleOCR JavaScript/TypeScript 库，同时支持 Node.js 与浏览器环境。

## 特性

- **跨平台**：支持 Node.js、Bun 和浏览器环境。
- **类型安全**：TypeScript 编写，完整类型定义。
- **零依赖**：极小体积，无额外图片处理库。
- **灵活模型加载**：模型文件以 ArrayBuffer 传入，支持自定义加载方式（如 fetch、fs.readFileSync）。
- **ONNX Runtime 支持**：兼容 `onnxruntime-web` 和 `onnxruntime-node`。
- **可自定义字典**：可传入自定义字符字典。
- **现代 API**：Promise 风格，易于集成。

## 安装

```bash
npm install paddleocr
# 或
yarn add paddleocr
# 或
pnpm add paddleocr
```

## 使用方法

### 1. 准备 ONNX Runtime 和模型文件

- 浏览器环境：
    ```js
    import * as ort from "onnxruntime-web";
    ```
- Node.js 或 Bun 环境：
    ```js
    import * as ort from "onnxruntime-node";
    ```

### 2. 加载模型文件和字典

可用 `fetch`、`fs.readFileSync` 等方式加载 ONNX 模型文件（ArrayBuffer）和字典（字符串数组）。

### 3. 初始化服务

```js
import { PaddleOcrService } from "paddleocr";

const paddleOcrService = await PaddleOcrService.createInstance({
    ort,
    detection: {
        modelBuffer: detectOnnx,
    },
    recognition: {
        modelBuffer: recOnnx,
        charactersDictionary: dict,
    },
});
```

### 4. 准备图片数据

`recognize` 方法需要传入包含 `width`、`height`、`data`（Uint8Array，RGB(A)）的对象。推荐使用 `fast-png`、`image-js` 等库进行图片解码。

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

### 5. 识别文字

```js
const result = await paddleOcrService.recognize(input);
console.log(result);
```

### 6. 获取实时进度

可以在 `recognize` 的第二个参数里传入 `onProgress`，实时接收检测和识别阶段的事件。

```js
const result = await paddleOcrService.recognize(input, {
    onProgress(event) {
        console.log(event.type, event.stage, event.progress);

        if (event.type === "rec" && event.stage === "item") {
            console.log("部分结果:", event.result?.text, event.box);
        }
    },
});
```

事件约定：

- `det` 会依次发出 `preprocess`、`infer`、`postprocess`，对应固定进度 `1/3`、`2/3`、`3/3`
- `rec` 会先发 `start`，然后每个文本框完成时发一次 `item`，最后发 `complete`
- `rec/item` 会带上当前 `result` 和 `box`
- `det/postprocess` 会额外带上 `detectedCount`

## 模型文件

示例模型见github仓库的 `assets/` 目录：

- `PP-OCRv5_mobile_det_infer.onnx`
- `PP-OCRv5_mobile_rec_infer.onnx`
- `ppocrv5_dict.txt`

## 示例

更多用法见 `examples/` 目录。
关于浏览器vite用法，见[paddleocr-example](https://github.com/X3ZvaWQ/paddleocr-vite-example)

## 贡献

欢迎提交 PR 或 Issue！

## 许可证

MIT
