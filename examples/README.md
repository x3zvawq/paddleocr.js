# Examples

[中文说明](./README_zh.md)

These examples are written like code you would embed in your own project: load model files, load a
dictionary or label preset, pass caller-owned pixels, then handle the returned object.

The runtime still does not decode PNG/JPEG. Example code uses `fast-png` only on the examples side.
Set `PADDLEOCR_JS_ONNX_DIR` to the external model directory before running scripts. See
[`../paddleocr-js-onnx/README.md`](../paddleocr-js-onnx/README.md) for the expected model layout.

```sh
PADDLEOCR_JS_ONNX_DIR=./paddleocr-js-onnx bun examples/pipeline/ocr/run.ts
```

Pipeline examples use `PaddleOcrService.createInstance`,
`TableRecognitionV2Service.createInstance`, or `PaddleStructureService.createInstance` so callers
only need to provide ONNX `ArrayBuffer`s, dictionaries/tokenizers, and presets. Module examples show
the lower-level service shape for users who want to wire a single module directly.

## How To Read Result Images

Generated result images use a consistent three-panel layout:

- left: original input image;
- middle: final output visualization, such as recognized text, detected boxes, recovered table
  structure, or corrected image;
- right: useful intermediate metadata, such as scores, coordinates, structure tokens, OCR counts,
  or recovered HTML summaries.

The result images are not golden snapshots for tests. They are publishing assets and onboarding
material for users who want to understand what each module returns.

## Module Examples

| Module | What it demonstrates | Directory | Result |
| --- | --- | --- | --- |
| Document orientation classification | classify page rotation and rotate the image upright | [`module/document-orientation`](./module/document-orientation/) | [`module-document-orientation.png`](./result/module-document-orientation.png) |
| Textline orientation classification | classify a single text line, especially 0/180 degree correction | [`module/textline-orientation`](./module/textline-orientation/) | [`module-textline-orientation.png`](./result/module-textline-orientation.png) |
| Table classification | classify wired vs wireless table type | [`module/table-classification`](./module/table-classification/) | [`module-table-classification.png`](./result/module-table-classification.png) |
| Text detection | DB text boxes/polygons without recognition | [`module/text-detection`](./module/text-detection/) | [`module-text-detection.png`](./result/module-text-detection.png) |
| Text recognition | CTC recognition for a caller-provided text crop | [`module/text-recognition`](./module/text-recognition/) | [`module-text-recognition.png`](./result/module-text-recognition.png) |
| Seal text detection | curved seal text polygons | [`module/seal-text-detection`](./module/seal-text-detection/) | [`module-seal-text-detection.png`](./result/module-seal-text-detection.png) |
| Text image unwarping | UVDoc dewarping for curved document photos | [`module/text-image-unwarping`](./module/text-image-unwarping/) | [`module-text-image-unwarping.png`](./result/module-text-image-unwarping.png) |
| Region detection (PP-DocBlockLayout) | coarse document region detection | [`module/region-detection`](./module/region-detection/) | [`module-region-detection.png`](./result/module-region-detection.png) |
| Semantic layout detection (PP-DocLayout) | semantic layout labels such as text/table/title | [`module/layout-detection`](./module/layout-detection/) | [`module-layout-detection.png`](./result/module-layout-detection.png) |
| Table cell detection | RT-DETR table cell boxes | [`module/table-cell-detection`](./module/table-cell-detection/) | [`module-table-cell-detection.png`](./result/module-table-cell-detection.png) |
| Table structure recognition | SLANet structure tokens and recovered grid | [`module/table-structure`](./module/table-structure/) | [`module-table-structure.png`](./result/module-table-structure.png) |
| Formula recognition | PP-FormulaNet LaTeX token decoding | [`module/formula-recognition`](./module/formula-recognition/) | [`module-formula-recognition.png`](./result/module-formula-recognition.png) |

## Pipeline Examples

| Pipeline | What it demonstrates | Directory | Result |
| --- | --- | --- | --- |
| OCR det + textline orientation + rec | full page/sign OCR with detected boxes and recognized text | [`pipeline/ocr`](./pipeline/ocr/) | [`pipeline-ocr.png`](./result/pipeline-ocr.png) |
| Table Recognition V2 | SLANeXt + cell detector + OCR + HTML recovery | [`pipeline/table-recognition-v2`](./pipeline/table-recognition-v2/) | [`pipeline-table-recognition-v2.png`](./result/pipeline-table-recognition-v2.png) |
| PP-Structure-like document parsing | orientation, unwarping, layout, reading order, OCR, table regions, markdown | [`pipeline/pp-structure`](./pipeline/pp-structure/) | [`pipeline-pp-structure.png`](./result/pipeline-pp-structure.png) |

Shared sample images live in [`input/`](./input/). Generated comparison images live in
[`result/`](./result/) and show the original image on the left, the final recognition or detection
visualization in the middle, and useful intermediate outputs on the right when a module or pipeline
exposes them.

Regenerate all result images:

```sh
bun examples/generate-results.ts
```

Regenerate selected result images:

```sh
bun examples/generate-results.ts --only pipeline-ocr
EXAMPLE_ONLY=module-layout-detection,pipeline-pp-structure bun examples/generate-results.ts
```

Available task names match the result file names without `.png`, for example
`module-text-recognition`, `module-table-structure`, `pipeline-ocr`, and
`pipeline-pp-structure`.

## Commands

```sh
bun examples/module/document-orientation/run.ts
bun examples/module/textline-orientation/run.ts
bun examples/module/table-classification/run.ts
bun examples/module/text-detection/run.ts
bun examples/module/text-recognition/run.ts
bun examples/module/seal-text-detection/run.ts
bun examples/module/text-image-unwarping/run.ts
bun examples/module/region-detection/run.ts
bun examples/module/layout-detection/run.ts
bun examples/module/table-cell-detection/run.ts
bun examples/module/table-structure/run.ts
bun examples/module/formula-recognition/run.ts
bun examples/pipeline/ocr/run.ts
bun examples/pipeline/table-recognition-v2/run.ts
bun examples/pipeline/pp-structure/run.ts
```
