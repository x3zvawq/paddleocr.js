# Examples 示例

[English](./README.md)

这些示例按“用户项目里应该怎么接入”的方式编写：加载模型、加载字典或 label preset、
选择模块或流水线、传入调用方自己的像素，然后处理返回对象。

runtime 本身仍然不解码 PNG/JPEG。examples 侧为了读取样例图片使用 `fast-png`，这不是
runtime dependency。运行脚本前请把 `PADDLEOCR_JS_ONNX_DIR` 指向外部模型目录；模型目录结构见
[`../paddleocr-js-onnx/README_zh.md`](../paddleocr-js-onnx/README_zh.md)。

```sh
PADDLEOCR_JS_ONNX_DIR=./paddleocr-js-onnx bun examples/pipeline/ocr/run.ts
```

Pipeline 示例会使用 `PaddleOcrService.createInstance`、
`TableRecognitionV2Service.createInstance` 或 `PaddleStructureService.createInstance`，
用户只需要提供 ONNX `ArrayBuffer`、字典/tokenizer 和 preset。Module 示例展示更底层的单模块
service 接法，适合只想单独使用某个模块的用户。

## 如何看结果图

生成的 result 图统一是三栏：

- 左侧：原始输入图；
- 中间：最终输出可视化，例如识别文字、检测框、恢复的表格结构或矫正后的图片；
- 右侧：有用的中间信息，例如 score、坐标、结构 token、OCR 数量、HTML 摘要等。

这些 result 图不是测试 golden snapshot，而是给用户看的接入说明和效果展示。

## 单模块示例

| 模块 | 展示内容 | 目录 | 结果图 |
| --- | --- | --- | --- |
| 文档图像方向分类 | 判断页面旋转角度，并展示摆正后的图片 | [`module/document-orientation`](./module/document-orientation/) | [`module-document-orientation.png`](./result/module-document-orientation.png) |
| 文本行方向分类 | 判断单行文本方向，尤其是 0/180 度修正 | [`module/textline-orientation`](./module/textline-orientation/) | [`module-textline-orientation.png`](./result/module-textline-orientation.png) |
| 表格分类 | 判断有线表格 / 无线表格 | [`module/table-classification`](./module/table-classification/) | [`module-table-classification.png`](./result/module-table-classification.png) |
| 文本检测 | 只跑 DB 文本框/多边形检测，不做识别 | [`module/text-detection`](./module/text-detection/) | [`module-text-detection.png`](./result/module-text-detection.png) |
| 文本识别 | 对调用方给出的一行文本 crop 做 CTC 识别 | [`module/text-recognition`](./module/text-recognition/) | [`module-text-recognition.png`](./result/module-text-recognition.png) |
| 印章文本检测 | 检测弧形印章文字多边形 | [`module/seal-text-detection`](./module/seal-text-detection/) | [`module-seal-text-detection.png`](./result/module-seal-text-detection.png) |
| 文本图像矫正 | UVDoc 对弯曲拍照文档做去弯曲/矫正 | [`module/text-image-unwarping`](./module/text-image-unwarping/) | [`module-text-image-unwarping.png`](./result/module-text-image-unwarping.png) |
| 区域检测 PP-DocBlockLayout | 粗粒度文档区域检测 | [`module/region-detection`](./module/region-detection/) | [`module-region-detection.png`](./result/module-region-detection.png) |
| 语义版面检测 PP-DocLayout | 检测 text/table/title 等语义版面元素 | [`module/layout-detection`](./module/layout-detection/) | [`module-layout-detection.png`](./result/module-layout-detection.png) |
| 表格单元格检测 | RT-DETR 表格单元格框 | [`module/table-cell-detection`](./module/table-cell-detection/) | [`module-table-cell-detection.png`](./result/module-table-cell-detection.png) |
| 表格结构识别 | SLANet 结构 token 和恢复出的表格网格 | [`module/table-structure`](./module/table-structure/) | [`module-table-structure.png`](./result/module-table-structure.png) |
| 公式识别 | PP-FormulaNet LaTeX token 解码 | [`module/formula-recognition`](./module/formula-recognition/) | [`module-formula-recognition.png`](./result/module-formula-recognition.png) |

## 流水线示例

| 流水线 | 展示内容 | 目录 | 结果图 |
| --- | --- | --- | --- |
| OCR det + 文本行方向 + rec | 完整 OCR，展示检测框和最终文字 | [`pipeline/ocr`](./pipeline/ocr/) | [`pipeline-ocr.png`](./result/pipeline-ocr.png) |
| Table Recognition V2 | SLANeXt + 单元格检测 + OCR + HTML 恢复 | [`pipeline/table-recognition-v2`](./pipeline/table-recognition-v2/) | [`pipeline-table-recognition-v2.png`](./result/pipeline-table-recognition-v2.png) |
| 类 PP-Structure 文档解析 | 方向、矫正、layout、reading order、OCR、表格区域和 markdown | [`pipeline/pp-structure`](./pipeline/pp-structure/) | [`pipeline-pp-structure.png`](./result/pipeline-pp-structure.png) |

共享样例图片在 [`input/`](./input/)。生成的对比效果图在 [`result/`](./result/)。

重新生成全部效果图：

```sh
bun examples/generate-results.ts
```

只重新生成部分效果图：

```sh
bun examples/generate-results.ts --only pipeline-ocr
EXAMPLE_ONLY=module-layout-detection,pipeline-pp-structure bun examples/generate-results.ts
```

可用 task 名称和结果图文件名一致，只是不带 `.png`，例如
`module-text-recognition`、`module-table-structure`、`pipeline-ocr`、
`pipeline-pp-structure`。

## 运行命令

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
