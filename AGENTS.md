# paddleocr.js agent guide

这个仓库是一个轻量 TypeScript PaddleOCR runtime，目标是在不引入大型运行时依赖的前提下，尽量贴合官方 PaddleOCR / PaddleX 的 ONNX 推理链路、模型 preset、模块能力和 examples 展示。

## 参考文档

1. [docs/official-parity-roadmap.md](./docs/official-parity-roadmap.md)
    - 当前已支持模块、官方证据、关键差异和下一步优先级都在这里。
2. [assets/README.md](./assets/README.md)
    - 本地 ONNX 资产来源、大文件放置规则、`assets/local/` 下载/转换说明。
3. [examples/README.md](./examples/README.md)
    - examples 的设计口径：像用户项目嵌入本库，而不是内部 CLI。

## 当前项目概况

- Runtime 入口在 `src/`，按模块拆成 `processor/`、`presets/`、`utils/`。
- 调用方负责传入像素，runtime 不解码 PNG/JPEG，不应新增图像 codec、大型 OpenCV、pyclipper 等运行时依赖。
- 开发期脚本和验证用依赖可以放在 `tmp/` 或 examples 侧，但不能变成 runtime dependency。
- 已真实可跑的主链路包括 PP-OCRv5 mobile、PP-OCRv6 tiny/small 的 det + rec。
- 已有本地验证资产包括分类模块、seal detection、SLANet、UVDoc、PP-DocBlockLayout、RT-DETR table cell，以及本地转换后的 PP-FormulaNet_plus-M。
- 大模型或转换后资产应放在忽略的 `assets/local/`，或后续发布到 Hugging Face；不要直接提交大二进制。

## 重要设计边界

- 优先同步真实影响输出的逻辑：resize、normalize、channel order、DBPostProcess、CTC decode、排序、裁剪、unclip、NMS、table/formula/layout 后处理。
- 轻量原则优先：如果官方依赖 pyclipper/OpenCV 的步骤需要近似实现，必须在文档中标明差异，不要伪装成 bit-exact parity。
- 新 preset 应尽量从官方 `inference.yml` / `inference.json` / ONNX metadata 推导，不要凭相似模型猜默认值。
- examples 应展示用户如何加载 ONNX、字典/label、选择 preset/模块服务、传入自有像素、处理输出对象。
- README 和 examples 输出图应展示识别前后效果；生成 PNG 可以用开发期依赖，不要加入 runtime。

## 提交习惯

- 每完成一个独立点单独提交。
- commit message 使用一行中文 conventional commit，例如：
    - `fix(detection): 贴合印章弧形轮廓`
- 不要 stage 或回滚用户已有的无关改动。
