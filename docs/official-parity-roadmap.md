# PaddleOCR 官方兼容路线

本文档用来记录 `paddleocr.js` 当前和官方
PaddleOCR / PaddleX 推理链路的对齐状态、真实验证证据和后续优先级。

官方参考以本机仓库 `/Users/x3zvawq/workspace/PaddleOCR` 为准；模型参数优先来自
官方 `inference.yml`、`inference.json`、ONNX metadata 和本地可跑资产。

## 当前已支持

### 通用 OCR

- 支持 PP-OCRv5 mobile、PP-OCRv6 tiny/small 的 det + rec 主链路。
- PP-OCRv6 medium 已有 preset 契约，但大 ONNX 未入库，下载说明在 `assets/README.md`。
- 检测前处理覆盖 `DetResizeForTest` 的 max/min/resize_long/fixed input shape、小图 padding、32 倍数约束。
- 检测/识别/分类的 resize 默认使用 OpenCV 风格双线性采样。
- 检测输出读取 DB 第一通道，即官方 `pred[:, 0, :, :]`。
- 检测输出 tensor shape 按官方 DB maps `[1, C, H, W]` 校验，不接受 NHWC 或扁平输出。
- 检测输出长度按完整 DB score map 通道校验，短输出或非整通道输出会 fail-fast。
- DB 阈值、box score 使用原始 float score map，不再先量化为 8-bit。
- DB 坐标按官方方式 round，并 clip 到 `[0, dest_width]` / `[0, dest_height]`。
- DB `use_dilation=True` 对齐为一次 2x2 全 1 kernel 膨胀。
- DB quad 支持 `score_mode: "fast" | "slow"`，对应 mini box / 原始 contour 打分。
- DB `box_type` 只接受 `quad` / `poly`，拼错配置会 fail-fast。
- DB contour 候选保留发现顺序，不再按面积重排后截断 `max_candidates`。
- DB contour 会保留连通域的内外边界 loop，贴近 OpenCV `RETR_LIST` 的候选集合。
- DB quad 使用原始 contour 做 mini box；polygon 分支才按官方 `0.002 * arcLength` 做 `approxPolyDP`。
- DB quad `get_mini_boxes` 点序按官方左右 x 分组，再按 y 排序。
- 识别支持 CTC decode、blank 过滤、重复字符折叠、字典长度校验。
- 识别支持 ONNX 固定输入宽度和官方 batch max width ratio padding。
- 识别多输出时支持选择最后一个匹配 CTC logits 的 3D tensor。
- OCR 主链路可按官方 `drop_score` 语义过滤低置信识别结果。
- 检测框排序接近官方 `sorted_boxes`：先按 top-left `(y, x)`，再做同行邻近交换。
- 支持可选 textline orientation correction：识别前对 crop 做 0/180 分类，超过阈值时旋转。

### 图像分类模块

- 有通用 `ImageClassificationService`。
- 已支持 document orientation、textline orientation、table classification。
- PP-LCNet 文档/表格分类使用 resize-short + center-crop + ImageNet normalize。
- PP-LCNet textline orientation 使用官方 160x80 fixed resize；旧 OCR pipeline 的 resize-pad 模式保留为显式选项。
- 支持 ONNX 固定输入 H/W metadata。

### Object detection / Layout / Table cell

- 支持 PaddleX 风格 object detection service。
- 已有 PP-DocBlockLayout、PP-DocLayout_plus-L/L/M/S、RT-DETR wired/wireless table cell preset 契约。
- 支持 DETR/RT-DETR fixed-size 前处理，输入 `image`、`im_shape`、`scale_factor`。
- 支持已导出 `[N,6]` / `[1,N,6]` box tensor 解码为 `{ cls_id, label, score, coordinate }`。
- 导出 box tensor 会校验正整数 shape 和 data length 一致，不接受动态占位维度。
- 支持 `bbox_num` / `boxes_num` 整数标量 companion tensor，避免 padded rows 被当成真实 box。
- 支持 scalar、per-class array、class-id dict 三种 threshold 形态。
- 已实现轻量 layout NMS、box unclip 和 large/small/union merge 选项。

### Seal detection

- 已有 `PP-OCRv4_mobile_seal_det` 和 server seal preset 契约。
- mobile seal ONNX 已本地验证，示例图能检测 4 个印章文字区域。
- seal 使用 `box_type: "poly"`、`resize_long: 736`、DB thresholds `0.2 / 0.6`、`unclip_ratio: 0.5`。
- 多边形 contour 走有序像素边界追踪，再做闭合轮廓 `approxPolyDP`。
- 多边形 score 使用 mask-filled score map 平均，并按 `cv2.fillPoly` 的整数截断语义处理点位。
- polygon unclip 使用轻量 round-join offset 近似，避免旧版 centroid radial expansion。
- polygon 输出保留 unclip 后的多边形路径，不再把少点数结果强制转成轴对齐外接框。

### Table structure

- 已有 SLANet preset、前处理、raw service 和后处理。
- 前处理对齐 long-side resize 到 488、先 normalize 后 padding、CHW tensor、BGR 输入。
- 后处理支持 `structure_probs` / `loc_preds`、no-span dictionary merge、`sos/eos`、score 平均、bbox restore。
- 结果包含 `structure`、`bbox`、`structureScore` 和由 token 拼接出的 HTML-like `html`。
- 支持官方 `predict_structure.py` 的 `<html><body><table>...` 完整 HTML 文档包装。
- 支持按官方 `TableMatch` 的 IoU + L1 distance 口径，把调用方 OCR 文本框匹配到 cell bbox 并填入 HTML。
- OCR-to-cell 文本填入对齐官方多段文本空格、HTML escape 和 `<b>` 粗体包裹语义。
- 本地 SLANet ONNX 示例已验证。

### Formula recognition

- 已有 PP-FormulaNet-S/L、PP-FormulaNet_plus-S/M/L preset、前处理、raw service 和 token/logit 后处理。
- 前处理对齐 UniMERNet 灰度、裁边、thumbnail、pad；S/plus-S/plus-M 使用 384 输入，L/plus-L 使用 768 输入。
- 后处理依赖调用方提供 Nougat tokenizer vocabulary；runtime 不捆绑大 tokenizer。
- examples 已展示如何加载本地 PP-FormulaNet ONNX、官方 tokenizer JSON、选择 preset 和调用方像素。
- 本地转换后的五个 ONNX 均放在外部 `paddleocr-js-onnx/` 工作区，不作为源码提交资产；已用
  `examples/module/formula-recognition/input/formula.png` 验证输出同为 `fetch_name_0` int64 token 序列。

### UVDoc text image unwarping

- 已有 UVDoc preset、前处理、raw service 和 DocTr 风格图像输出后处理。
- 前处理对齐 BGR、normalize、CHW、动态 shape metadata。
- 本地 UVDoc ONNX 示例已验证。

## 已验证资产

源码仓库不再提交 ONNX 模型文件；已验证模型统一外置到被 git 忽略的
`paddleocr-js-onnx/` 工作区，后续作为 Hugging Face 仓库维护。当前包括：

- `ppocr_v5_mobile/`：PP-OCRv5 mobile det/rec、字典。
- `ppocr_v6_tiny/`：PP-OCRv6 tiny det/rec、字典、metadata。
- `ppocr_v6_small/`：PP-OCRv6 small det/rec、字典、metadata。
- `pp_lcnet_x1_0_doc_ori/`：文档方向分类。
- `pp_lcnet_x0_25_textline_ori/`：文本行方向分类。
- `pp_lcnet_x1_0_table_cls/`：表格分类。
- `ppocr_v4_mobile_seal_det/`：印章文本检测。
- `slanet/`、`slanext_wired/`、`slanext_wireless/`：表格结构识别。
- `uvdoc/`：文本图像矫正。
- `pp_docblocklayout/`、`pp_doclayout_plus_l/`、`pp_doclayout_l/m/s/`：版面检测。
- `rt_detr_wired_table_cell_det/`、`rt_detr_wireless_table_cell_det/`：表格单元格检测。
- `PP_FormulaNet_*`、`pp_formulanet_plus_m/`：公式识别。

## 主要差异

### DBPostProcess 仍是最大差异

- 官方依赖 OpenCV contour、`cv2.approxPolyDP`、`cv2.minAreaRect`、mask score、pyclipper round unclip。
- 本库已对齐 float score、dilation、score mode、坐标 clip、点序、闭合 contour 近似和 mask scoring。
- 仍不是 OpenCV/pyclipper 等价实现，尤其 concave、强弯曲、多层 contour hierarchy 和 round join 细节仍可能不同。
- quad unclip 仍是 line-offset 近似；polygon unclip 是轻量 round-join offset 近似。

### Object detection 只覆盖导出 box 简单路径

- 当前能解 `[N,6]` / `[1,N,6]`，适合本地 PP-DocBlockLayout 和 RT-DETR table cell 资产。
- 还没有 GFL/PicoDet raw head decode、DFL decode、anchor/stride generation。
- PP-DocLayout-M/S 等 GFL 家族如果直接暴露 raw heads，需要单独实现，不应复用 DETR 简单路径。

### OCR 识别仍只覆盖 CTC 主流路径

- PP-OCR CTC 已可用。
- SAR/NRTR/attention decoder、word-box metadata、Arabic/right-to-left reverse grouping 尚未实现。
- `cropRotated` 是 TypeScript 透视采样，不可能和 `cv2.getPerspectiveTransform + warpPerspective` bit-exact。

### Table 还不是完整恢复流水线

- SLANet 结构 token、cell bbox、HTML-like skeleton 和基础 OCR-to-cell 文本填入已有。
- 还缺 row/col span recovery 后的最终 table 输出，以及更完整的 table recognition v2 编排。
- table classification 和 table cell detection 还只是独立模块，没有组成完整 table pipeline。

### Formula 资产分发还没完成

- Runtime 能 decode token ids/logits，但不拥有 tokenizer package。
- 当前可跑模型仍是本地转换资产；更好的用户路径应是 Hugging Face ONNX + tokenizer 下载，而不是让用户自己转换和 patch。

### Pipeline orchestration 比官方薄

- 官方 PP-Structure 会串联方向、矫正、layout、OCR、seal、table、formula 等模块。
- 本库目前偏向独立 module service，没有完整 PP-Structure / table recognition v2 / formula pipeline / seal recognition pipeline。
- Chart parsing 和 Document VLM 依赖重 VLM 栈，暂不适合轻量 runtime。

## 后续优先级

1. 继续收紧 DBPostProcess parity。
    - 优先看 OpenCV contour hierarchy、approxPolyDP 对真实 seal 的差异、pyclipper round unclip 对照。
    - 保持轻量原则，必要时增加可解释近似和文档边界。

2. 按模型家族补 object detection raw head decode。
    - 保留 `[N,6]` 简单路径。
    - GFL/PicoDet/RT-DETR raw heads 分家实现，不能用模糊 fallback 猜。
    - 不认识的 tensor layout 继续 fail-fast。

3. 做 table OCR-to-cell / HTML recovery。
    - 基础输入：SLANet structure + cell bbox + OCR 文本框已具备。
    - 后续扩展 row/col span recovery、table cell detector 结果融合和 table recognition v2 编排。

4. 处理 formula 分发。
    - 把 PP-FormulaNet 系列 ONNX/tokenizer 做成可下载资产说明。
    - 保留本地转换流程作为 reproducibility 文档，不作为普通用户主路径。

5. 扩 OCR decoder 覆盖面。
    - 先补官方仍常见且能拿到资产验证的 decoder。
    - 非 CTC decoder 必须从官方配置和真实输出 shape 推导。

6. examples 和 README 只随能力补齐。
    - 新模块必须像用户项目嵌入：加载 ONNX、字典/label、选择 preset/service、传入像素、处理输出。
    - 不新增内部 CLI 风格示例。
