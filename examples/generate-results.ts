import { execFile } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import * as ort from "onnxruntime-node";
import {
    type Box,
    createFormulaTokenizerVocabulary,
    DetectionService,
    type FormulaRecognitionPresetName,
    FormulaRecognitionService,
    getFormulaRecognitionPresetOptions,
    getImageClassificationPresetOptions,
    getObjectDetectionPresetOptions,
    getTableStructureRecognitionPresetOptions,
    getTextDetectionPresetOptions,
    getTextImageUnwarpingPresetOptions,
    getTextRecognitionPresetOptions,
    Image,
    ImageClassificationService,
    ObjectDetectionService,
    type PaddleOcrModelPresetName,
    type PaddleOcrProgressEvent,
    PaddleOcrService,
    PaddleStructureService,
    RecognitionService,
    TableRecognitionV2Service,
    TableStructureRecognitionService,
    TextImageUnwarpingService,
} from "../src/index.ts";
import {
    createSession,
    formulaTokenizerPath,
    loadJson,
    loadPngImage,
    loadPresetDictionary,
    modelPath,
    readRequiredFile,
    toArrayBuffer,
} from "./_shared.ts";
import {
    annotateImage,
    composePanels,
    createAnnotationBand,
    createSummaryPanel,
    createTableGridPanel,
    drawBoxes,
    drawCellBoxes,
    drawObjectBoxes,
    type RgbImage,
    savePng,
    toRgbImage,
} from "./_visualize.ts";

const RESULT_DIR = "examples/result";
const execFileAsync = promisify(execFile);
const unicodeFontPath =
    process.env.EXAMPLE_RESULT_FONT ?? "/System/Library/Fonts/Supplemental/Arial Unicode.ttf";
const modelPreset: PaddleOcrModelPresetName = "PP-OCRv6_small";
const tasks = [
    ["module-document-orientation", runDocumentOrientation],
    ["module-textline-orientation", runTextlineOrientation],
    ["module-table-classification", runTableClassification],
    ["module-text-detection", runTextDetection],
    ["module-text-recognition", runTextRecognition],
    ["module-seal-text-detection", runSealTextDetection],
    ["module-text-image-unwarping", runTextImageUnwarping],
    ["module-region-detection", runRegionDetection],
    ["module-layout-detection", runLayoutDetection],
    ["module-table-cell-detection", runTableCellDetection],
    ["module-table-structure", runTableStructure],
    ["module-formula-recognition", runFormulaRecognition],
    ["pipeline-ocr", runOcrPipeline],
    ["pipeline-table-recognition-v2", runTableRecognitionV2],
    ["pipeline-pp-structure", runPpStructure],
] as const;

await mkdir(RESULT_DIR, { recursive: true });
const selectedTasks = selectTasks();
for (const [name, run] of selectedTasks) {
    console.log(`[examples] start ${name}`);
    const started = Date.now();
    await run();
    console.log(`[examples] done ${name} in ${((Date.now() - started) / 1000).toFixed(1)}s`);
}

async function runDocumentOrientation() {
    const imagePath = "examples/input/doc_test_rot90ccw.png";
    const referencePath = "examples/input/doc_test.png";
    const image = await loadPngImage(imagePath);
    const reference = await loadPngImage(referencePath);
    const session = await createSession(
        ort,
        modelPath("pp_lcnet_x1_0_doc_ori", "PP-LCNet_x1_0_doc_ori_infer.onnx"),
        "Download the document orientation model first."
    );
    const classifier = new ImageClassificationService(
        ort,
        session,
        getImageClassificationPresetOptions("PP-LCNet_x1_0_doc_ori")
    );
    const result = await classifier.run(image);
    const top = result[0];
    const angle = parseOrientationAngle(top?.label ?? "");
    const corrected = rotateRgbImage(toRgbImage(image), angle);
    await saveResult("module-document-orientation.png", [
        { title: "INPUT", image: toRgbImage(image) },
        { title: "OUTPUT", image: corrected },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: `ANGLE ${angle}` },
                    { text: `LABEL ${top?.label ?? "UNKNOWN"}` },
                    { text: `SCORE ${formatScore(top?.score)}` },
                    ...result.slice(0, 4).map((item) => ({
                        text: `${item.label} ${formatScore(item.score)}`,
                    })),
                    { text: `REFERENCE ${reference.width}X${reference.height}` },
                ],
                720,
                520,
                "CLASSIFIER"
            ),
        },
    ]);
}

async function runTextlineOrientation() {
    const imagePath = "examples/input/textline_rot180_demo.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("pp_lcnet_x0_25_textline_ori", "PP-LCNet_x0_25_textline_ori_infer.onnx"),
        "Download the textline orientation model first."
    );
    const classifier = new ImageClassificationService(
        ort,
        session,
        getImageClassificationPresetOptions("PP-LCNet_x0_25_textline_ori")
    );
    const result = await classifier.run(image);
    const top = result[0];
    await saveResult("module-textline-orientation.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: annotateImage(toRgbImage(image), [
                { text: `${top?.label ?? "UNKNOWN"}` },
                { text: `SCORE ${formatScore(top?.score)}` },
            ]),
        },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: "CLASSIFIER PP-LCNET X0 25" },
                    { text: "TARGET TEXTLINE ORIENTATION" },
                    ...result.slice(0, 4).map((item) => ({
                        text: `${item.label} ${formatScore(item.score)}`,
                    })),
                ],
                720,
                520,
                "CLASSIFIER"
            ),
        },
    ]);
}

async function runTableClassification() {
    const imagePath = "examples/input/table_wireless.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("pp_lcnet_x1_0_table_cls", "PP-LCNet_x1_0_table_cls_infer.onnx"),
        "Download the table classification model first."
    );
    const classifier = new ImageClassificationService(
        ort,
        session,
        getImageClassificationPresetOptions("PP-LCNet_x1_0_table_cls")
    );
    const result = await classifier.run(image);
    await saveResult("module-table-classification.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: annotateImage(toRgbImage(image), [
                { text: `${result[0]?.label ?? "UNKNOWN"}` },
                { text: `SCORE ${formatScore(result[0]?.score)}` },
            ]),
        },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                result.slice(0, 4).map((item, index) => ({
                    text: `#${index + 1} ${item.label} ${formatScore(item.score)}`,
                })),
                720,
                520,
                "CLASSIFIER"
            ),
        },
    ]);
}

async function runTextDetection() {
    const imagePath = "examples/input/general_ocr_001.png";
    const image = await loadPngImage(imagePath);
    const detector = await createTextDetector("ppocr_v6_small", "PP-OCRv6_small_det_infer.onnx");
    const boxes = await detector.run(image, {
        onProgress: logOcrProgress("module-text-detection"),
    });
    await saveResult("module-text-detection.png", [
        { title: "INPUT", image: toRgbImage(image) },
        { title: "OUTPUT", image: drawBoxes(toRgbImage(image), boxes) },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(boxDetailLines(boxes), 720, 520, "TEXT BOXES"),
        },
    ]);
}

async function runTextRecognition() {
    const imagePath = "examples/input/general_ocr_rec_001.png";
    const image = await loadPngImage(imagePath);
    const recognizer = await createRecognizer();
    const fullImageBox: Box = { x: 0, y: 0, width: image.width, height: image.height };
    const result = await recognizer.run(image, [fullImageBox]);
    const text = result[0]?.text ?? "";
    await saveResult("module-text-recognition.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: await drawUnicodeLines(createAnnotationBand(toRgbImage(image), 2), [
                `TEXT ${text}`,
                `CONF ${formatScore(result[0]?.confidence)}`,
            ]),
        },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: "FULL IMAGE AS ONE TEXT BOX" },
                    { text: "CTC LABEL DECODED WITH PP-OCRV6 DICT" },
                    { text: `CHARS ${Array.from(text).length}` },
                    { text: `WIDTH ${image.width}` },
                    { text: `HEIGHT ${image.height}` },
                ],
                720,
                520,
                "RECOGNITION"
            ),
        },
    ]);
}

async function runSealTextDetection() {
    const imagePath = "examples/input/seal_text_det.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("ppocr_v4_mobile_seal_det", "PP-OCRv4_mobile_seal_det_infer.onnx"),
        "Download the PP-OCRv4 seal detection model first."
    );
    const detector = new DetectionService(
        ort,
        session,
        getTextDetectionPresetOptions("PP-OCRv4_mobile_seal_det")
    );
    const boxes = await detector.run(image, {
        onProgress: logOcrProgress("module-seal-text-detection"),
    });
    await saveResult("module-seal-text-detection.png", [
        { title: "INPUT", image: toRgbImage(image) },
        { title: "OUTPUT", image: drawBoxes(toRgbImage(image), boxes) },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(boxDetailLines(boxes), 720, 520, "SEAL POLYGONS"),
        },
    ]);
}

async function runTextImageUnwarping() {
    const imagePath = "examples/input/distorted_document.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("uvdoc", "UVDoc_infer.onnx"),
        "Download the UVDoc model first."
    );
    const unwarper = new TextImageUnwarpingService(
        ort,
        session,
        getTextImageUnwarpingPresetOptions("UVDoc")
    );
    const result = await unwarper.run(image);
    await saveResult("module-text-image-unwarping.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: toRgbImage({ ...result.doctrImage, data: result.doctrImage.data }),
        },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: `INPUT ${image.width}X${image.height}` },
                    { text: `OUTPUT ${result.doctrImage.width}X${result.doctrImage.height}` },
                    { text: "MODULE TEXT IMAGE UNWARPING" },
                    { text: "MODEL UVDOC" },
                    { text: "PURPOSE DEWARP CURVED DOCUMENT PHOTOS" },
                ],
                720,
                520,
                "UVDOC"
            ),
        },
    ]);
}

async function runRegionDetection() {
    const imagePath = "examples/input/layout.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("pp_docblocklayout", "PP-DocBlockLayout_infer.onnx"),
        "Download the PP-DocBlockLayout model first."
    );
    const detector = new ObjectDetectionService(
        ort,
        session,
        getObjectDetectionPresetOptions("PP-DocBlockLayout")
    );
    const boxes = await detector.run(image);
    await saveResult("module-region-detection.png", [
        { title: "INPUT", image: toRgbImage(image) },
        { title: "OUTPUT", image: drawObjectBoxes(toRgbImage(image), boxes) },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(objectBoxDetailLines(boxes), 720, 520, "REGIONS"),
        },
    ]);
}

async function runLayoutDetection() {
    const imagePath = "examples/input/layout.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("pp_doclayout_plus_l", "PP-DocLayout_plus-L_infer.onnx"),
        "Download the PP-DocLayout_plus-L model first."
    );
    const detector = new ObjectDetectionService(
        ort,
        session,
        getObjectDetectionPresetOptions("PP-DocLayout_plus-L")
    );
    const boxes = await detector.run(image);
    await saveResult("module-layout-detection.png", [
        { title: "INPUT", image: toRgbImage(image) },
        { title: "OUTPUT", image: drawObjectBoxes(toRgbImage(image), boxes) },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(objectBoxDetailLines(boxes), 720, 520, "LAYOUT BOXES"),
        },
    ]);
}

async function runTableCellDetection() {
    const imagePath = "examples/input/table_recognition.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("rt_detr_wired_table_cell_det", "RT-DETR-L_wired_table_cell_det_infer.onnx"),
        "Download the wired table cell detection model first."
    );
    const detector = new ObjectDetectionService(
        ort,
        session,
        getObjectDetectionPresetOptions("RT-DETR-L_wired_table_cell_det")
    );
    const boxes = await detector.run(image);
    await saveResult("module-table-cell-detection.png", [
        { title: "INPUT", image: toRgbImage(image) },
        { title: "OUTPUT", image: drawObjectBoxes(toRgbImage(image), boxes) },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(objectBoxDetailLines(boxes), 720, 520, "CELL BOXES"),
        },
    ]);
}

async function runTableStructure() {
    const imagePath = "examples/input/table_recognition.png";
    const image = await loadPngImage(imagePath);
    const session = await createSession(
        ort,
        modelPath("slanet", "SLANet_infer.onnx"),
        "Download the SLANet model first."
    );
    const recognizer = new TableStructureRecognitionService(
        ort,
        session,
        getTableStructureRecognitionPresetOptions("SLANet")
    );
    const result = await recognizer.run(image);
    await saveResult("module-table-structure.png", [
        { title: "INPUT", image: toRgbImage(image) },
        { title: "OUTPUT", image: createTableGridPanel(result.html) },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: `PRESET SLANET` },
                    { text: `STRUCTURE TOKENS ${result.structure.length}` },
                    { text: `CELL BBOX ${result.bbox.length}` },
                    { text: `SCORE ${formatScore(result.structureScore)}` },
                    ...result.structure.slice(0, 10).map((token, index) => ({
                        text: `T${index + 1} ${token}`,
                    })),
                    ...result.bbox.slice(0, 4).map((box, index) => ({
                        text: `B${index + 1} ${bboxText(box)}`,
                    })),
                ],
                720,
                520,
                "STRUCTURE"
            ),
        },
    ]);
}

async function runFormulaRecognition() {
    const imagePath = "examples/input/general_formula_rec_001.png";
    const image = await loadPngImage(imagePath);
    const preset: FormulaRecognitionPresetName = "PP-FormulaNet_plus-M";
    const session = await createSession(
        ort,
        modelPath("pp_formulanet_plus_m", "PP-FormulaNet_plus-M_infer.onnx"),
        `Download the ${preset} formula model first.`
    );
    const tokenizerJson = await loadJson(
        formulaTokenizerPath,
        "Prepare the official UniMERNet tokenizer JSON first."
    );
    const recognizer = new FormulaRecognitionService(ort, session, {
        ...getFormulaRecognitionPresetOptions(preset),
        tokenizerVocabulary: createFormulaTokenizerVocabulary(tokenizerJson),
    });
    const result = await recognizer.run(image);
    await saveResult("module-formula-recognition.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: await drawUnicodeLines(
                createSummaryPanel([], 720, 520, "LATEX"),
                [`$${result.formula}$`],
                { x: 28, y: 92, fontSize: 20, lineSpacing: 8 }
            ),
        },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: `MODEL ${preset.replaceAll("-", "_")}` },
                    { text: `TOKENS ${result.tokens.length}` },
                    { text: `TOKEN IDS ${result.tokenIds.length}` },
                    { text: `FORMULA CHARS ${result.formula.length}` },
                    { text: "TOKENIZER UNIMERNET BYTE LEVEL" },
                    { text: `PREVIEW ${sanitizeAscii(result.formula).slice(0, 46)}` },
                ],
                720,
                520,
                "FORMULA META"
            ),
        },
    ]);
}

async function runOcrPipeline() {
    const imagePath = "examples/input/street_net_bar.png";
    const image = await loadPngImage(imagePath);
    const ocr = await createOcrService();
    const results = await ocr.recognize(image, {
        onProgress: logOcrProgress("pipeline-ocr"),
    });
    const processed = ocr.processRecognition(results);
    await saveResult("pipeline-ocr.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: await drawUnicodeLabels(
                await drawUnicodeLines(
                    createAnnotationBand(
                        drawBoxes(
                            toRgbImage(image),
                            results.map((result) => result.box)
                        ),
                        2
                    ),
                    [`TEXT ${processed.text}`, `CONF ${formatScore(processed.confidence)}`]
                ),
                results.map((result) => ({
                    text: result.text,
                    x: result.box.x,
                    y: Math.max(0, result.box.y - 24),
                }))
            ),
        },
        {
            title: "INTERMEDIATE",
            image: await drawUnicodeLines(
                createSummaryPanel(
                    [
                        { text: `TEXT BOXES ${results.length}` },
                        { text: `CONF ${formatScore(processed.confidence)}` },
                    ],
                    720,
                    520,
                    "OCR TEXT"
                ),
                [`TEXT ${processed.text}`],
                { x: 28, y: 176, fontSize: 22, lineSpacing: 8 }
            ),
        },
    ]);
}

async function runTableRecognitionV2() {
    const imagePath = "examples/input/table_recognition.png";
    const image = await loadPngImage(imagePath);
    const [
        structureModelBuffer,
        cellModelBuffer,
        detectionModelBuffer,
        recognitionModelBuffer,
        textlineOrientationModelBuffer,
        charactersDictionary,
    ] = await Promise.all([
        readRequiredFile(
            modelPath("slanext_wired", "SLANeXt_wired_infer.onnx"),
            "Download the SLANeXt wired model first."
        ),
        readRequiredFile(
            modelPath("rt_detr_wired_table_cell_det", "RT-DETR-L_wired_table_cell_det_infer.onnx"),
            "Download the wired cell detection model first."
        ),
        readRequiredFile(
            modelPath("ppocr_v6_small", "PP-OCRv6_small_det_infer.onnx"),
            "Download the PP-OCRv6 detection model first."
        ),
        readRequiredFile(
            modelPath("ppocr_v6_small", "PP-OCRv6_small_rec_infer.onnx"),
            "Download the PP-OCRv6 recognition model first."
        ),
        readRequiredFile(
            modelPath("pp_lcnet_x0_25_textline_ori", "PP-LCNet_x0_25_textline_ori_infer.onnx"),
            "Download the textline orientation model first."
        ),
        loadPresetDictionary(modelPath("ppocr_v6_small", "ppocrv6_dict.txt"), modelPreset),
    ]);
    const recognizer = await TableRecognitionV2Service.createInstance({
        ort,
        wiredTableStructure: {
            modelBuffer: toArrayBuffer(structureModelBuffer),
            preset: "SLANeXt_wired",
        },
        wiredTableCellsDetection: {
            modelBuffer: toArrayBuffer(cellModelBuffer),
            preset: "RT-DETR-L_wired_table_cell_det",
        },
        ocr: {
            modelPreset,
            detection: {
                modelBuffer: toArrayBuffer(detectionModelBuffer),
            },
            recognition: {
                modelBuffer: toArrayBuffer(recognitionModelBuffer),
                charactersDictionary,
            },
            textlineOrientation: {
                modelBuffer: toArrayBuffer(textlineOrientationModelBuffer),
                threshold: 0.9,
            },
        },
        options: {
            tableClassification: { enabled: false },
            ocr: { enabled: true },
        },
    });
    const result = await recognizer.run(image, {
        tableType: "wired",
        useWiredTableCellsTransToHtml: true,
        ocr: {
            onProgress: logOcrProgress("pipeline-table-recognition-v2/ocr"),
        },
    });
    await saveResult("pipeline-table-recognition-v2.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: await drawUnicodeLabels(
                drawCellBoxes(toRgbImage(image), result.cellBoxList),
                (result.ocr ?? []).map((item) => ({
                    text: item.text,
                    x: item.box.x,
                    y: Math.max(0, item.box.y - 18),
                    fontSize: 18,
                }))
            ),
        },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: `TABLE TYPE ${result.tableType}` },
                    { text: `STRUCTURE TOKENS ${result.structure?.structure.length ?? 0}` },
                    { text: `CELL BOXES ${result.cellBoxList.length}` },
                    { text: `RECOVERED CELLS ${result.cells.length}` },
                    { text: `OCR LINES ${result.ocr?.length ?? 0}` },
                    { text: `HTML ${sanitizeAscii(result.predHtml).slice(0, 48)}` },
                ],
                720,
                520,
                "TABLE V2"
            ),
        },
    ]);
}

async function runPpStructure() {
    const imagePath = "examples/input/layout.png";
    const image = await loadPngImage(imagePath);
    const [
        documentOrientationModel,
        unwarpingModel,
        regionDetectionModel,
        layoutModelBuffer,
        detectionModelBuffer,
        recognitionModelBuffer,
        textlineOrientationModelBuffer,
        charactersDictionary,
        tableStructureModel,
    ] = await Promise.all([
        readRequiredFile(
            modelPath("pp_lcnet_x1_0_doc_ori", "PP-LCNet_x1_0_doc_ori_infer.onnx"),
            "Download the document orientation model first."
        ),
        readRequiredFile(modelPath("uvdoc", "UVDoc_infer.onnx"), "Download the UVDoc model first."),
        readRequiredFile(
            modelPath("pp_docblocklayout", "PP-DocBlockLayout_infer.onnx"),
            "Download the PP-DocBlockLayout region model first."
        ),
        readRequiredFile(
            modelPath("pp_doclayout_plus_l", "PP-DocLayout_plus-L_infer.onnx"),
            "Download the PP-DocLayout_plus-L model first."
        ),
        readRequiredFile(
            modelPath("ppocr_v6_small", "PP-OCRv6_small_det_infer.onnx"),
            "Download the PP-OCRv6 detection model first."
        ),
        readRequiredFile(
            modelPath("ppocr_v6_small", "PP-OCRv6_small_rec_infer.onnx"),
            "Download the PP-OCRv6 recognition model first."
        ),
        readRequiredFile(
            modelPath("pp_lcnet_x0_25_textline_ori", "PP-LCNet_x0_25_textline_ori_infer.onnx"),
            "Download the textline orientation model first."
        ),
        loadPresetDictionary(modelPath("ppocr_v6_small", "ppocrv6_dict.txt"), modelPreset),
        readRequiredFile(modelPath("slanet", "SLANet_infer.onnx"), "Download SLANet first."),
    ]);
    const structure = await PaddleStructureService.createInstance({
        ort,
        documentOrientation: {
            modelBuffer: toArrayBuffer(documentOrientationModel),
        },
        textImageUnwarping: {
            modelBuffer: toArrayBuffer(unwarpingModel),
        },
        regionDetection: {
            modelBuffer: toArrayBuffer(regionDetectionModel),
        },
        layout: {
            modelBuffer: toArrayBuffer(layoutModelBuffer),
            preset: "PP-DocLayout_plus-L",
        },
        ocr: {
            modelPreset,
            detection: {
                modelBuffer: toArrayBuffer(detectionModelBuffer),
            },
            recognition: {
                modelBuffer: toArrayBuffer(recognitionModelBuffer),
                charactersDictionary,
            },
            textlineOrientation: {
                modelBuffer: toArrayBuffer(textlineOrientationModelBuffer),
                threshold: 0.9,
            },
        },
        tableStructure: {
            modelBuffer: toArrayBuffer(tableStructureModel),
        },
    });
    const result = await structure.run(image, {
        layout: {
            fallbackRegionType: false,
        },
        ocr: {
            onProgress: logOcrProgress("pipeline-pp-structure/ocr"),
        },
        formula: {
            enabled: false,
        },
    });
    const layoutBoxes = result.regions.map((region) => ({
        classId: 0,
        label: region.label,
        score: region.score,
        coordinate: region.bbox,
    }));
    const tableCells = result.regions.flatMap(
        (region) =>
            region.table?.structure?.bbox.map((cell) => offsetCellBox(cell, region.bbox)) ?? []
    );
    const regionLabels = result.regions.slice(0, 16).map((region) => ({
        text: regionLabelText(region),
        x: region.bbox[0],
        y: Math.max(0, region.bbox[1] - 20),
        fontSize: 15,
    }));
    const structureOutput = drawCellBoxes(
        drawObjectBoxes(toRgbImage(result.image), layoutBoxes),
        tableCells
    );
    await saveResult("pipeline-pp-structure.png", [
        { title: "INPUT", image: toRgbImage(image) },
        {
            title: "OUTPUT",
            image: await drawUnicodeLabels(structureOutput, regionLabels),
        },
        {
            title: "INTERMEDIATE",
            image: createSummaryPanel(
                [
                    { text: `FINAL IMAGE ${result.image.width}X${result.image.height}` },
                    { text: `REGIONS ${result.regions.length}` },
                    { text: `PAGE OCR ${result.stages.ocr.result?.length ?? 0}` },
                    { text: `TABLE CELLS ${tableCells.length}` },
                    { text: `MARKDOWN ${result.markdown?.text.length ?? 0} CHARS` },
                    ...result.regions.slice(0, 9).map((region, index) => ({
                        text: `#${index + 1} ${region.type} ${bboxText(region.bbox)}`,
                    })),
                ],
                720,
                520,
                "STRUCTURE"
            ),
        },
    ]);
}

async function createTextDetector(dir: string, file: string): Promise<DetectionService> {
    const session = await createSession(
        ort,
        modelPath(dir, file),
        "Download the text detection model first."
    );
    return new DetectionService(ort, session, getTextDetectionPresetOptions("PP-OCRv6_small_det"));
}

async function createRecognizer(): Promise<RecognitionService> {
    const [session, charactersDictionary] = await Promise.all([
        createSession(
            ort,
            modelPath("ppocr_v6_small", "PP-OCRv6_small_rec_infer.onnx"),
            "Download the PP-OCRv6 small recognition model first."
        ),
        loadPresetDictionary(modelPath("ppocr_v6_small", "ppocrv6_dict.txt"), modelPreset),
    ]);
    return new RecognitionService(ort, session, {
        ...getTextRecognitionPresetOptions("PP-OCRv6_small_rec"),
        charactersDictionary,
    });
}

async function createOcrService(): Promise<PaddleOcrService> {
    const [
        detectionModelBuffer,
        recognitionModelBuffer,
        textlineOrientationModelBuffer,
        charactersDictionary,
    ] = await Promise.all([
        readRequiredFile(
            modelPath("ppocr_v6_small", "PP-OCRv6_small_det_infer.onnx"),
            "Download the PP-OCRv6 detection model first."
        ),
        readRequiredFile(
            modelPath("ppocr_v6_small", "PP-OCRv6_small_rec_infer.onnx"),
            "Download the PP-OCRv6 recognition model first."
        ),
        readRequiredFile(
            modelPath("pp_lcnet_x0_25_textline_ori", "PP-LCNet_x0_25_textline_ori_infer.onnx"),
            "Download the textline orientation model first."
        ),
        loadPresetDictionary(modelPath("ppocr_v6_small", "ppocrv6_dict.txt"), modelPreset),
    ]);
    return PaddleOcrService.createInstance({
        ort,
        modelPreset,
        detection: {
            modelBuffer: toArrayBuffer(detectionModelBuffer),
        },
        recognition: {
            modelBuffer: toArrayBuffer(recognitionModelBuffer),
            charactersDictionary,
        },
        textlineOrientation: {
            modelBuffer: toArrayBuffer(textlineOrientationModelBuffer),
            threshold: 0.9,
        },
    });
}

async function saveResult(fileName: string, panels: Parameters<typeof composePanels>[0]) {
    const image = composePanels(panels);
    await savePng(`${RESULT_DIR}/${fileName}`, image);
    console.log(`wrote ${RESULT_DIR}/${fileName}`);
}

function selectTasks(): typeof tasks {
    const onlyValues = parseOnlyValues();
    if (!onlyValues.size) {
        return tasks;
    }
    const selected = tasks.filter(([name]) => onlyValues.has(name));
    const missing = [...onlyValues].filter(
        (name) => !tasks.some(([taskName]) => taskName === name)
    );
    if (missing.length) {
        throw new Error(
            `Unknown example task(s): ${missing.join(", ")}. Available tasks: ${tasks
                .map(([name]) => name)
                .join(", ")}.`
        );
    }
    return selected as typeof tasks;
}

function parseOnlyValues(): Set<string> {
    const values: string[] = [];
    const args = process.argv.slice(2);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === "--only") {
            const value = args[index + 1];
            if (!value) {
                throw new Error("--only requires a comma-separated task name list.");
            }
            values.push(value);
            index += 1;
        } else if (arg?.startsWith("--only=")) {
            values.push(arg.slice("--only=".length));
        }
    }
    if (process.env.EXAMPLE_ONLY) {
        values.push(process.env.EXAMPLE_ONLY);
    }
    return new Set(
        values
            .flatMap((value) => value.split(","))
            .map((value) => value.trim())
            .filter(Boolean)
    );
}

function logOcrProgress(scope: string) {
    return (event: PaddleOcrProgressEvent) => {
        if (event.type === "det") {
            const detected =
                event.detectedCount === undefined ? "" : ` detected=${event.detectedCount}`;
            console.log(
                `[${scope}] det ${event.stage} ${event.progress.current}/${event.progress.total}${detected}`
            );
            return;
        }
        const suffix = event.result?.text ? ` text="${sanitizeAscii(event.result.text)}"` : "";
        console.log(
            `[${scope}] rec ${event.stage} ${event.progress.current}/${event.progress.total}${suffix}`
        );
    };
}

function parseOrientationAngle(label: string): number {
    const match = label.match(/(?:^|[^0-9])(90|180|270)(?:[^0-9]|$)/);
    return match ? Number(match[1]) : 0;
}

function rotateRgbImage(image: RgbImage, angle: number): RgbImage {
    const runtimeImage = new Image(image.width, image.height, 3, image.data);
    if (angle === 90) {
        return toRgbImage(runtimeImage.rotateCounterClockwise());
    }
    if (angle === 180) {
        return toRgbImage(runtimeImage.rotate180());
    }
    if (angle === 270) {
        return toRgbImage(runtimeImage.rotateClockwise());
    }
    return image;
}

function boxDetailLines(boxes: Box[]) {
    return [
        { text: `COUNT ${boxes.length}` },
        ...boxes.slice(0, 12).map((box, index) => ({
            text: `#${index + 1} ${boxPointsText(box)}`,
        })),
    ];
}

function objectBoxDetailLines(
    boxes: Array<{ label: string; score: number; coordinate: readonly number[] }>
) {
    return [
        { text: `COUNT ${boxes.length}` },
        ...boxes.slice(0, 12).map((box, index) => ({
            text: `#${index + 1} ${box.label} ${formatScore(box.score)} ${bboxText(box.coordinate)}`,
        })),
    ];
}

function bboxText(values: readonly number[]): string {
    const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = values;
    return `[${formatCoord(x1)},${formatCoord(y1)},${formatCoord(x2)},${formatCoord(y2)}]`;
}

function boxPointsText(box: Box): string {
    const points = box.polygon ?? box.points;
    if (points?.length) {
        const text = points
            .slice(0, 4)
            .map((point) => `${formatCoord(point.x)},${formatCoord(point.y)}`)
            .join(" ");
        return `[${text}]`;
    }
    return bboxText([box.x, box.y, box.x + box.width, box.y + box.height]);
}

function offsetCellBox(
    cell: readonly number[],
    region: readonly [number, number, number, number]
): number[] {
    const [offsetX, offsetY] = region;
    if (cell.length >= 8) {
        return cell.map((value, index) => value + (index % 2 === 0 ? offsetX : offsetY));
    }
    if (cell.length >= 4) {
        const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = cell;
        return [x1 + offsetX, y1 + offsetY, x2 + offsetX, y2 + offsetY];
    }
    return [...cell];
}

function regionLabelText(region: {
    type: string;
    ocr?: Array<{ text: string }>;
    table?: { structure?: { bbox: readonly unknown[] } };
    formula?: { formula: string };
    seal?: { boxes: readonly unknown[]; recognition: Array<{ text: string }> };
}): string {
    if (region.table) {
        return `TABLE CELLS ${region.table.structure?.bbox.length ?? 0}`;
    }
    if (region.formula) {
        return `FORMULA ${region.formula.formula.slice(0, 28)}`;
    }
    if (region.seal) {
        return `SEAL ${region.seal.boxes.length}`;
    }
    const text = region.ocr
        ?.map((item) => item.text)
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();
    return text ? `${region.type.toUpperCase()} ${text.slice(0, 36)}` : region.type.toUpperCase();
}

async function drawUnicodeLines(
    image: RgbImage,
    lines: string[],
    options: {
        x?: number;
        y?: number;
        fontSize?: number;
        lineSpacing?: number;
        color?: string;
    } = {}
): Promise<RgbImage> {
    if (!lines.length) {
        return image;
    }
    const wrapped = lines.flatMap((line) => wrapUnicodeText(line, 46));
    return drawUnicodeTextFiles(image, [
        {
            text: wrapped.join("\n"),
            x: options.x ?? 18,
            y: options.y ?? 18,
            fontSize: options.fontSize ?? 24,
            lineSpacing: options.lineSpacing ?? 8,
            color: options.color ?? "0x1c222d",
        },
    ]);
}

async function drawUnicodeLabels(
    image: RgbImage,
    labels: Array<{
        text: string;
        x: number;
        y: number;
        fontSize?: number;
        color?: string;
    }>
): Promise<RgbImage> {
    if (!labels.length) {
        return image;
    }
    return drawUnicodeTextFiles(
        image,
        labels.slice(0, 24).map((label) => ({
            text: label.text,
            x: Math.max(0, Math.round(label.x)),
            y: Math.max(0, Math.round(label.y)),
            fontSize: label.fontSize ?? 20,
            lineSpacing: 4,
            color: label.color ?? "0xd9447c",
            box: true,
        }))
    );
}

async function drawUnicodeTextFiles(
    image: RgbImage,
    overlays: Array<{
        text: string;
        x: number;
        y: number;
        fontSize: number;
        lineSpacing: number;
        color: string;
        box?: boolean;
    }>
): Promise<RgbImage> {
    const tempDir = await mkdtemp(join(tmpdir(), "paddleocr-js-example-"));
    try {
        const inputPath = join(tempDir, "input.png");
        const outputPath = join(tempDir, "output.png");
        await savePng(inputPath, image);
        const filters: string[] = [];
        for (let index = 0; index < overlays.length; index++) {
            const overlay = overlays[index];
            const textPath = join(tempDir, `text-${index}.txt`);
            await writeFile(textPath, overlay.text);
            filters.push(createDrawTextFilter(overlay, textPath));
        }
        await execFileAsync("ffmpeg", [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            inputPath,
            "-vf",
            filters.join(","),
            "-frames:v",
            "1",
            outputPath,
        ]);
        return toRgbImage(await loadPngImage(outputPath));
    } finally {
        await rm(tempDir, { recursive: true, force: true });
    }
}

function createDrawTextFilter(
    overlay: {
        x: number;
        y: number;
        fontSize: number;
        lineSpacing: number;
        color: string;
        box?: boolean;
    },
    textPath: string
): string {
    const boxOptions = overlay.box ? ":box=1:boxcolor=white@0.78:boxborderw=4" : "";
    return [
        `drawtext=fontfile=${escapeDrawTextValue(unicodeFontPath)}`,
        `textfile=${escapeDrawTextValue(textPath)}`,
        `fontcolor=${overlay.color}`,
        `fontsize=${overlay.fontSize}`,
        `line_spacing=${overlay.lineSpacing}`,
        "expansion=none",
        "fix_bounds=1",
        `x=${overlay.x}`,
        `y=${overlay.y}${boxOptions}`,
    ].join(":");
}

function escapeDrawTextValue(value: string): string {
    return value.replace(/\\/g, "\\\\").replace(/:/g, "\\:").replace(/'/g, "\\'");
}

function wrapUnicodeText(text: string, maxLength: number): string[] {
    if (Array.from(text).length <= maxLength) {
        return [text];
    }
    const lines: string[] = [];
    let current = "";
    for (const part of text.split(/\s+/)) {
        const next = current ? `${current} ${part}` : part;
        if (Array.from(next).length > maxLength && current) {
            lines.push(current);
            current = part;
        } else {
            current = next;
        }
    }
    if (current) {
        lines.push(current);
    }
    return lines;
}

function formatCoord(value: number): string {
    return Number.isFinite(value) ? String(Math.round(value)) : "N/A";
}

function formatScore(value: number | undefined): string {
    return Number.isFinite(value) ? (value as number).toFixed(3) : "N/A";
}

function sanitizeAscii(text: string): string {
    return text
        .replace(/[^\x20-\x7e]/g, "?")
        .replace(/\s+/g, " ")
        .trim();
}
