import * as ort from "onnxruntime-node";
import { TableRecognitionV2Service, type TableRecognitionV2TableType } from "../../../src/index.ts";
import { loadPngImage, modelPath, readRequiredFile, toArrayBuffer } from "../../_shared.ts";

const tableType = parseTableType(process.env.PADDLEOCR_TABLE_TYPE);
const imagePath =
    process.env.PADDLEOCR_TABLE_IMAGE ??
    (tableType === "wired"
        ? "examples/input/table_recognition.png"
        : "examples/input/table_wireless.png");
const structureModel = modelPath(
    tableType === "wired" ? "slanext_wired" : "slanext_wireless",
    tableType === "wired" ? "SLANeXt_wired_infer.onnx" : "SLANeXt_wireless_infer.onnx"
);
const cellModel = modelPath(
    tableType === "wired" ? "rt_detr_wired_table_cell_det" : "rt_detr_wireless_table_cell_det",
    tableType === "wired"
        ? "RT-DETR-L_wired_table_cell_det_infer.onnx"
        : "RT-DETR-L_wireless_table_cell_det_infer.onnx"
);
const structurePreset = tableType === "wired" ? "SLANeXt_wired" : "SLANeXt_wireless";
const cellPreset =
    tableType === "wired" ? "RT-DETR-L_wired_table_cell_det" : "RT-DETR-L_wireless_table_cell_det";

function parseTableType(value: string | undefined): TableRecognitionV2TableType {
    if (!value) {
        return "wired";
    }
    if (value === "wired" || value === "wireless") {
        return value;
    }
    throw new Error(`Unsupported PADDLEOCR_TABLE_TYPE: ${value}. Expected wired or wireless.`);
}

const [image, structureModelBuffer, cellModelBuffer] = await Promise.all([
    loadPngImage(imagePath),
    readRequiredFile(structureModel, `Download the ${structurePreset} model first.`),
    readRequiredFile(cellModel, `Download the ${cellPreset} model first.`),
]);
const recognizer =
    tableType === "wired"
        ? await TableRecognitionV2Service.createInstance({
              ort,
              wiredTableStructure: {
                  modelBuffer: toArrayBuffer(structureModelBuffer),
                  preset: "SLANeXt_wired",
              },
              wiredTableCellsDetection: {
                  modelBuffer: toArrayBuffer(cellModelBuffer),
                  preset: "RT-DETR-L_wired_table_cell_det",
              },
              options: {
                  tableClassification: { enabled: false },
                  ocr: { enabled: false },
              },
          })
        : await TableRecognitionV2Service.createInstance({
              ort,
              wirelessTableStructure: {
                  modelBuffer: toArrayBuffer(structureModelBuffer),
                  preset: "SLANeXt_wireless",
              },
              wirelessTableCellsDetection: {
                  modelBuffer: toArrayBuffer(cellModelBuffer),
                  preset: "RT-DETR-L_wireless_table_cell_det",
              },
              options: {
                  tableClassification: { enabled: false },
                  ocr: { enabled: false },
              },
          });
const result = await recognizer.run(image, {
    tableType,
    useWiredTableCellsTransToHtml: tableType === "wired",
    useWirelessTableCellsTransToHtml: tableType === "wireless",
});

console.dir(
    {
        imagePath,
        tableType: result.tableType,
        structureLength: result.structure?.structure.length,
        cellBoxCount: result.cellBoxList.length,
        recoveredCellCount: result.cells.length,
        predHtmlPreview: result.predHtml.slice(0, 260),
    },
    { depth: null }
);
