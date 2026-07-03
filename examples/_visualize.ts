import { writeFile } from "node:fs/promises";
import { encode } from "fast-png";
import {
    type Box,
    Image,
    type ImageInput,
    type ObjectDetectionBox,
    type Point,
} from "../src/index.ts";

export interface RgbImage {
    width: number;
    height: number;
    data: Uint8Array;
}

export interface Panel {
    title: string;
    image: RgbImage;
}

export interface TextLine {
    text: string;
    color?: Color;
}

type Color = [number, number, number];

const WHITE: Color = [255, 255, 255];
const INK: Color = [28, 34, 45];
const MUTED: Color = [94, 107, 126];
const PANEL_BG: Color = [247, 249, 252];
const BORDER: Color = [210, 218, 230];
const BLUE: Color = [45, 116, 255];
const GREEN: Color = [20, 160, 104];
const ORANGE: Color = [242, 142, 28];
const PINK: Color = [222, 68, 124];
const CYAN: Color = [18, 160, 184];
const COLORS: Color[] = [BLUE, GREEN, ORANGE, PINK, CYAN, [132, 76, 214], [215, 72, 56]];

const FONT: Record<string, string[]> = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ";": ["00000", "01100", "01100", "00000", "01100", "01100", "01000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "\\": ["10000", "01000", "00100", "00010", "00001", "00000", "00000"],
    _: ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    $: ["00100", "01111", "10100", "01110", "00101", "11110", "00100"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "&": ["01100", "10010", "10100", "01000", "10101", "10010", "01101"],
    "'": ["01100", "01100", "01000", "00000", "00000", "00000", "00000"],
    '"': ["01010", "01010", "00000", "00000", "00000", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "[": ["01110", "01000", "01000", "01000", "01000", "01000", "01110"],
    "]": ["01110", "00010", "00010", "00010", "00010", "00010", "01110"],
    "{": ["00110", "00100", "00100", "01000", "00100", "00100", "00110"],
    "}": ["01100", "00100", "00100", "00010", "00100", "00100", "01100"],
    "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "*": ["00000", "10101", "01110", "11111", "01110", "10101", "00000"],
    "^": ["00100", "01010", "10001", "00000", "00000", "00000", "00000"],
    "#": ["01010", "11111", "01010", "11111", "01010", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    B: ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    C: ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    D: ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    F: ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    G: ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    H: ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    I: ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    J: ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    K: ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    L: ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    M: ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    N: ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    P: ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    Q: ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    S: ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    V: ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    W: ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    X: ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    Y: ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    Z: ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
};

export function toRgbImage(input: ImageInput): RgbImage {
    const channels = Math.round(input.data.length / (input.width * input.height));
    if (channels === 3) {
        return { width: input.width, height: input.height, data: new Uint8Array(input.data) };
    }
    const data = new Uint8Array(input.width * input.height * 3);
    for (let index = 0; index < input.width * input.height; index++) {
        const source = index * channels;
        const target = index * 3;
        data[target] = input.data[source] ?? 0;
        data[target + 1] = input.data[source + 1] ?? data[target];
        data[target + 2] = input.data[source + 2] ?? data[target];
    }
    return { width: input.width, height: input.height, data };
}

export async function savePng(path: string, image: RgbImage): Promise<void> {
    await writeFile(
        path,
        encode({ width: image.width, height: image.height, channels: 3, data: image.data })
    );
}

export function cloneImage(image: RgbImage): RgbImage {
    return { width: image.width, height: image.height, data: new Uint8Array(image.data) };
}

export function drawBoxes(image: RgbImage, boxes: Box[], color: Color = BLUE): RgbImage {
    const output = cloneImage(image);
    boxes.forEach((box, index) => {
        const boxColor = COLORS[index % COLORS.length] ?? color;
        if (box.polygon?.length) {
            drawPolygon(output, box.polygon, boxColor, 3);
        } else if (box.points?.length) {
            drawPolygon(output, box.points, boxColor, 3);
        } else {
            drawRect(output, box.x, box.y, box.width, box.height, boxColor, 3);
        }
    });
    return output;
}

export function drawObjectBoxes(image: RgbImage, boxes: ObjectDetectionBox[]): RgbImage {
    const output = cloneImage(image);
    boxes.forEach((box, index) => {
        const color = COLORS[index % COLORS.length] ?? BLUE;
        const [x1, y1, x2, y2] = box.coordinate;
        drawRect(output, x1, y1, x2 - x1, y2 - y1, color, 3);
    });
    return output;
}

export function drawCellBoxes(image: RgbImage, cells: number[][], color: Color = GREEN): RgbImage {
    const output = cloneImage(image);
    cells.forEach((cell, index) => {
        const cellColor = COLORS[index % COLORS.length] ?? color;
        if (cell.length >= 8) {
            drawPolygon(
                output,
                [
                    { x: cell[0] ?? 0, y: cell[1] ?? 0 },
                    { x: cell[2] ?? 0, y: cell[3] ?? 0 },
                    { x: cell[4] ?? 0, y: cell[5] ?? 0 },
                    { x: cell[6] ?? 0, y: cell[7] ?? 0 },
                ],
                cellColor,
                2
            );
        } else if (cell.length >= 4) {
            const [x1, y1, x2, y2] = cell;
            drawRect(
                output,
                x1 ?? 0,
                y1 ?? 0,
                (x2 ?? 0) - (x1 ?? 0),
                (y2 ?? 0) - (y1 ?? 0),
                cellColor,
                2
            );
        }
    });
    return output;
}

export function createSummaryPanel(
    lines: TextLine[],
    width = 720,
    height = 520,
    title = "SUMMARY"
): RgbImage {
    const image = createCanvas(width, height, PANEL_BG);
    drawFilledRect(image, 0, 0, width, 58, INK);
    drawText(image, title, 24, 20, WHITE, 3);
    let y = 92;
    for (const line of lines) {
        const wrappedLines = wrapText(line.text, 54);
        for (const wrappedLine of wrappedLines) {
            drawText(image, wrappedLine, 28, y, line.color ?? INK, 2);
            y += 34;
            if (y > height - 34) {
                return image;
            }
        }
        if (y > height - 34) {
            break;
        }
    }
    return image;
}

export function createTableGridPanel(html: string, width = 720, height = 520): RgbImage {
    const image = createCanvas(width, height, PANEL_BG);
    drawFilledRect(image, 0, 0, width, 58, INK);
    drawText(image, "TABLE GRID", 24, 20, WHITE, 3);
    const rows = parseTableRows(html);
    if (!rows.length) {
        drawText(image, "NO TABLE STRUCTURE", 28, 118, MUTED, 2);
        return image;
    }

    const columnCount = Math.max(
        1,
        ...rows.map((row) => row.reduce((total, cell) => total + cell.colspan, 0))
    );
    const gridLeft = 44;
    const gridTop = 96;
    const gridWidth = width - gridLeft * 2;
    const gridHeight = Math.min(height - gridTop - 42, rows.length * 72);
    const columnWidth = gridWidth / columnCount;
    const rowHeight = gridHeight / rows.length;

    rows.forEach((row, rowIndex) => {
        let columnIndex = 0;
        row.forEach((cell) => {
            const x = gridLeft + columnIndex * columnWidth;
            const y = gridTop + rowIndex * rowHeight;
            const cellWidth = columnWidth * cell.colspan;
            drawRect(
                image,
                x,
                y,
                cellWidth,
                rowHeight,
                COLORS[rowIndex % COLORS.length] ?? BLUE,
                2
            );
            if (cell.colspan > 1) {
                drawText(image, `COLSPAN ${cell.colspan}`, x + 10, y + 14, MUTED, 2);
            }
            columnIndex += cell.colspan;
        });
    });

    drawText(image, `ROWS ${rows.length} COLS ${columnCount}`, 44, height - 34, INK, 2);
    return image;
}

export function createCropMosaic(image: RgbImage, boxes: Box[], title: string): RgbImage {
    const panel = createCanvas(720, 520, PANEL_BG);
    drawFilledRect(panel, 0, 0, panel.width, 58, INK);
    drawText(panel, title, 24, 20, WHITE, 3);
    const crops = boxes.slice(0, 8).map((box) => cropBox(image, box));
    const cellWidth = 320;
    const cellHeight = 90;
    crops.forEach((crop, index) => {
        const fitted = fitImage(crop, cellWidth, cellHeight, WHITE);
        const x = 28 + (index % 2) * 346;
        const y = 86 + Math.floor(index / 2) * 104;
        drawRect(panel, x - 2, y - 2, cellWidth + 4, cellHeight + 4, BORDER, 2);
        paste(panel, fitted, x, y);
    });
    if (!crops.length) {
        drawText(panel, "NO CROPS", 28, 118, MUTED, 2);
    }
    return panel;
}

export function composePanels(panels: Panel[]): RgbImage {
    const panelWidth = 520;
    const panelHeight = 680;
    const headerHeight = 52;
    const gap = 22;
    const margin = 28;
    const width = margin * 2 + panels.length * panelWidth + (panels.length - 1) * gap;
    const height = margin * 2 + headerHeight + panelHeight;
    const canvas = createCanvas(width, height, [238, 243, 249]);

    panels.forEach((panel, index) => {
        const x = margin + index * (panelWidth + gap);
        const y = margin;
        drawFilledRect(canvas, x, y, panelWidth, headerHeight + panelHeight, WHITE);
        drawRect(canvas, x, y, panelWidth, headerHeight + panelHeight, BORDER, 2);
        drawFilledRect(canvas, x, y, panelWidth, headerHeight, INK);
        drawText(canvas, panel.title, x + 18, y + 18, WHITE, 2);
        const fitted = fitImage(panel.image, panelWidth - 28, panelHeight - 28, WHITE);
        paste(canvas, fitted, x + 14, y + headerHeight + 14);
    });

    return canvas;
}

export function annotateImage(image: RgbImage, lines: TextLine[]): RgbImage {
    const output = cloneImage(image);
    const bandHeight = Math.min(120, 32 + lines.length * 28);
    drawFilledRect(output, 0, 0, output.width, bandHeight, [255, 255, 255]);
    drawRect(output, 0, 0, output.width, bandHeight, BORDER, 2);
    let y = 18;
    for (const line of lines) {
        drawText(output, line.text, 18, y, line.color ?? INK, 2);
        y += 28;
    }
    return output;
}

export function createAnnotationBand(image: RgbImage, lineCount: number): RgbImage {
    const output = cloneImage(image);
    const bandHeight = Math.min(180, 34 + Math.max(1, lineCount) * 30);
    drawFilledRect(output, 0, 0, output.width, bandHeight, [255, 255, 255]);
    drawRect(output, 0, 0, output.width, bandHeight, BORDER, 2);
    return output;
}

export function cropBox(image: RgbImage, box: Box): RgbImage {
    const x = clamp(Math.floor(box.x), 0, Math.max(0, image.width - 1));
    const y = clamp(Math.floor(box.y), 0, Math.max(0, image.height - 1));
    const width = clamp(Math.ceil(box.width), 1, image.width - x);
    const height = clamp(Math.ceil(box.height), 1, image.height - y);
    return toRgbImage(
        new Image(image.width, image.height, 3, image.data).crop({ x, y, width, height })
    );
}

function fitImage(
    image: RgbImage,
    maxWidth: number,
    maxHeight: number,
    background: Color
): RgbImage {
    const scale = Math.min(maxWidth / image.width, maxHeight / image.height, 1);
    const width = Math.max(1, Math.round(image.width * scale));
    const height = Math.max(1, Math.round(image.height * scale));
    const resized =
        width === image.width && height === image.height
            ? image
            : toRgbImage(
                  new Image(image.width, image.height, 3, image.data).resize({ width, height })
              );
    const canvas = createCanvas(maxWidth, maxHeight, background);
    paste(
        canvas,
        resized,
        Math.floor((maxWidth - width) / 2),
        Math.floor((maxHeight - height) / 2)
    );
    return canvas;
}

function createCanvas(width: number, height: number, color: Color): RgbImage {
    const data = new Uint8Array(width * height * 3);
    for (let index = 0; index < width * height; index++) {
        data[index * 3] = color[0];
        data[index * 3 + 1] = color[1];
        data[index * 3 + 2] = color[2];
    }
    return { width, height, data };
}

function paste(target: RgbImage, source: RgbImage, x: number, y: number): void {
    for (let sy = 0; sy < source.height; sy++) {
        const ty = y + sy;
        if (ty < 0 || ty >= target.height) {
            continue;
        }
        for (let sx = 0; sx < source.width; sx++) {
            const tx = x + sx;
            if (tx < 0 || tx >= target.width) {
                continue;
            }
            const sourceIndex = (sy * source.width + sx) * 3;
            const targetIndex = (ty * target.width + tx) * 3;
            target.data[targetIndex] = source.data[sourceIndex] ?? 0;
            target.data[targetIndex + 1] = source.data[sourceIndex + 1] ?? 0;
            target.data[targetIndex + 2] = source.data[sourceIndex + 2] ?? 0;
        }
    }
}

function drawPolygon(
    image: RgbImage,
    points: readonly Point[],
    color: Color,
    lineWidth: number
): void {
    for (let index = 0; index < points.length; index++) {
        const current = points[index];
        const next = points[(index + 1) % points.length];
        if (current && next) {
            drawLine(image, current.x, current.y, next.x, next.y, color, lineWidth);
        }
    }
}

function drawRect(
    image: RgbImage,
    x: number,
    y: number,
    width: number,
    height: number,
    color: Color,
    lineWidth: number
): void {
    if (![x, y, width, height].every(Number.isFinite)) {
        return;
    }
    drawLine(image, x, y, x + width, y, color, lineWidth);
    drawLine(image, x + width, y, x + width, y + height, color, lineWidth);
    drawLine(image, x + width, y + height, x, y + height, color, lineWidth);
    drawLine(image, x, y + height, x, y, color, lineWidth);
}

function drawLine(
    image: RgbImage,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color: Color,
    lineWidth: number
): void {
    if (![x1, y1, x2, y2].every(Number.isFinite)) {
        return;
    }
    let x = Math.round(x1);
    let y = Math.round(y1);
    const endX = Math.round(x2);
    const endY = Math.round(y2);
    const dx = Math.abs(endX - x);
    const sx = x < endX ? 1 : -1;
    const dy = -Math.abs(endY - y);
    const sy = y < endY ? 1 : -1;
    let error = dx + dy;
    while (true) {
        drawPoint(image, x, y, color, lineWidth);
        if (x === endX && y === endY) {
            break;
        }
        const e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            y += sy;
        }
    }
}

function drawPoint(image: RgbImage, x: number, y: number, color: Color, size: number): void {
    const radius = Math.floor(size / 2);
    for (let yy = y - radius; yy <= y + radius; yy++) {
        for (let xx = x - radius; xx <= x + radius; xx++) {
            setPixel(image, xx, yy, color);
        }
    }
}

function drawFilledRect(
    image: RgbImage,
    x: number,
    y: number,
    width: number,
    height: number,
    color: Color
): void {
    const left = clamp(Math.round(x), 0, image.width);
    const top = clamp(Math.round(y), 0, image.height);
    const right = clamp(Math.round(x + width), 0, image.width);
    const bottom = clamp(Math.round(y + height), 0, image.height);
    for (let yy = top; yy < bottom; yy++) {
        for (let xx = left; xx < right; xx++) {
            setPixel(image, xx, yy, color);
        }
    }
}

function drawText(
    image: RgbImage,
    text: string,
    x: number,
    y: number,
    color: Color,
    scale: number
): void {
    let cursor = x;
    for (const raw of text) {
        const glyph = FONT[raw] ?? FONT[raw.toUpperCase()] ?? FONT[" "];
        if (!glyph) {
            cursor += 6 * scale;
            continue;
        }
        for (let row = 0; row < glyph.length; row++) {
            const line = glyph[row] ?? "";
            for (let col = 0; col < line.length; col++) {
                if (line[col] === "1") {
                    drawFilledRect(
                        image,
                        cursor + col * scale,
                        y + row * scale,
                        scale,
                        scale,
                        color
                    );
                }
            }
        }
        cursor += 6 * scale;
    }
}

function wrapText(text: string, maxLength: number): string[] {
    if (text.length <= maxLength) {
        return [text];
    }
    const lines: string[] = [];
    let current = "";
    for (const part of text.split(/\s+/)) {
        const next = current ? `${current} ${part}` : part;
        if (next.length > maxLength && current) {
            lines.push(current);
            current = part;
        } else {
            current = next;
        }
    }
    if (current) {
        lines.push(current);
    }
    return lines.length ? lines : [text.slice(0, maxLength)];
}

function parseTableRows(html: string): Array<Array<{ colspan: number }>> {
    return [...html.matchAll(/<tr[^>]*>(.*?)<\/tr>/gis)].map((rowMatch) => {
        const rowHtml = rowMatch[1] ?? "";
        const cells = [...rowHtml.matchAll(/<td([^>]*)>/gis)];
        return cells.length
            ? cells.map((cellMatch) => ({
                  colspan: parseColspan(cellMatch[1] ?? ""),
              }))
            : [{ colspan: 1 }];
    });
}

function parseColspan(attributes: string): number {
    const match = attributes.match(/colspan\s*=\s*["']?(\d+)/i);
    const value = match ? Number(match[1]) : 1;
    return Number.isFinite(value) && value > 0 ? value : 1;
}

function setPixel(image: RgbImage, x: number, y: number, color: Color): void {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return;
    }
    const index = (Math.round(y) * image.width + Math.round(x)) * 3;
    image.data[index] = color[0];
    image.data[index + 1] = color[1];
    image.data[index + 2] = color[2];
}

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}
