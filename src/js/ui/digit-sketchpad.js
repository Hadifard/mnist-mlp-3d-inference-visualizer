/**
 * Excerpt from the application source (Main.js) - the 28x28 drawing pad.
 * Author: Hadi Sarhangi Fard
 *
 * The pad stores one float per pixel (0..1). A soft round brush adds intensity
 * with a falloff from the centre, which gives strokes that look much closer to
 * real MNIST digits than hard on/off pixels.
 */
import { clamp } from "../core/math-utils.js";

export class DigitSketchPad {
  constructor(rows = 28, cols = 28, brush = {}) {
    this.rows = rows;
    this.cols = cols;
    this.values = new Float32Array(rows * cols);
    this.cells = []; // DOM cells, created by buildGrid() in the full source
    this.brush = Object.assign(
      { drawRadius: 1.2, eraseRadius: 1.2, drawStrength: 0.85, eraseStrength: 0.8, softness: 0.5 },
      brush
    );
  }

  // Paints (or erases) around one cell. Returns true if any pixel changed.
  applyBrush(centerRow, centerCol, erase = false) {
    const radius = erase ? this.brush.eraseRadius : this.brush.drawRadius;
    const strength = erase ? -this.brush.eraseStrength : this.brush.drawStrength;
    const softness = clamp(this.brush.softness ?? 0.5, 0, 0.95);
    const span = Math.ceil(radius);
    let modified = false;

    for (let row = centerRow - span; row <= centerRow + span; row += 1) {
      if (row < 0 || row >= this.rows) continue;
      for (let col = centerCol - span; col <= centerCol + span; col += 1) {
        if (col < 0 || col >= this.cols) continue;

        const distance = Math.hypot(row - centerRow, col - centerCol);
        if (distance > radius) continue;

        const falloff = 1 - distance / radius;
        if (falloff <= 0) continue;

        const influence = Math.pow(falloff, 1 + softness * 2);
        const delta = strength * influence;
        if (Math.abs(delta) < 1e-3) continue;

        const cellIndex = row * this.cols + col;
        const current = this.values[cellIndex];
        const nextValue = clamp(current + delta, 0, 1);
        if (nextValue === current) continue;

        this.values[cellIndex] = nextValue;
        this.updateCellVisual(cellIndex);
        modified = true;
      }
    }
    return modified;
  }

  // Cyan for weak strokes, moving towards green as the pixel gets brighter.
  updateCellVisual(index) {
    const cell = this.cells[index];
    if (!cell) return;
    const value = this.values[index];
    if (value <= 0) {
      cell.style.background = "rgba(255, 255, 255, 0.05)";
      cell.classList.remove("active");
      return;
    }
    const hue = 180 - value * 70;
    const saturation = 70 + value * 25;
    const lightness = 25 + value * 40;
    cell.style.background = `hsl(${hue.toFixed(0)}, ${saturation.toFixed(0)}%, ${lightness.toFixed(0)}%)`;
    cell.classList.add("active");
  }

  // Flat Float32Array(784) handed to the model
  getPixels() {
    return Float32Array.from(this.values);
  }
}
