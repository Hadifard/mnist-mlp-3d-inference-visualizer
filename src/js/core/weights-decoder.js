/**
 * Excerpt from the application source (Main.js) - weight decoding.
 * Author: Hadi Sarhangi Fard
 *
 * The trained weights are exported as Base64-encoded Float16 buffers to keep the
 * download small. The browser decodes them back to Float32 at start-up.
 */

export function decodeBase64ToUint8Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

// IEEE 754 half precision -> single precision
export function float16ToFloat32(value) {
  const sign = (value & 0x8000) >> 15;
  const exponent = (value & 0x7c00) >> 10;
  const fraction = value & 0x03ff;

  let result;
  if (exponent === 0) {
    // zero or subnormal
    result = fraction === 0 ? 0 : (fraction / 0x400) * Math.pow(2, -14);
  } else if (exponent === 0x1f) {
    result = fraction === 0 ? Number.POSITIVE_INFINITY : Number.NaN;
  } else {
    result = (1 + fraction / 0x400) * Math.pow(2, exponent - 15);
  }
  return sign === 1 ? -result : result;
}

export function decodeFloat16Base64(base64, expectedLength) {
  const bytes = decodeBase64ToUint8Array(base64);
  if (bytes.byteLength % 2 !== 0) {
    throw new Error("Float16 data has an invalid length.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = bytes.byteLength / 2;
  if (Number.isFinite(expectedLength) && expectedLength > 0 && length !== expectedLength) {
    throw new Error(`Expected ${expectedLength} Float16 values, but received ${length}.`);
  }
  const result = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    result[index] = float16ToFloat32(view.getUint16(index * 2, true));
  }
  return result;
}

// One flat buffer -> array of rows. Row i holds the incoming weights of neuron i.
export function decodeWeightMatrix(encoded, shape) {
  const rows = Math.max(0, Number(shape?.[0]) || 0);
  const cols = Math.max(0, Number(shape?.[1]) || 0);
  if (rows === 0 || cols === 0) return [];

  const flat = decodeFloat16Base64(encoded, rows * cols);
  const result = [];
  for (let row = 0; row < rows; row += 1) {
    result.push(flat.slice(row * cols, (row + 1) * cols));
  }
  return result;
}
