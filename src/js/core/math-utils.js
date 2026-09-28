/**
 * Excerpt from the application source (Main.js) - small numeric helpers.
 * Author: Hadi Sarhangi Fard
 */

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// Numerically stable softmax: subtracting the maximum avoids overflow in exp().
export function softmax(values) {
  if (!values.length) return [];
  const maxVal = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxVal));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => (sum === 0 ? 0 : value / sum));
}
