/**
 * Excerpt from the application source (Main.js) - choosing which connections to draw.
 * Author: Hadi Sarhangi Fard
 *
 * The network has roughly two million weights, far too many to render. For every
 * target neuron only the strongest incoming connections are kept:
 *   1. drop weights whose magnitude is below the threshold of that layer,
 *   2. sort the rest by magnitude,
 *   3. keep the first `maxConnectionsPerNeuron`.
 */
export function findImportantConnections(layer, layerIndex, options) {
  const limit = options.maxConnectionsPerNeuron;
  const layerThreshold = options.connectionWeightThresholds[layerIndex] ?? 0;

  const selected = [];
  let maxAbsWeight = 0;

  for (let target = 0; target < layer.weights.length; target += 1) {
    const row = layer.weights[target];
    const candidates = [];

    for (let source = 0; source < row.length; source += 1) {
      const weight = row[source];
      if (!Number.isFinite(weight)) continue;

      const magnitude = Math.abs(weight);
      if (magnitude < layerThreshold) continue;

      candidates.push({ sourceIndex: source, targetIndex: target, weight, magnitude });
      if (magnitude > maxAbsWeight) maxAbsWeight = magnitude;
    }

    candidates.sort((a, b) => b.magnitude - a.magnitude);

    const take = Math.min(limit, candidates.length);
    for (let i = 0; i < take; i += 1) {
      const { sourceIndex, targetIndex, weight } = candidates[i];
      selected.push({ sourceIndex, targetIndex, weight });
    }
  }

  // maxAbsWeight is used later to normalise line colour and opacity
  return { selected, maxAbsWeight };
}
