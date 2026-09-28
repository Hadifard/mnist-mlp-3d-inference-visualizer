/**
 * Excerpt from the application source (Main.js) - which digit does each neuron prefer?
 * Author: Hadi Sarhangi Fard
 *
 * A number of MNIST test images are pushed through the network. For every hidden
 * neuron the positive activation is summed per digit class.
 *   bestDigit   : the class with the largest sum
 *   selectivity : largest sum / total sum. About 0.1 means a general-purpose neuron,
 *                 1.0 means a neuron that only fires for one digit.
 */
export async function computeNeuronClusters(mlp, mnistLoader, sampleCount = 1000) {
  const layerCount = mlp.architecture.length;
  const profiles = {};
  const activationStats = {};

  for (let l = 1; l < layerCount - 1; l++) {
    const neuronCount = mlp.architecture[l];
    activationStats[l] = Array.from({ length: neuronCount }, () => new Float32Array(10));
    profiles[l] = new Array(neuronCount);
  }

  const totalSamples = Math.min(sampleCount, mnistLoader.totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    const sample = mnistLoader.getSampleByIndex(i);
    if (!sample) continue;

    const propagation = mlp.propagate(sample.pixels);
    for (let l = 1; l < layerCount - 1; l++) {
      const acts = propagation.preActivations[l - 1];
      for (let n = 0; n < acts.length; n++) {
        if (acts[n] > 0) activationStats[l][n][sample.digit] += acts[n];
      }
    }
  }

  for (let l = 1; l < layerCount - 1; l++) {
    for (let n = 0; n < mlp.architecture[l]; n++) {
      const stats = activationStats[l][n];
      let maxAct = -1;
      let bestDigit = 0;
      let sum = 0;

      for (let d = 0; d < 10; d++) {
        sum += stats[d];
        if (stats[d] > maxAct) {
          maxAct = stats[d];
          bestDigit = d;
        }
      }

      profiles[l][n] = {
        bestDigit: sum === 0 ? Math.floor(Math.random() * 10) : bestDigit,
        selectivity: sum > 0 ? maxAct / sum : 0,
      };
    }
  }
  return profiles;
}
