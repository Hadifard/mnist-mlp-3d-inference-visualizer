/**
 * Excerpt from the application source (Main.js) - the neuron analysis panel.
 * Author: Hadi Sarhangi Fard
 *
 * Two of the views of the panel:
 *
 *  - Important Pixels : numerical sensitivity of one neuron to each input pixel.
 *  - Learned Pattern  : average of the test images that excite the neuron the most.
 *
 * Both only need forward passes, so no gradients or extra libraries are required.
 */

// Sensitivity map (finite differences). A pixel is increased by a small epsilon
// and the change of the neuron's activation is measured.
// In the application this is used for neurons of the first hidden layer, whose
// inputs are the 784 pixels.
export function computeSaliencyMap(mlp, inputPixels, layerIndex, neuronIndex) {
  const activationOf = (pixels) => mlp.propagate(pixels).activations[layerIndex][neuronIndex];

  const saliency = new Float32Array(784);
  const epsilon = 0.01;
  const baseActivation = activationOf(inputPixels);

  for (let i = 0; i < 784; i++) {
    const modified = inputPixels.slice();
    modified[i] += epsilon;
    saliency[i] = Math.abs(activationOf(modified) - baseActivation) / epsilon;
  }
  return saliency;
}

// Average image of the K test samples with the highest activation for this neuron.
export function computeAverageTopPattern(mlp, samples, layerIndex, neuronIndex, topK = 20) {
  const scored = samples.map((sample) => ({
    pixels: sample.pixels,
    activation: mlp.propagate(sample.pixels).activations[layerIndex]?.[neuronIndex] || 0,
  }));
  scored.sort((a, b) => b.activation - a.activation);

  const top = scored.slice(0, topK);
  const average = new Float32Array(784);
  if (top.length === 0) return average;

  for (const item of top) {
    for (let i = 0; i < 784; i++) average[i] += item.pixels[i];
  }
  for (let i = 0; i < 784; i++) average[i] /= top.length;
  return average;
}
