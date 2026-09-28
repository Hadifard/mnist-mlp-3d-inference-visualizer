/**
 * Excerpt from the application source (Main.js) - forward propagation.
 * Author: Hadi Sarhangi Fard
 *
 * The whole network runs in the browser. For every change of the drawing the
 * model returns the pre-activation (z) and the activation of EVERY neuron, so the
 * 3D scene and the inspection panels can show real numbers.
 *
 * The constructor is shortened for this excerpt; propagate() is shown as used.
 */
export class FeedForwardModel {
  constructor(definition) {
    this.normalization = definition.normalization ?? { mean: 0, std: 1 };
    this.layers = definition.layers.map((layer, index) => ({
      name: layer.name ?? `dense_${index}`,
      activation: layer.activation ?? "relu",
      weights: layer.weights.map((row) => Float32Array.from(row)), // weights[neuron][source]
      biases: Float32Array.from(layer.biases),
    }));
  }

  propagate(pixels) {
    const { mean, std } = this.normalization;
    const input = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i += 1) {
      input[i] = (pixels[i] - mean) / std;
    }

    const activations = [input];
    const preActivations = [];
    let current = input;

    for (const layer of this.layers) {
      const outSize = layer.biases.length;
      const linear = new Float32Array(outSize);

      // z = sum(weight * input) + bias
      for (let neuron = 0; neuron < outSize; neuron += 1) {
        let sum = layer.biases[neuron];
        const weights = layer.weights[neuron];
        for (let source = 0; source < weights.length; source += 1) {
          sum += weights[source] * current[source];
        }
        linear[neuron] = sum;
      }
      preActivations.push(linear);

      // ReLU for hidden layers. The output layer stays linear (logits);
      // softmax is applied afterwards for the confidence bars.
      let activated;
      if (layer.activation === "relu") {
        activated = new Float32Array(outSize);
        for (let i = 0; i < outSize; i += 1) {
          activated[i] = linear[i] > 0 ? linear[i] : 0;
        }
      } else {
        activated = linear.slice();
      }
      activations.push(activated);
      current = activated;
    }

    return { normalizedInput: activations[0], activations, preActivations };
  }
}
