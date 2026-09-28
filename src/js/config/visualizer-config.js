/**
 * Excerpt from the application source (Main.js) - global scene configuration.
 * Author: Hadi Sarhangi Fard
 *
 * Only a subset of the options is shown here.
 */
export const VISUALIZER_CONFIG = {
  weightUrl: "./exports/mlp_weights.json",
  maxConnectionsPerNeuron: 5,
  inputSpacing: 0.85,
  inputNodeSize: 0.6,
  hiddenNodeRadius: 0.25,

  // Minimum |weight| a connection needs in order to be drawn.
  // Order: [input->H1, H1->H2, H2->H3, H3->output]
  connectionWeightThresholds: [0, 0, 0, 0],

  // Brightness of the connection lines in Silk Glow Mode, per layer group (0.0 - 1.0).
  silkBrightnessMultipliers: [0.3, 0.3, 0.3, 1],

  showFpsOverlay: true,
  silkMode: false,

  // Drawing brush of the 28x28 sketch pad
  brush: {
    drawRadius: 1.2,
    eraseRadius: 2.5,
    drawStrength: 0.95,
    eraseStrength: 0.95,
    softness: 0.3,
  },

  // Distance between consecutive layers along the X axis
  layerSpacing: {
    inputToHidden1: 20,
    hidden1ToHidden2: 20,
    hidden2ToHidden3: 20,
    hidden3ToOutput: 25,
  },

  // Each hidden layer is drawn as a cylinder
  hidden1Cylinder: { diameter: 36, length: 8, densityPower: 0.1 },  // 1024 neurons
  hidden2Cylinder: { diameter: 36, length: 8, densityPower: 0.3 },  // 784 neurons
  hidden3Cylinder: { diameter: 36, length: 8, densityPower: 0.15 }, // 484 neurons

  // The 10 output neurons sit on a circle
  outputLayer: { circleRadius: 10 },
};
