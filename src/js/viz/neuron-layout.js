/**
 * Excerpt from the application source (Main.js) - placing hidden neurons in 3D.
 * Author: Hadi Sarhangi Fard
 *
 * Every hidden layer is a cylinder whose axis is the X axis. The cylinder is cut
 * into 10 angular sectors of 36 degrees, one per digit. A neuron is placed inside the
 * sector of the digit it responds to most (see neuron-profiles.js), so groups of
 * neurons that specialise in the same digit sit next to each other.
 * Even-numbered layers are rotated by 180 degrees to separate them visually.
 *
 * In the full source this logic lives inside a method of the scene class and creates
 * THREE.Vector3 objects; here it returns plain {x, y, z} objects.
 */
export function computeCylinderPositions({ layerIndex, neuronCount, layerX, config, profiles }) {
  const positions = new Array(neuronCount);

  const cylinderLength = config.length;
  const outerRadius = config.diameter / 2;
  const innerRadius = outerRadius * 0.15;

  const layerPhaseShift = layerIndex % 2 === 0 ? Math.PI : 0;
  const sectorAngle = (Math.PI * 2) / 10;

  for (let i = 0; i < neuronCount; i++) {
    let xPos, theta, r;

    if (profiles && profiles[i]) {
      // 1. base angle of the neuron's preferred digit
      const baseTheta = profiles[i].bestDigit * sectorAngle + layerPhaseShift;

      // 2. spread neurons evenly inside the sector (fills 90 percent of its width)
      const thetaSpread = ((i * 137.5) % 100) / 100 - 0.5; // -0.5 .. 0.5
      theta = baseTheta + thetaSpread * sectorAngle * 0.9;

      // 3. position along the cylinder axis
      xPos = (((i * 73) % 1000) / 1000 - 0.5) * cylinderLength;

      // 4. radius: more neurons near the wall, fewer near the axis
      const randRadius = ((i * 19) % 1000) / 1000;
      const densityBias = 1 - Math.pow(randRadius, 3);
      r = innerRadius + (outerRadius - innerRadius) * densityBias;
    } else {
      // Fallback when no profiles exist: golden-angle spiral
      const goldenAngle = Math.PI * (3 - Math.sqrt(5));
      xPos = (i / neuronCount - 0.5) * cylinderLength;
      theta = i * goldenAngle;
      const densityBias = 1 - Math.pow(Math.random(), 3);
      r = innerRadius + (outerRadius - innerRadius) * densityBias;
    }

    positions[i] = { x: layerX + xPos, y: r * Math.cos(theta), z: r * Math.sin(theta) };
  }
  return positions;
}
