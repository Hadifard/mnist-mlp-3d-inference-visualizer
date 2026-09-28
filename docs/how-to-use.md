# User guide

A practical walkthrough of the visualizer, from the first stroke to a full analysis of one neuron.
For a general introduction see the [main README](../README.md).

![Overview](images/app-overview-annotated.png)

## 1. Draw and predict

1. Move the mouse over the grid at the top right and hold the **left button** to draw.
2. Use the **right button** to erase. `RESET` clears the whole grid.
3. Read the prediction in the **Confidence** chart. It updates while you draw.

Good habits:

- Draw the digit large and roughly centred, like the MNIST examples.
- Use one continuous, reasonably thick stroke. Adjust **Brush thickness** and **Brush intensity** in the settings if the digit looks too faint.
- Try to draw a digit that is easy to confuse (4 and 9, 3 and 5, 7 and 1) and watch how the probabilities split.

## 2. Load a real sample

The buttons **0 to 9** next to the grid load a random image of that digit from the MNIST test set. Press the same button again for another example. Draw over it or erase parts to see how the prediction reacts to damage.

## 3. Explore the 3D scene

| Goal | How |
|------|-----|
| Rotate | Left click and drag, or the arrow keys |
| Move sideways | Right click and drag |
| Zoom | Mouse wheel |
| Look from a standard side | Camera widget at the bottom right |
| Zoom into a region | Zoom tool at the bottom left, then draw a rectangle |
| See everything again | Fit button under the zoom tool |
| More room for the scene | Sidebar toggle at the top of the scene |

What to look for:

- The **input wall** on the left shows the pixels of your drawing.
- **Bright neurons** in the hidden cylinders are the active ones. Because neurons are grouped by preferred digit, drawing a `3` lights up the same angular region in each cylinder.
- On the right, the label of the predicted digit glows green.

## 4. Tune the picture

Open the settings panel on the left.

| If you want | Change |
|-------------|--------|
| A calmer picture | Lower **Max connections per neuron** or raise the threshold sliders |
| To see more of the flow | Turn on **Silk Glow Mode** and raise the connection count |
| Individual connections | Turn Silk Glow off and use a small connection count |
| Brighter or dimmer connections in Silk mode | **Lines opacity** and the per-layer **Brightness** sliders |
| To start again | **Default** |

## 5. Inspect a neuron

Click any neuron.

![Inspector](images/inspector-layer1-neuron33.png)

**Left sidebar: numbers**

- `Bias`, the learned offset.
- `Pre-activation (z)`, the weighted sum plus bias.
- `Activation`, which is `ReLU(z)`.
- The list of strongest incoming connections with input, weight and product.

**Left panel: images**

- *Input activations*: what reaches the neuron.
- *Weights*: what the neuron is tuned to (blue is positive, red is negative).
- *Weighted contributions*: input times weight, whose sum gives `z`.

Ask yourself: where do the blue areas of the weight image coincide with bright input pixels? Those positions raise `z`. Where a red area meets a bright pixel, `z` is pushed down.

## 6. Analyse what a neuron detects

Use the panel on the right of the scene.

1. **Learned Pattern** shows the weights right away for first-layer neurons.
2. Press **Generate** to compute the average of the 20 test images that excite the neuron the most. The status line changes from `Direct Weights` to `Avg of Top 20`.
3. **Important Pixels** shows which pixels the neuron is most sensitive to.
4. **Activation Heatmap** overlays the contribution of each pixel of your own drawing.

Click a picture to enlarge it.

![Feature analysis](images/feature-analysis.png)

## 7. Travel through training

Use the **Training Progress** slider at the bottom.

- Start at the left end: the network has seen almost no data and its predictions are unreliable.
- Move to the right: predictions become confident, and the neuron pattern becomes more organised.
- The text under the slider shows how many images and batches were processed, the test accuracy and the loss of each snapshot.

## 8. Keyboard and mouse summary

| Action | Input |
|--------|-------|
| Draw / erase | Left / right button on the grid |
| Rotate / pan / zoom the scene | Left drag / right drag / wheel |
| Rotate with keys | Arrow keys |
| Close a dialog or cancel rectangle zoom | `Esc` |

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| The scene is slow | Turn off Silk Glow, reduce the number of connections, close other GPU-heavy tabs. |
| Nothing lights up | Draw something first, or press a digit button. |
| Prediction is wrong | Redraw larger and more centred. A fully connected network is sensitive to position. |
| Too many lines | Increase the threshold of layer 0 to 1, for example to 0.35. |
| Lost in the 3D view | Press the Fit button or choose the Front view. |
