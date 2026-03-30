# Actin Retrograde Flow Analyzer

Measure the retrograde flow rate of F-actin in neuronal growth cones from TIRF microscopy image stacks.

## Background

In neuronal growth cones, actin filaments undergo a continuous rearward movement known as **retrograde flow**. This flow is driven by actin polymerization at the leading edge (pushing filaments backward) and myosin II motor activity pulling on the actin network deeper in the growth cone. Retrograde flow is a fundamental process in growth cone motility: it opposes growth cone advancement, and its rate is modulated by adhesion to the substrate. When transmembrane adhesion molecules (such as synCAMs, integrins, or cadherins) mechanically couple the actin cytoskeleton to the extracellular substrate, retrograde flow slows — a phenomenon called the "molecular clutch." Measuring the retrograde flow rate under different adhesion conditions is therefore critical for understanding how growth cones extend or collapse.

**Total Internal Reflection Fluorescence (TIRF) microscopy** is the standard imaging modality for visualizing actin retrograde flow. TIRF selectively illuminates fluorophores within ~100-200 nm of the coverslip, eliminating out-of-focus background and revealing the thin actin-rich lamellipodium of the growth cone with high contrast. When neurons express a fluorescent actin reporter (e.g., LifeAct-GFP), individual actin features appear as small speckles or streaks that move rearward over time. By acquiring TIRF images at regular intervals (typically every 2-3 seconds), the movement of these actin features can be tracked.

This tool provides a complete pipeline for going from a raw TIRF image stack to quantified actin retrograde flow rates, using **kymograph analysis** — the standard method in the field.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/abien/actin-retrograde-flow.git
cd actin-retrograde-flow
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and numerical computation |
| `matplotlib` | Visualization, interactive line drawing GUI, and figure generation. Requires a GUI backend (TkAgg is used by default) |
| `scipy` | `scipy.ndimage.map_coordinates` for subpixel intensity sampling along lines |
| `tifffile` | Reading and writing TIFF stacks, the standard format for microscopy image series |
| `anthropic` | Anthropic API client for Claude Sonnet vision-based kymograph analysis |

You will also need an Anthropic API key. Set it as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Platform-specific notes

matplotlib requires a GUI backend to display the interactive line-drawing window. If you get an error about no display or no backend, install Tk:

**macOS:**
```bash
brew install python-tk
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**Windows:** Tk is included with the standard Python installer. No extra steps needed.

## Usage

```bash
python actin_retrograde_flow.py <image_stack> [options]
```

### Input formats

`<image_stack>` accepts two formats:

1. **Multi-frame TIFF file** — A single `.tif` or `.tiff` file containing all time points as separate frames. This is the standard output format from most microscopy acquisition software (e.g., Micro-Manager, NIS-Elements, MetaMorph).

2. **Directory of single-frame TIFFs** — A folder containing one TIFF per time point, named so that alphabetical sorting gives the correct temporal order (e.g., `frame_001.tif`, `frame_002.tif`, ...). The tool sorts by filename and stacks them.

If the input contains multiple channels (4D array), the first channel is used.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pixel-size` | `0.108` | Spatial calibration in microns per pixel. This value depends on your objective and camera. For a typical 60x TIRF objective with a standard sCMOS camera, 0.108 um/px is common. Check your microscope's calibration. |
| `--frame-interval` | `2.0` | Time between consecutive frames in seconds. Must match your acquisition settings. |
| `--line-width` | `3` | Number of parallel lines to average when sampling the kymograph (see Kymograph Generation below). Higher values reduce noise but blur spatial features. Must be an odd integer. |
| `--output-dir` | `<input>_results/` | Directory where all output files are written. Created automatically if it doesn't exist. |
| `--model` | `claude-sonnet-4-6` | Vision model to use for kymograph analysis. |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key. Defaults to the `ANTHROPIC_API_KEY` environment variable. |
| `--interactive` | off | Use manual interactive mode instead of the vision model. In this mode, you click points along streaks and the tool fits lines with `numpy.polyfit`. |

### Example

```bash
# Basic usage with default calibration (60x TIRF, 2 sec intervals)
python actin_retrograde_flow.py data/wt_laminin.tif

# Custom calibration for a 100x objective with 3 sec intervals
python actin_retrograde_flow.py data/wt_laminin.tif --pixel-size 0.065 --frame-interval 3.0

# Process a directory of individual frames with wider averaging
python actin_retrograde_flow.py data/growth_cone_frames/ --line-width 5

# Specify output location
python actin_retrograde_flow.py data/wt_laminin.tif --output-dir results/experiment1/

# Use manual interactive mode (no API key needed)
python actin_retrograde_flow.py data/wt_laminin.tif --interactive
```

## Step-by-Step Workflow

### Step 1: Loading the image stack

The tool reads your TIFF data into a 3D NumPy array of shape `(T, Y, X)` where `T` is the number of time frames and `Y, X` are the spatial dimensions. All pixel values are converted to 64-bit floating point for subsequent computation. The tool prints the stack dimensions so you can verify the correct number of frames were loaded.

### Step 2: Interactive line drawing

A matplotlib window opens showing the **first frame** of your image stack. This frame is used as a reference for placing measurement lines.

**How to draw lines:**
- **Left-click** once to place the first endpoint of a line, then **left-click** again to place the second endpoint. The line appears in red and is labeled (L1, L2, etc.).
- You can draw **multiple lines** on the same image — each one will produce its own independent kymograph and flow rate measurement. This is useful for measuring flow along different axes of the growth cone (e.g., along the central domain vs. along a filopodium), or for getting replicate measurements.
- **Right-click** or press **`u`** to undo the last line (or cancel a partially-placed line).
- Press **Enter** or click the **"Done"** button when you have placed all the lines you want.

**Line placement guidelines:**
- Draw lines **parallel to the expected direction of actin retrograde flow**, which is typically from the leading edge of the growth cone inward toward the central domain and axon shaft.
- Lines should be long enough to capture several actin features moving along the flow axis, but short enough to stay within the growth cone boundary. Typical line lengths are 3-10 um (30-100 pixels at 0.108 um/px).
- Avoid drawing lines that cross the boundary of the growth cone into background, as this will add noise to the kymograph.
- The line direction matters: the kymograph's spatial axis runs from the first click to the second click. For consistent results, click first at the leading edge and second toward the cell body, so that retrograde flow appears as features moving from left to right (or top to bottom) in the kymograph.

### Step 3: Kymograph generation

For each drawn line, the tool constructs a **kymograph** — a 2D image where one axis represents position along the line (in microns) and the other represents time (in seconds).

**How the kymograph is built:**

For every frame in the stack, pixel intensities are sampled at evenly-spaced points along the drawn line using **bilinear interpolation** (`scipy.ndimage.map_coordinates` with `order=1`). The number of sample points equals the line length in pixels (rounded up), so sampling is at the native resolution.

To reduce noise, the tool doesn't sample just the single-pixel-wide line itself — it samples a **band** of parallel lines centered on the drawn line and averages them. The width of this band is controlled by `--line-width` (default 3, meaning the drawn line plus one parallel line on each side, spaced 1 pixel apart perpendicular to the line direction). This is equivalent to averaging over a rectangular region that slides along the line, which suppresses shot noise without blurring the features moving along the flow axis.

The result is a 2D array of shape `(T, N)` where `T` is the number of time frames and `N` is the number of sample points along the line. When displayed as an image:
- The **horizontal axis** (columns) represents distance along the line
- The **vertical axis** (rows) represents time, increasing downward
- **Stationary features** appear as vertical streaks
- **Moving features** appear as diagonal streaks — the slope of these streaks is the velocity

Each kymograph is saved as a 32-bit TIFF file so it can be opened in ImageJ/FIJI for further manual analysis if desired.

### Step 4: Flow rate measurement via vision model

Each kymograph is rendered as a PNG image with calibrated axes (distance in μm on x-axis, time in seconds on y-axis) and sent to **Claude Sonnet**, a multimodal vision model that reads the image the same way a human researcher would:

1. The model identifies **diagonal streaks** in the kymograph — these are the moving actin features whose slope encodes the flow velocity.

2. For each streak, the model reads **two coordinate points** directly from the calibrated axis labels: (x1 μm, t1 s) and (x2 μm, t2 s).

3. The flow rate is computed from the slope using point-slope form:

```
rate (μm/min) = |x2 - x1| / |t2 - t1| × 60
```

4. If multiple streaks are detected, the tool averages their rates.

5. The detected streak endpoints are overlaid on the kymograph in the output figure so you can visually verify the model's readings.

**Why a vision model?**

We benchmarked 7 models (Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5, llava-llama3, minicpm-v, llava:7b, moondream) on kymographs with known flow rates. Claude Sonnet achieved the lowest mean absolute error (0.7 μm/min on the presentation kymographs, 1.4 μm/min across 5 test videos). Traditional computer vision methods (Hough transform, structure tensor, cross-correlation) all failed on compressed 8-bit video data, while the vision model correctly reads the slopes even in noisy kymographs — just like a human researcher does.

**Fallback: interactive mode**

If you don't have an API key, use `--interactive` to manually click points along streaks. The tool fits a line through your points with `numpy.polyfit` and computes the rate from the slope.

### Step 5: Output generation

The tool produces several output files, described below.

## Output Files

| File | Description |
|------|-------------|
| `actin_flow_results.png` | **Main results figure.** A multi-panel composite containing: (1) the first frame of the image stack with all drawn lines overlaid in red and labeled, (2) one kymograph panel per line with calibrated axes (distance in μm, time in seconds), fitted slope lines overlaid in red, and the measured flow rate in the title, and (3) a bar chart summarizing all flow rates. Saved at 200 DPI. |
| `L{n}_kymograph.tif` | **Raw kymograph data** for line `n`, saved as a 32-bit floating point TIFF. Can be opened in ImageJ/FIJI for manual inspection or further processing. |
| `flow_rate_summary.csv` | **Summary table** with one row per line. Columns: line label, start/end coordinates (in pixels), line length (in μm), flow rate (in μm/min), number of streaks measured, fitted slope (μm/s), and R² value. Suitable for import into Excel, R, or Python for downstream statistical analysis. |

## Interpreting Results

### Reading a kymograph

In the kymograph image:
- **Bright diagonal streaks** running from upper-left to lower-right (or upper-right to lower-left, depending on line direction) represent actin features moving along the flow axis over time.
- **Steeper streaks** (closer to vertical) indicate slower flow — the feature doesn't move far between frames.
- **Shallower streaks** (closer to horizontal) indicate faster flow — the feature covers more distance per time step.
- **Vertical streaks** indicate stationary features (e.g., adhesion sites that are not moving with the flow).
- A **noisy or featureless** kymograph suggests the line was poorly placed (e.g., in background, or perpendicular to the flow axis).

### Typical flow rates

Published values for actin retrograde flow in neuronal growth cones typically range from **1-15 um/min**, depending on the substrate, adhesion molecules present, and neuron type:
- **High flow (~10-15 um/min):** Growth cones on non-adhesive substrates (e.g., GFP-coated glass) or with weak adhesion. The actin network is not coupled to the substrate and flows freely rearward.
- **Low flow (~3-7 um/min):** Growth cones on adhesive substrates (e.g., laminin) or expressing adhesion molecules (e.g., integrins, ICAM1 synCAMs) that engage the molecular clutch. Actin-substrate coupling transmits traction force and slows retrograde flow.

### Sanity checks

- If the measured flow rate is 0 or near 0, the line may be perpendicular to the flow direction, or the image stack may lack sufficient temporal resolution.
- If the shift plot shows wildly varying values with no consistent trend, the kymograph may be too noisy. Try increasing `--line-width` or repositioning the line.
- If the kymograph looks good but the automated rate seems wrong, open the `L{n}_kymograph.tif` in ImageJ and manually measure the slope of a few streaks to cross-check.

## License

MIT
