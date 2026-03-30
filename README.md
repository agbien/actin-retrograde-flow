# Actin Retrograde Flow Analyzer

Fully automated measurement of F-actin retrograde flow rates in neuronal growth cones from TIRF microscopy image stacks.

The tool automatically detects the growth cone, places measurement lines along the retrograde flow axis, generates kymographs, and uses Claude Sonnet (a multimodal vision model) to read the diagonal streak slopes and compute flow rates — no manual steps required.

## Background

In neuronal growth cones, actin filaments undergo a continuous rearward movement known as **retrograde flow**. This flow is driven by actin polymerization at the leading edge and myosin II motor activity pulling on the actin network. Retrograde flow opposes growth cone advancement, and its rate is modulated by adhesion to the substrate via the **molecular clutch** — when transmembrane adhesion molecules (synCAMs, integrins, cadherins) couple the actin cytoskeleton to the extracellular substrate, retrograde flow slows.

**TIRF microscopy** selectively illuminates fluorophores within ~100-200 nm of the coverslip, revealing the actin-rich lamellipodium with high contrast. When neurons express a fluorescent actin reporter (e.g., LifeAct-GFP), individual actin features appear as small speckles that move rearward over time. By acquiring images at regular intervals (typically every 2-3 seconds), this motion can be captured and quantified.

This tool uses **kymograph analysis** — the standard method in the field — combined with a vision model to fully automate the measurement pipeline.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/agbien/actin-retrograde-flow.git
cd actin-retrograde-flow
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and numerical computation |
| `matplotlib` | Figure generation and optional interactive GUI (requires TkAgg backend) |
| `scipy` | `scipy.ndimage.map_coordinates` for subpixel intensity sampling along lines |
| `tifffile` | Reading and writing TIFF stacks |
| `anthropic` | Anthropic API client for Claude Sonnet vision-based kymograph analysis |
| `nd2` | Reading Nikon NIS-Elements `.nd2` files |

#### Platform-specific notes

If matplotlib can't open a window (only needed for `--interactive` or `--manual-lines` mode):

```bash
# macOS
brew install python-tk
# Ubuntu/Debian
sudo apt-get install python3-tk
```

## Usage

```bash
python actin_retrograde_flow.py <image_stack> [options]
```

### Input formats

The tool accepts:
- **Nikon `.nd2` files** — loaded directly via the `nd2` package. Pixel size and frame interval are read from metadata automatically.
- **Multi-frame TIFF files** (`.tif`, `.tiff`) — standard output from Micro-Manager, NIS-Elements, MetaMorph, etc.
- **Directory of single-frame TIFFs** — sorted by filename and stacked.

Multi-channel and z-stack data are handled automatically (first channel, first z-slice).

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pixel-size` | `0.108` | Microns per pixel. Auto-detected from nd2 metadata if available. |
| `--frame-interval` | `2.0` | Seconds between frames. Auto-detected from nd2 metadata if available. |
| `--line-width` | `3` | Width (in pixels) of the sampling band around each line, for noise averaging. |
| `--n-lines` | `3` | Number of measurement lines to auto-place on the growth cone. |
| `--output-dir` | `<input>_results/` | Directory for output files. |
| `--model` | `claude-sonnet-4-6` | Vision model for kymograph analysis. |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key. |
| `--manual-lines` | off | Draw lines manually instead of auto-detecting the growth cone. |
| `--interactive` | off | Fully manual mode: draw lines by hand + click on kymograph streaks. No API needed. |

### Examples

```bash
# Fully automated — just point at an nd2 file
python actin_retrograde_flow.py "Tether 1000xGFP003.nd2"

# TIFF with custom calibration
python actin_retrograde_flow.py data/wt_laminin.tif --pixel-size 0.065 --frame-interval 3.0

# More measurement lines for better statistics
python actin_retrograde_flow.py data/stack.nd2 --n-lines 5

# Manual line drawing, automated measurement
python actin_retrograde_flow.py data/stack.nd2 --manual-lines

# Fully manual (no API key needed)
python actin_retrograde_flow.py data/stack.nd2 --interactive
```

### Batch processing

To process all nd2 files in a directory:

```bash
for f in data/*.nd2; do
    python actin_retrograde_flow.py "$f"
done
```

**Cost estimate:** Each file requires ~3 API calls (one per kymograph line). At Sonnet pricing (~$0.01/call), a full dataset of ~150 files costs approximately **$4 total**.

## How It Works

### Step 1: Load the image stack

The tool reads nd2 or TIFF data into a 3D array `(T, Y, X)`. For nd2 files, pixel size and frame interval are extracted from embedded metadata. Multi-channel data (e.g., LifeAct + synCAM flag) uses the first channel by default.

### Step 2: Detect the growth cone and place lines

The tool automatically identifies the growth cone:

1. **Threshold** the first frame to find bright regions.
2. **Find the largest connected component** (the neuron).
3. **Locate the tip** — the point on the neuron boundary furthest from the centroid. This is the leading edge of the growth cone.
4. **Compute the retrograde flow axis** — the direction from the tip inward toward the cell body.
5. **Place measurement lines** through the lamellipodium, centered between the tip and the growth cone body, at 0 degrees and +-30 degrees from the flow axis.

If auto-detection fails (e.g., multiple neurons in the field), use `--manual-lines` to draw lines by hand.

### Step 3: Generate kymographs

For each line, pixel intensities are sampled at every point along the line for every frame using bilinear interpolation. A band of parallel lines (controlled by `--line-width`) is averaged to reduce noise. The result is a 2D kymograph where:

- **X-axis** = distance along the line (in um)
- **Y-axis** = time (in seconds, increasing downward)
- **Vertical streaks** = stationary features
- **Diagonal streaks** = moving actin features (the slope is the velocity)

### Step 4: Measure flow rate with Claude Sonnet

Each kymograph is rendered as a PNG with calibrated axes and sent to Claude Sonnet, which:

1. **Identifies diagonal streaks** in the kymograph.
2. **Reads two coordinate points** on each streak directly from the axis labels.
3. **Computes the flow rate** via point-slope form:

```
rate (um/min) = |x2 - x1| / |t2 - t1| * 60
```

The detected endpoints are overlaid on the kymograph in the output figure for visual verification.

### Why a vision model?

We benchmarked 10 models on kymographs from real 16-bit nd2 data (Tether synCAM, expected ~12.5 um/min):

| Model | Mean rate | MAE | Local? |
|-------|----------|-----|--------|
| **Claude Sonnet 4.6** | **12.7** | **0.2** | No |
| Claude Opus 4.6 | 13.0 | 0.8 | No |
| Claude Haiku 4.5 | — | 1.9 | No |
| gemma3:12b | 16.5 | 4.0 | Yes |
| gemma3:4b | 20.0 | 7.5 | Yes |
| minicpm-v | 22.4 | 21.8 | Yes |
| llava-llama3 | 40.0 | 35.8 | Yes |
| llama3.2-vision | 60.0 | 47.5 | Yes |
| llava:7b | 0.0 | 12.5 | Yes |
| moondream | 0.0 | 12.5 | Yes |

Traditional CV methods (cross-correlation, Hough transform, structure tensor) were also tested and failed on both compressed and raw data. The vision model approach matches human expert readings within 0.2 um/min.

### Step 5: Output

| File | Description |
|------|-------------|
| `actin_flow_results.png` | Composite figure: annotated growth cone with auto-detected lines, kymographs with detected streak lines overlaid in red, and bar chart of flow rates. |
| `L{n}_kymograph.png` | Individual kymograph images with calibrated axes. |
| `L{n}_kymograph.tif` | Raw kymograph data as 32-bit TIFF (for ImageJ/FIJI). |
| `flow_rate_summary.csv` | One row per line: coordinates, length, flow rate, number of streaks detected. |
| `L{n}_streaks.csv` | Per-streak detail: endpoint coordinates and rate for each detected streak. |

## Interpreting Results

### Reading a kymograph

- **Diagonal streaks** (upper-right to lower-left, or vice versa) = moving actin features. Slope = velocity.
- **Steeper streaks** (closer to vertical) = slower flow.
- **Shallower streaks** (closer to horizontal) = faster flow.
- **Vertical streaks** = stationary features (e.g., adhesion sites).
- **Featureless kymograph** = line missed the growth cone or is perpendicular to the flow axis.

### Typical flow rates

| Condition | Expected rate | Mechanism |
|-----------|--------------|-----------|
| Tether synCAM / WT on GFP | 10-15 um/min | No clutch engagement, free retrograde flow |
| ITGB1 / ICAM1 synCAM | 3-7 um/min | Molecular clutch engaged, flow slowed by substrate coupling |
| WT on laminin | 5-10 um/min | Endogenous integrin-mediated adhesion |
| + ROCK inhibitor | Reduced | Myosin II inhibition reduces flow |
| + Blebbistatin | Reduced | Myosin II inhibition reduces flow |

### Sanity checks

- Flow rate of 0 = line perpendicular to flow direction, or featureless kymograph.
- Rate > 20 um/min = likely artifact or misidentified feature.
- Open `L{n}_kymograph.tif` in ImageJ to visually verify the streaks match the reported rate.
- Compare across the 3 auto-placed lines — consistent rates indicate reliable measurement.

## License

MIT
