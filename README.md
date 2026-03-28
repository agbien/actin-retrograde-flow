# Actin Retrograde Flow Analyzer

Measure the retrograde flow rate of F-actin in neuronal growth cones from TIRF microscopy image stacks.

The tool lets you interactively draw lines along actin flow axes on a growth cone image, generates kymographs (space-time plots) from those lines, and computes flow rates by cross-correlating consecutive kymograph rows.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/abien/actin-retrograde-flow.git
cd actin-retrograde-flow
pip install -r requirements.txt
```

### Dependencies

- `numpy` - array operations
- `matplotlib` - visualization and interactive line drawing (requires a GUI backend like TkAgg)
- `scipy` - cross-correlation for flow measurement
- `scikit-image` - image processing utilities
- `tifffile` - TIFF stack I/O

On macOS you may need to install `python-tk` if matplotlib can't open a window:

```bash
brew install python-tk
```

On Ubuntu/Debian:

```bash
sudo apt-get install python3-tk
```

## Usage

```bash
python actin_retrograde_flow.py <image_stack> [options]
```

`<image_stack>` can be:
- A multi-frame TIFF file (e.g. `growth_cone.tif`)
- A directory of sequentially numbered single-frame TIFFs

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pixel-size` | `0.108` | Microns per pixel |
| `--frame-interval` | `2.0` | Seconds between frames |
| `--line-width` | `3` | Width (in pixels) of the sampling band around each drawn line, for noise averaging |
| `--output-dir` | `<input>_results/` | Directory for output files |

### Example

```bash
python actin_retrograde_flow.py data/wt_laminin.tif --pixel-size 0.108 --frame-interval 2.0
```

## Workflow

1. **Load** - The tool reads your TIFF stack and displays the first frame.
2. **Draw lines** - Click two points to define each line along an actin flow axis. Draw as many lines as needed. Press Enter or click "Done" when finished. Right-click or press `u` to undo.
3. **Kymograph generation** - Pixel intensities are sampled along each line across all time frames, producing a kymograph where the x-axis is distance along the line and the y-axis is time. Diagonal streaks in the kymograph correspond to moving actin features.
4. **Flow rate measurement** - Cross-correlation between consecutive kymograph rows determines the spatial shift per time step. The median shift is converted to microns/min.
5. **Output** - Figures and data files are saved to the output directory.

## Outputs

| File | Description |
|------|-------------|
| `actin_flow_results.png` | Composite figure: annotated growth cone, kymographs with flow rates, bar chart |
| `actin_flow_shifts.png` | Per-frame shift diagnostic plots for each line |
| `L{n}_kymograph.tif` | Raw kymograph data as 32-bit TIFF |
| `flow_rate_summary.csv` | Line coordinates, lengths, and flow rates |
| `L{n}_shifts.csv` | Frame-by-frame shift data for each line |

## How It Works

A **kymograph** is constructed by sampling pixel intensities along the drawn line for every frame in the stack. In the resulting 2D image, one axis is position along the line and the other is time. Actin features moving along the line appear as diagonal streaks. The slope of these streaks equals the flow velocity:

```
velocity = spatial_shift / time_interval
```

The tool estimates the shift between consecutive frames using cross-correlation, then takes the median across all frame pairs for a robust measurement.

## License

MIT
