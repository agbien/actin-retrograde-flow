"""
Actin Retrograde Flow Measurement Tool

Measures retrograde flow rate of actin in neuronal growth cones from
TIRF microscopy image stacks. The user draws lines along flow axes on
the growth cone image, and the tool generates kymographs and computes
flow rates from the slope of features in those kymographs.

Usage:
    python actin_retrograde_flow.py <image_stack> [--pixel-size 0.108] [--frame-interval 2.0]

    image_stack:      Path to a TIFF stack (multi-frame) or a directory of
                      sequentially numbered single-frame TIFFs.
    --pixel-size:     μm per pixel (default: 0.108 μm/px, typical for 60x TIRF)
    --frame-interval: seconds between frames (default: 2.0 s)
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import map_coordinates
from scipy.signal import correlate
import tifffile


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_stack(path: str) -> np.ndarray:
    """Load a TIRF image stack. Returns array of shape (T, Y, X)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
        if not files:
            sys.exit(f"No TIFF files found in directory: {path}")
        frames = [tifffile.imread(str(f)) for f in files]
        stack = np.stack(frames, axis=0)
    else:
        stack = tifffile.imread(str(p))

    if stack.ndim == 2:
        sys.exit("Only a single frame found — need a time series (multiple frames).")
    # Handle multi-channel: take first channel or average
    if stack.ndim == 4:
        stack = stack[:, 0]  # first channel
    print(f"Loaded stack: {stack.shape[0]} frames, {stack.shape[1]}×{stack.shape[2]} px")
    return stack.astype(np.float64)


# ---------------------------------------------------------------------------
# Interactive line drawing
# ---------------------------------------------------------------------------

class LineDrawer:
    """Let the user draw multiple lines on an image. Each line is defined by
    clicking two endpoints. Right-click or press 'u' to undo the last line.
    Press Enter or close the window when done."""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.lines = []          # list of ((x0,y0), (x1,y1))
        self._current_pt = None
        self._line_artists = []
        self._dot_artists = []
        self.done = False

    def run(self) -> list:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 9))
        self.ax.imshow(self.image, cmap="gray", origin="upper")
        self.ax.set_title(
            "Click two points to draw a line along the flow axis.\n"
            "Draw multiple lines, then press Enter or close window when done.\n"
            "Right-click or 'u' to undo last line."
        )
        self.ax.set_xlabel("x (px)")
        self.ax.set_ylabel("y (px)")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # "Done" button
        ax_btn = self.fig.add_axes([0.8, 0.01, 0.15, 0.04])
        self._btn = Button(ax_btn, "Done")
        self._btn.on_clicked(lambda _: self._finish())

        plt.show()
        return self.lines

    def _on_click(self, event):
        if event.inaxes != self.ax or self.done:
            return
        if event.button == 3:  # right click = undo
            self._undo()
            return
        x, y = event.xdata, event.ydata
        if self._current_pt is None:
            self._current_pt = (x, y)
            dot, = self.ax.plot(x, y, "r+", markersize=10, markeredgewidth=2)
            self._dot_artists.append(dot)
            self.fig.canvas.draw_idle()
        else:
            x0, y0 = self._current_pt
            self.lines.append(((x0, y0), (x, y)))
            ln, = self.ax.plot([x0, x], [y0, y], "r-", linewidth=1.5)
            self._line_artists.append(ln)
            # Label
            mx, my = (x0 + x) / 2, (y0 + y) / 2
            self.ax.text(mx, my, f"L{len(self.lines)}", color="yellow",
                         fontsize=9, fontweight="bold",
                         ha="center", va="bottom")
            self._current_pt = None
            # Remove the dot for first click
            if self._dot_artists:
                self._dot_artists[-1].remove()
                self._dot_artists.pop()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "enter":
            self._finish()
        elif event.key == "u":
            self._undo()

    def _undo(self):
        if self._current_pt is not None:
            self._current_pt = None
            if self._dot_artists:
                self._dot_artists[-1].remove()
                self._dot_artists.pop()
            self.fig.canvas.draw_idle()
        elif self.lines:
            self.lines.pop()
            if self._line_artists:
                self._line_artists[-1].remove()
                self._line_artists.pop()
            self.fig.canvas.draw_idle()

    def _finish(self):
        self.done = True
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Kymograph generation
# ---------------------------------------------------------------------------

def make_kymograph(stack: np.ndarray, line, width: int = 3) -> np.ndarray:
    """Sample pixel intensities along `line` for every frame in `stack`.

    Args:
        stack:  (T, Y, X) array
        line:   ((x0, y0), (x1, y1))
        width:  number of parallel lines to average (odd, for noise reduction)

    Returns:
        kymograph: (T, N) array where N = number of sample points along line
    """
    (x0, y0), (x1, y1) = line
    length = np.hypot(x1 - x0, y1 - y0)
    n_pts = int(np.ceil(length))

    # Unit vectors along and perpendicular to the line
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -dy, dx  # normal

    offsets = np.arange(-(width // 2), width // 2 + 1)  # e.g. [-1, 0, 1]
    t_vals = np.linspace(0, 1, n_pts)

    kymo = np.zeros((stack.shape[0], n_pts))

    for frame_idx in range(stack.shape[0]):
        frame = stack[frame_idx]
        accum = np.zeros(n_pts)
        for off in offsets:
            xs = x0 + t_vals * (x1 - x0) + off * nx
            ys = y0 + t_vals * (y1 - y0) + off * ny
            coords = np.vstack([ys, xs])  # map_coordinates expects (row, col)
            accum += map_coordinates(frame, coords, order=1, mode="nearest")
        kymo[frame_idx] = accum / len(offsets)

    return kymo


# ---------------------------------------------------------------------------
# Flow rate measurement
# ---------------------------------------------------------------------------

def measure_flow_rate(kymo: np.ndarray, pixel_size: float,
                      frame_interval: float) -> dict:
    """Estimate flow rate from a kymograph using cross-correlation between
    consecutive frames to find the average spatial shift per time step.

    Returns dict with rate in μm/min and px/frame, plus per-frame shifts.
    """
    shifts = []
    for i in range(kymo.shape[0] - 1):
        row0 = kymo[i] - kymo[i].mean()
        row1 = kymo[i + 1] - kymo[i + 1].mean()
        cc = correlate(row0, row1, mode="full")
        lags = np.arange(-len(row0) + 1, len(row0))
        peak = lags[np.argmax(cc)]
        shifts.append(peak)

    shifts = np.array(shifts, dtype=float)
    mean_shift_px = np.median(shifts)  # robust to outliers
    rate_um_per_min = abs(mean_shift_px) * pixel_size / frame_interval * 60.0

    return {
        "shifts_px": shifts,
        "mean_shift_px_per_frame": mean_shift_px,
        "rate_um_per_min": rate_um_per_min,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(stack, lines, kymographs, measurements,
                 pixel_size, frame_interval, output_dir):
    """Generate a multi-panel figure with the annotated image, kymographs,
    and a summary bar chart of flow rates."""

    n_lines = len(lines)
    fig = plt.figure(figsize=(5 + 5 * n_lines, 10))

    # -- Panel 1: annotated first frame --
    ax_img = fig.add_subplot(2, n_lines + 1, 1)
    ax_img.imshow(stack[0], cmap="gray")
    for i, line in enumerate(lines):
        (x0, y0), (x1, y1) = line
        ax_img.plot([x0, x1], [y0, y1], "r-", linewidth=1.5)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax_img.text(mx, my, f"L{i+1}", color="yellow", fontsize=9,
                    fontweight="bold", ha="center", va="bottom")
    ax_img.set_title("Growth cone + lines")
    ax_img.axis("off")

    # -- Kymograph panels --
    for i in range(n_lines):
        ax_k = fig.add_subplot(2, n_lines + 1, 2 + i)
        kymo = kymographs[i]
        t_extent = kymo.shape[0] * frame_interval  # seconds
        d_extent = kymo.shape[1] * pixel_size       # μm
        ax_k.imshow(kymo, cmap="gray", aspect="auto",
                    extent=[0, d_extent, t_extent, 0])
        ax_k.set_xlabel("Distance (μm)")
        ax_k.set_ylabel("Time (s)")
        rate = measurements[i]["rate_um_per_min"]
        ax_k.set_title(f"L{i+1} kymograph\n{rate:.1f} μm/min")

    # -- Bar chart of flow rates --
    ax_bar = fig.add_subplot(2, 1, 2)
    labels = [f"L{i+1}" for i in range(n_lines)]
    rates = [m["rate_um_per_min"] for m in measurements]
    ax_bar.bar(labels, rates, color="steelblue", edgecolor="black")
    ax_bar.set_ylabel("Actin flow rate (μm/min)")
    ax_bar.set_title("Actin retrograde flow")
    for j, r in enumerate(rates):
        ax_bar.text(j, r + 0.2, f"{r:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "actin_flow_results.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved results figure: {out_path}")
    plt.show()

    # -- Per-line shift plots --
    fig2, axes2 = plt.subplots(1, n_lines, figsize=(5 * n_lines, 3),
                               squeeze=False)
    for i in range(n_lines):
        ax = axes2[0, i]
        shifts = measurements[i]["shifts_px"]
        ax.plot(np.arange(len(shifts)) * frame_interval, shifts * pixel_size,
                "k.-", markersize=4)
        ax.axhline(measurements[i]["mean_shift_px_per_frame"] * pixel_size,
                    color="red", linestyle="--", label="median")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Shift (μm)")
        ax.set_title(f"L{i+1} frame-to-frame shift")
        ax.legend(fontsize=8)
    fig2.tight_layout()
    out_path2 = os.path.join(output_dir, "actin_flow_shifts.png")
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"Saved shift plots: {out_path2}")
    plt.show()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(lines, measurements, pixel_size, frame_interval, output_dir):
    """Write a summary CSV and per-line shift CSVs."""
    summary_path = os.path.join(output_dir, "flow_rate_summary.csv")
    with open(summary_path, "w") as f:
        f.write("line,x0,y0,x1,y1,length_um,rate_um_per_min,median_shift_px\n")
        for i, (line, m) in enumerate(zip(lines, measurements)):
            (x0, y0), (x1, y1) = line
            length = np.hypot(x1 - x0, y1 - y0) * pixel_size
            f.write(f"L{i+1},{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f},"
                    f"{length:.2f},{m['rate_um_per_min']:.2f},"
                    f"{m['mean_shift_px_per_frame']:.2f}\n")
    print(f"Saved summary: {summary_path}")

    for i, m in enumerate(measurements):
        shift_path = os.path.join(output_dir, f"L{i+1}_shifts.csv")
        with open(shift_path, "w") as f:
            f.write("frame_pair,time_s,shift_px,shift_um\n")
            for j, s in enumerate(m["shifts_px"]):
                f.write(f"{j}-{j+1},{j * frame_interval:.1f},"
                        f"{s:.2f},{s * pixel_size:.4f}\n")
        print(f"Saved: {shift_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure actin retrograde flow from TIRF image stacks."
    )
    parser.add_argument("image_stack",
                        help="Path to a multi-frame TIFF or directory of TIFFs")
    parser.add_argument("--pixel-size", type=float, default=0.108,
                        help="μm per pixel (default: 0.108)")
    parser.add_argument("--frame-interval", type=float, default=2.0,
                        help="Seconds between frames (default: 2.0)")
    parser.add_argument("--line-width", type=int, default=3,
                        help="Width of sampling band around each line in px (default: 3)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    # Load
    stack = load_stack(args.image_stack)

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        p = Path(args.image_stack)
        out_dir = str(p.parent if p.is_file() else p) + "_results"
    os.makedirs(out_dir, exist_ok=True)

    # Draw lines interactively
    print("Draw lines on the growth cone along the actin flow axes.")
    print("Click two points per line. Press Enter or click Done when finished.")
    drawer = LineDrawer(stack[0])
    lines = drawer.run()

    if not lines:
        sys.exit("No lines drawn — exiting.")

    print(f"\n{len(lines)} line(s) drawn. Generating kymographs...")

    # Generate kymographs + measure
    kymographs = []
    measurements = []
    for i, line in enumerate(lines):
        kymo = make_kymograph(stack, line, width=args.line_width)
        kymographs.append(kymo)
        m = measure_flow_rate(kymo, args.pixel_size, args.frame_interval)
        measurements.append(m)
        print(f"  L{i+1}: {m['rate_um_per_min']:.2f} μm/min "
              f"(median shift: {m['mean_shift_px_per_frame']:.2f} px/frame)")

    # Save kymographs as TIFFs
    for i, kymo in enumerate(kymographs):
        kymo_path = os.path.join(out_dir, f"L{i+1}_kymograph.tif")
        tifffile.imwrite(kymo_path, kymo.astype(np.float32))
        print(f"  Saved kymograph TIFF: {kymo_path}")

    # Plot & export
    plot_results(stack, lines, kymographs, measurements,
                 args.pixel_size, args.frame_interval, out_dir)
    export_csv(lines, measurements, args.pixel_size, args.frame_interval, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
