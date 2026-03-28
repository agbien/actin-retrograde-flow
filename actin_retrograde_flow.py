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
from scipy.ndimage import map_coordinates, gaussian_filter
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
    """Automatically measure the dominant streak slope in a kymograph
    using the structure tensor method.

    The structure tensor captures the local orientation of intensity
    gradients at every pixel. By averaging the tensor over the entire
    kymograph (after removing stationary features), the dominant
    orientation of moving streaks is recovered — even in noisy or
    compressed data where edge detection fails.

    The dominant angle is converted to a flow rate via point-slope form:
        rate (μm/min) = |Δx / Δt| * 60

    where Δx and Δt are read from the kymograph axes (space in μm,
    time in seconds).
    """
    T, N = kymo.shape

    # --- Preprocess: isolate moving features ---
    # Temporal derivative: diff along time axis highlights motion,
    # suppresses stationary structures completely
    kymo_diff = np.diff(kymo.astype(np.float64), axis=0)
    # Also subtract the spatial mean per row to remove global flicker
    kymo_hp = kymo_diff - kymo_diff.mean(axis=1, keepdims=True)
    # Smooth to reduce noise
    kymo_hp = gaussian_filter(kymo_hp, sigma=1.0)
    T, N = kymo_hp.shape

    # --- Structure tensor in physical coordinates ---
    # Compute gradients in physical units so the angle is meaningful.
    # x-axis: space (μm), y-axis: time (s)
    # np.gradient with axis spacing: dx = pixel_size, dy = frame_interval
    Iy, Ix = np.gradient(kymo_hp, frame_interval, pixel_size)

    # Structure tensor components, smoothed over a neighborhood
    sigma_st = max(3.0, min(T, N) * 0.15)
    Jxx = gaussian_filter(Ix * Ix, sigma=sigma_st)
    Jxy = gaussian_filter(Ix * Iy, sigma=sigma_st)
    Jyy = gaussian_filter(Iy * Iy, sigma=sigma_st)

    # Global structure tensor (sum over all pixels)
    S_xx = Jxx.sum()
    S_xy = Jxy.sum()
    S_yy = Jyy.sum()

    # Dominant gradient orientation
    theta = 0.5 * np.arctan2(2 * S_xy, S_xx - S_yy)

    # The gradient is perpendicular to the streak direction.
    # Streak direction = theta + π/2.
    # Streak slope (dx/dt) where x=μm, t=seconds:
    #   streak runs along direction (cos(theta+π/2), sin(theta+π/2))
    #                              = (-sin(theta), cos(theta))
    #   dx/dt = -sin(theta) / cos(theta) = -tan(theta)
    # But we only care about |dx/dt| for the rate.

    if abs(np.cos(theta)) < 1e-6:
        velocity_um_per_s = 0.0
    else:
        velocity_um_per_s = abs(np.sin(theta) / np.cos(theta))

    rate_um_per_min = velocity_um_per_s * 60.0

    # Coherence: how strongly oriented is the structure tensor?
    trace = S_xx + S_yy
    det = S_xx * S_yy - S_xy * S_xy
    discrim = max(0, trace * trace - 4 * det)
    lam1 = (trace + np.sqrt(discrim)) / 2
    lam2 = (trace - np.sqrt(discrim)) / 2
    coherence = ((lam1 - lam2) / (lam1 + lam2)) ** 2 if (lam1 + lam2) > 0 else 0

    return {
        "streaks": [{
            "rate_um_per_min": rate_um_per_min,
            "slope_um_per_s": velocity_um_per_s,
            "angle_deg": np.degrees(theta),
            "coherence": coherence,
        }],
        "rate_um_per_min": rate_um_per_min,
        "coherence": coherence,
        "angle_deg": np.degrees(theta),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(stack, lines, kymographs, measurements,
                 pixel_size, frame_interval, output_dir):
    """Generate a multi-panel figure with the annotated image, kymographs
    with fitted slopes overlaid, and a summary bar chart of flow rates."""

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

    # -- Kymograph panels with fit lines --
    for i in range(n_lines):
        ax_k = fig.add_subplot(2, n_lines + 1, 2 + i)
        kymo = kymographs[i]
        t_extent = kymo.shape[0] * frame_interval
        d_extent = kymo.shape[1] * pixel_size
        ax_k.imshow(kymo, cmap="gray", aspect="auto",
                    extent=[0, d_extent, t_extent, 0])
        ax_k.set_xlabel("Distance (μm)")
        ax_k.set_ylabel("Time (s)")
        rate = measurements[i]["rate_um_per_min"]
        ax_k.set_title(f"L{i+1} kymograph\n{rate:.1f} μm/min")

        # Overlay detected streak lines
        for streak in measurements[i].get("streaks", []):
            angle_rad = np.radians(streak["angle_deg"])
            dist = streak["dist"]
            # Reconstruct line endpoints in pixel space, then convert
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            # Line in pixel coords: x*cos + y*sin = dist
            # Solve for x at y=0 and y=T-1
            if abs(sin_a) > 1e-6:
                x_at_y0 = dist / cos_a if abs(cos_a) > 1e-6 else 0
                y0_px, y1_px = 0, kymo.shape[0] - 1
                x0_px = (dist - y0_px * sin_a) / cos_a if abs(cos_a) > 1e-6 else 0
                x1_px = (dist - y1_px * sin_a) / cos_a if abs(cos_a) > 1e-6 else 0
                ax_k.plot([x0_px * pixel_size, x1_px * pixel_size],
                          [y0_px * frame_interval, y1_px * frame_interval],
                          "r-", linewidth=1.5, alpha=0.7)

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


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(lines, measurements, pixel_size, frame_interval, output_dir):
    """Write a summary CSV with flow rates and fit quality."""
    summary_path = os.path.join(output_dir, "flow_rate_summary.csv")
    with open(summary_path, "w") as f:
        f.write("line,x0,y0,x1,y1,length_um,rate_um_per_min,"
                "n_streaks,slope_um_per_s,r_squared\n")
        for i, (line, m) in enumerate(zip(lines, measurements)):
            (x0, y0), (x1, y1) = line
            length = np.hypot(x1 - x0, y1 - y0) * pixel_size
            streaks = m.get("streaks", [])
            if streaks:
                avg_slope = np.mean([s["slope_um_per_s"] for s in streaks])
                avg_r2 = np.mean([s["r_squared"] for s in streaks])
            else:
                avg_slope = 0.0
                avg_r2 = 0.0
            f.write(f"L{i+1},{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f},"
                    f"{length:.2f},{m['rate_um_per_min']:.2f},"
                    f"{len(streaks)},{avg_slope:.4f},{avg_r2:.3f}\n")
    print(f"Saved summary: {summary_path}")


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

    # Generate kymographs
    kymographs = []
    for i, line in enumerate(lines):
        kymo = make_kymograph(stack, line, width=args.line_width)
        kymographs.append(kymo)

    # Automatically detect streaks and measure flow rates
    measurements = []
    for i, kymo in enumerate(kymographs):
        m = measure_flow_rate(kymo, args.pixel_size, args.frame_interval)
        measurements.append(m)
        n_streaks = len(m.get("streaks", []))
        print(f"  L{i+1}: {m['rate_um_per_min']:.2f} μm/min "
              f"({n_streaks} streak(s) detected)")

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
