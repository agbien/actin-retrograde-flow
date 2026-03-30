"""
Actin Retrograde Flow Measurement Tool

Measures retrograde flow rate of actin in neuronal growth cones from
TIRF microscopy image stacks. The user draws lines along flow axes on
the growth cone image, and the tool generates kymographs and uses a
vision model (Claude Sonnet) to identify diagonal streaks and compute
flow rates from their slopes via point-slope form.

Usage:
    python actin_retrograde_flow.py <image_stack> [--pixel-size 0.108] [--frame-interval 2.0]

    image_stack:      Path to a TIFF stack (multi-frame) or a directory of
                      sequentially numbered single-frame TIFFs.
    --pixel-size:     um per pixel (default: 0.108 um/px, typical for 60x TIRF)
    --frame-interval: seconds between frames (default: 2.0 s)

Requires ANTHROPIC_API_KEY environment variable to be set.
"""

import argparse
import sys
import os
import re
import json
import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import map_coordinates
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
    if stack.ndim == 4:
        stack = stack[:, 0]
    print(f"Loaded stack: {stack.shape[0]} frames, {stack.shape[1]}x{stack.shape[2]} px")
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
        self.lines = []
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

        ax_btn = self.fig.add_axes([0.8, 0.01, 0.15, 0.04])
        self._btn = Button(ax_btn, "Done")
        self._btn.on_clicked(lambda _: self._finish())

        plt.show()
        return self.lines

    def _on_click(self, event):
        if event.inaxes != self.ax or self.done:
            return
        if event.button == 3:
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
            mx, my = (x0 + x) / 2, (y0 + y) / 2
            self.ax.text(mx, my, f"L{len(self.lines)}", color="yellow",
                         fontsize=9, fontweight="bold",
                         ha="center", va="bottom")
            self._current_pt = None
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

    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -dy, dx

    offsets = np.arange(-(width // 2), width // 2 + 1)
    t_vals = np.linspace(0, 1, n_pts)

    kymo = np.zeros((stack.shape[0], n_pts))

    for frame_idx in range(stack.shape[0]):
        frame = stack[frame_idx]
        accum = np.zeros(n_pts)
        for off in offsets:
            xs = x0 + t_vals * (x1 - x0) + off * nx
            ys = y0 + t_vals * (y1 - y0) + off * ny
            coords = np.vstack([ys, xs])
            accum += map_coordinates(frame, coords, order=1, mode="nearest")
        kymo[frame_idx] = accum / len(offsets)

    return kymo


def render_kymograph_image(kymo: np.ndarray, pixel_size: float,
                           frame_interval: float, label: str = "") -> bytes:
    """Render a kymograph as a PNG image with calibrated axes.
    Returns the PNG bytes."""
    t_extent = kymo.shape[0] * frame_interval
    d_extent = kymo.shape[1] * pixel_size

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(kymo, cmap="gray", aspect="auto",
              extent=[0, d_extent, t_extent, 0])
    ax.set_xlabel("Distance (um)", fontsize=14)
    ax.set_ylabel("Time (s)", fontsize=14)
    if label:
        ax.set_title(label, fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Vision model flow rate measurement
# ---------------------------------------------------------------------------

KYMOGRAPH_PROMPT = """You are analyzing a kymograph image from TIRF microscopy of actin retrograde flow in a neuronal growth cone.

In this image:
- The x-axis shows distance in micrometers (um).
- The y-axis shows time in seconds (s), increasing downward.
- Diagonal streaks represent moving actin features.
- The slope of a streak gives the flow velocity.

Your task:
1. Identify the clearest diagonal streak(s) — these are lines that are tilted, NOT vertical (vertical = stationary).
2. For each streak, read two points precisely from the axis labels: (x1 um, t1 s) and (x2 um, t2 s).
3. Compute the flow rate for each streak: rate = |x2 - x1| / |t2 - t1| * 60 um/min

You MUST respond in this exact JSON format and nothing else:
{
  "streaks": [
    {"x1": 0.0, "t1": 0.0, "x2": 0.0, "t2": 0.0, "rate_um_per_min": 0.0}
  ]
}

If you identify multiple streaks, include them all. Read the axis values carefully."""


def measure_flow_rate_vision(kymo_png: bytes, api_key: str,
                             model: str = "claude-sonnet-4-6") -> dict:
    """Send a kymograph image to a Claude vision model and extract
    the flow rate from its analysis of diagonal streak slopes.

    Args:
        kymo_png: PNG image bytes of the rendered kymograph
        api_key:  Anthropic API key
        model:    Model to use (default: claude-sonnet-4-6)

    Returns:
        dict with rate_um_per_min and streak details
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = base64.b64encode(kymo_png).decode()

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": KYMOGRAPH_PROMPT,
                },
            ],
        }],
    )

    response_text = message.content[0].text.strip()

    # Extract JSON from response
    try:
        # Try direct parse
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"    Warning: Could not parse model response as JSON")
                print(f"    Response: {response_text[:200]}")
                return {"streaks": [], "rate_um_per_min": 0.0}
        else:
            print(f"    Warning: No JSON found in model response")
            print(f"    Response: {response_text[:200]}")
            return {"streaks": [], "rate_um_per_min": 0.0}

    streaks = data.get("streaks", [])
    if not streaks:
        return {"streaks": [], "rate_um_per_min": 0.0}

    # Validate and recompute rates from the coordinates
    valid_streaks = []
    for s in streaks:
        try:
            x1, t1 = float(s["x1"]), float(s["t1"])
            x2, t2 = float(s["x2"]), float(s["t2"])
            dt = abs(t2 - t1)
            dx = abs(x2 - x1)
            if dt < 0.1:
                continue
            rate = dx / dt * 60.0
            valid_streaks.append({
                "x1": x1, "t1": t1, "x2": x2, "t2": t2,
                "rate_um_per_min": rate,
            })
        except (KeyError, ValueError, TypeError):
            continue

    if not valid_streaks:
        return {"streaks": [], "rate_um_per_min": 0.0}

    avg_rate = np.mean([s["rate_um_per_min"] for s in valid_streaks])
    return {
        "streaks": valid_streaks,
        "rate_um_per_min": float(avg_rate),
    }


def measure_flow_rate(kymo: np.ndarray, pixel_size: float,
                      frame_interval: float, api_key: str = None,
                      model: str = "claude-sonnet-4-6") -> dict:
    """Measure flow rate from a kymograph.

    If an API key is provided, uses Claude Sonnet vision to read the
    kymograph slopes. Otherwise falls back to manual interactive mode.
    """
    if api_key:
        kymo_png = render_kymograph_image(kymo, pixel_size, frame_interval)
        return measure_flow_rate_vision(kymo_png, api_key, model)
    else:
        # Fallback: manual interactive slope picking
        return _measure_flow_rate_interactive(kymo, pixel_size, frame_interval)


def _measure_flow_rate_interactive(kymo, pixel_size, frame_interval):
    """Fallback: interactive matplotlib window where user clicks points
    along streaks and numpy.polyfit fits a line."""
    t_extent = kymo.shape[0] * frame_interval
    d_extent = kymo.shape[1] * pixel_size

    print("    Interactive mode: click points along a streak, right-click to fit.")
    print("    Press Enter when done.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(kymo, cmap="gray", aspect="auto",
              extent=[0, d_extent, t_extent, 0])
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Click points along streak, right-click to fit. Enter to finish.")

    state = {"pts": [], "streaks": [], "dots": []}

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 3:  # right-click = fit
            if len(state["pts"]) >= 2:
                pts = np.array(state["pts"])
                coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
                rate = abs(coeffs[0]) * 60.0
                state["streaks"].append({"rate_um_per_min": rate})
                t_fit = np.array([pts[:, 1].min(), pts[:, 1].max()])
                x_fit = np.polyval(coeffs, t_fit)
                ax.plot(x_fit, t_fit, "r-", lw=2)
                ax.text(x_fit.mean(), t_fit.mean(), f"{rate:.1f} um/min",
                        color="yellow", fontsize=10, fontweight="bold")
                fig.canvas.draw_idle()
                print(f"      Streak: {rate:.1f} um/min")
            for d in state["dots"]:
                d.remove()
            state["dots"].clear()
            state["pts"].clear()
            fig.canvas.draw_idle()
            return
        state["pts"].append((event.xdata, event.ydata))
        dot, = ax.plot(event.xdata, event.ydata, "r+", ms=12, mew=2)
        state["dots"].append(dot)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            if state["pts"] and len(state["pts"]) >= 2:
                # Fit remaining points
                pts = np.array(state["pts"])
                coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
                rate = abs(coeffs[0]) * 60.0
                state["streaks"].append({"rate_um_per_min": rate})
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not state["streaks"]:
        return {"streaks": [], "rate_um_per_min": 0.0}
    rates = [s["rate_um_per_min"] for s in state["streaks"]]
    return {"streaks": state["streaks"], "rate_um_per_min": np.mean(rates)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(stack, lines, kymographs, measurements,
                 pixel_size, frame_interval, output_dir):
    """Generate a multi-panel figure with the annotated image, kymographs
    with detected slopes overlaid, and a summary bar chart of flow rates."""

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

    # -- Kymograph panels with detected streak lines --
    for i in range(n_lines):
        ax_k = fig.add_subplot(2, n_lines + 1, 2 + i)
        kymo = kymographs[i]
        t_extent = kymo.shape[0] * frame_interval
        d_extent = kymo.shape[1] * pixel_size
        ax_k.imshow(kymo, cmap="gray", aspect="auto",
                    extent=[0, d_extent, t_extent, 0])
        ax_k.set_xlabel("Distance (um)")
        ax_k.set_ylabel("Time (s)")
        rate = measurements[i]["rate_um_per_min"]
        ax_k.set_title(f"L{i+1} kymograph\n{rate:.1f} um/min")

        # Overlay detected streak lines from vision model
        for streak in measurements[i].get("streaks", []):
            if "x1" in streak and "t1" in streak:
                ax_k.plot([streak["x1"], streak["x2"]],
                          [streak["t1"], streak["t2"]],
                          "r-", linewidth=2, alpha=0.8)
                mx = (streak["x1"] + streak["x2"]) / 2
                mt = (streak["t1"] + streak["t2"]) / 2
                ax_k.plot(streak["x1"], streak["t1"], "r+", ms=10, mew=2)
                ax_k.plot(streak["x2"], streak["t2"], "r+", ms=10, mew=2)

    # -- Bar chart of flow rates --
    ax_bar = fig.add_subplot(2, 1, 2)
    labels = [f"L{i+1}" for i in range(n_lines)]
    rates = [m["rate_um_per_min"] for m in measurements]
    ax_bar.bar(labels, rates, color="steelblue", edgecolor="black")
    ax_bar.set_ylabel("Actin flow rate (um/min)")
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
    """Write a summary CSV with flow rates and streak coordinates."""
    summary_path = os.path.join(output_dir, "flow_rate_summary.csv")
    with open(summary_path, "w") as f:
        f.write("line,x0_px,y0_px,x1_px,y1_px,length_um,"
                "rate_um_per_min,n_streaks\n")
        for i, (line, m) in enumerate(zip(lines, measurements)):
            (x0, y0), (x1, y1) = line
            length = np.hypot(x1 - x0, y1 - y0) * pixel_size
            n_streaks = len(m.get("streaks", []))
            f.write(f"L{i+1},{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f},"
                    f"{length:.2f},{m['rate_um_per_min']:.2f},{n_streaks}\n")
    print(f"Saved summary: {summary_path}")

    # Per-line streak details
    for i, m in enumerate(measurements):
        streaks = m.get("streaks", [])
        if not streaks:
            continue
        streak_path = os.path.join(output_dir, f"L{i+1}_streaks.csv")
        with open(streak_path, "w") as f:
            f.write("streak,x1_um,t1_s,x2_um,t2_s,rate_um_per_min\n")
            for j, s in enumerate(streaks):
                f.write(f"{j+1},{s.get('x1',0):.2f},{s.get('t1',0):.2f},"
                        f"{s.get('x2',0):.2f},{s.get('t2',0):.2f},"
                        f"{s['rate_um_per_min']:.2f}\n")
        print(f"Saved: {streak_path}")


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
                        help="um per pixel (default: 0.108)")
    parser.add_argument("--frame-interval", type=float, default=2.0,
                        help="Seconds between frames (default: 2.0)")
    parser.add_argument("--line-width", type=int, default=3,
                        help="Width of sampling band in px (default: 3)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Vision model to use (default: claude-sonnet-4-6)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)")
    parser.add_argument("--interactive", action="store_true",
                        help="Use interactive manual mode instead of vision model")
    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.interactive:
        print("Warning: No ANTHROPIC_API_KEY set. Falling back to interactive mode.")
        print("Set ANTHROPIC_API_KEY or use --interactive flag.\n")
        args.interactive = True

    if not args.interactive:
        try:
            import anthropic
        except ImportError:
            sys.exit("Error: 'anthropic' package not installed. "
                     "Run: pip install anthropic")

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
        sys.exit("No lines drawn -- exiting.")

    print(f"\n{len(lines)} line(s) drawn. Generating kymographs...")

    # Generate kymographs
    kymographs = []
    for i, line in enumerate(lines):
        kymo = make_kymograph(stack, line, width=args.line_width)
        kymographs.append(kymo)

    # Measure flow rates
    if args.interactive:
        print("\nInteractive mode: click points along streaks in each kymograph.")
    else:
        print(f"\nAnalyzing kymographs with {args.model}...")

    measurements = []
    for i, kymo in enumerate(kymographs):
        print(f"  L{i+1}:", end=" ", flush=True)
        m = measure_flow_rate(
            kymo, args.pixel_size, args.frame_interval,
            api_key=None if args.interactive else api_key,
            model=args.model,
        )
        measurements.append(m)
        n_streaks = len(m.get("streaks", []))
        print(f"{m['rate_um_per_min']:.2f} um/min ({n_streaks} streak(s))")

    # Save kymographs as TIFFs
    for i, kymo in enumerate(kymographs):
        kymo_path = os.path.join(out_dir, f"L{i+1}_kymograph.tif")
        tifffile.imwrite(kymo_path, kymo.astype(np.float32))

    # Save kymograph PNGs (for reference)
    for i, kymo in enumerate(kymographs):
        png_bytes = render_kymograph_image(
            kymo, args.pixel_size, args.frame_interval,
            label=f"L{i+1} kymograph")
        png_path = os.path.join(out_dir, f"L{i+1}_kymograph.png")
        with open(png_path, "wb") as f:
            f.write(png_bytes)

    # Plot & export
    plot_results(stack, lines, kymographs, measurements,
                 args.pixel_size, args.frame_interval, out_dir)
    export_csv(lines, measurements, args.pixel_size, args.frame_interval, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
