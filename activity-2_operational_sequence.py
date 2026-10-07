import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# Core DSP helpers
def sort_by_n(n, y):
    """Sort (n, y) pairs by ascending index n, for clean stem plots."""
    n = np.asarray(n)
    y = np.asarray(y)
    order = np.argsort(n)
    return n[order], y[order]


def compute_all_operations(x, n0, k):
    """
    Given sequence values x (array), its starting index n0, and shift
    amount k, return a dict with the index/value arrays for:
    original, delay, advance, reversal, composite method 1, composite method 2.
    """
    x = np.asarray(x, dtype=float)
    n_x = np.arange(n0, n0 + len(x))

    # 1. Time Delay: y[n] = x[n - k]  ->  n_y = n_x + k, values unchanged
    n_delay, y_delay = sort_by_n(n_x + k, x)

    # 2. Time Advance: y[n] = x[n + k]  ->  n_y = n_x - k, values unchanged
    n_adv, y_adv = sort_by_n(n_x - k, x)

    # 3. Time Reversal: y[n] = x[-n]  ->  n_y = -n_x, values paired as-is
    n_rev, y_rev = sort_by_n(-n_x, x)

    # 4. Composite Method 1 (Shift then Fold):
    #    v[n] = x[n + k]  -> n_v = n_x - k, values = x
    #    y[n] = v[-n]     -> n_y = -n_v = k - n_x, values = x
    n_v1, y_v1 = sort_by_n(n_x - k, x)              # intermediate v[n]
    n_c1, y_c1 = sort_by_n(k - n_x, x)              # final y[n]

    # 5. Composite Method 2 (Fold then Shift):
    #    v[n] = x[-n]      -> n_v = -n_x, values = x
    #    y[n] = v[n - k]   -> n_y = n_v + k = k - n_x, values = x
    n_v2, y_v2 = sort_by_n(-n_x, x)                 # intermediate v[n]
    n_c2, y_c2 = sort_by_n(k - n_x, x)              # final y[n]

    energy = float(np.sum(np.abs(x) ** 2))

    return {
        "original": (n_x, x),
        "delay": (n_delay, y_delay),
        "advance": (n_adv, y_adv),
        "reversal": (n_rev, y_rev),
        "composite1_intermediate": (n_v1, y_v1),
        "composite1_final": (n_c1, y_c1),
        "composite2_intermediate": (n_v2, y_v2),
        "composite2_final": (n_c2, y_c2),
        "energy": energy,
    }


def format_sequence(n, y, label, name="y"):
    """Return a human-readable multi-line string for a result sequence."""
    lines = [f"{label}:"]
    pairs = ", ".join(f"{name}[{int(ni)}]={yi:g}" for ni, yi in zip(n, y))
    lines.append("  " + pairs)
    values_only = "{" + ", ".join(f"{yi:g}" for yi in y) + "}"
    lines.append(f"  Result: {name}[n] = {values_only}"
                 f"  for n = {int(n[0])}..{int(n[-1])}")
    return "\n".join(lines)


# Tkinter application
class SequenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Discrete-Time Sequence Operations Visualizer")
        self.root.geometry("1250x850")

        self._build_input_frame()
        self._build_output_text()
        self._build_plot_area()

        # Defaults: x[n] = {1, 2, 3, 4, 5} for n = -2..2, k = 2
        self.compute_and_plot()

    # ---------------- UI construction ----------------
    def _build_input_frame(self):
        frame = ttk.LabelFrame(self.root, text="Input")
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(frame, text="Sequence x[n] values (comma-separated):").grid(
            row=0, column=0, sticky="w", padx=5, pady=5)
        self.seq_entry = ttk.Entry(frame, width=40)
        self.seq_entry.insert(0, "1, 2, 3, 4, 5")
        self.seq_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame, text="Starting index n0:").grid(
            row=0, column=2, sticky="w", padx=5, pady=5)
        self.n0_entry = ttk.Entry(frame, width=8)
        self.n0_entry.insert(0, "-2")
        self.n0_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(frame, text="Shift amount k:").grid(
            row=0, column=4, sticky="w", padx=5, pady=5)
        self.k_entry = ttk.Entry(frame, width=8)
        self.k_entry.insert(0, "2")
        self.k_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        compute_btn = ttk.Button(frame, text="Compute & Plot",
                                 command=self.compute_and_plot)
        compute_btn.grid(row=0, column=6, padx=10, pady=5)

        reset_btn = ttk.Button(frame, text="Reset Example",
                               command=self.reset_defaults)
        reset_btn.grid(row=0, column=7, padx=5, pady=5)

    def _build_output_text(self):
        frame = ttk.LabelFrame(self.root, text="Numeric Results")
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        self.output_text = ScrolledText(frame, height=11, wrap=tk.WORD,
                                        font=("Consolas", 10))
        self.output_text.pack(fill=tk.X, padx=5, pady=5)

    def _build_plot_area(self):
        frame = ttk.LabelFrame(self.root, text="Graphs")
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.axes = self.fig.subplots(2, 3)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    # ---------------- Actions ----------------
    def reset_defaults(self):
        self.seq_entry.delete(0, tk.END)
        self.seq_entry.insert(0, "1, 2, 3, 4, 5")
        self.n0_entry.delete(0, tk.END)
        self.n0_entry.insert(0, "-2")
        self.k_entry.delete(0, tk.END)
        self.k_entry.insert(0, "2")
        self.compute_and_plot()

    def _parse_inputs(self):
        raw_seq = self.seq_entry.get().strip()
        if not raw_seq:
            raise ValueError("Please enter at least one sequence value.")
        try:
            x = [float(v.strip()) for v in raw_seq.split(",") if v.strip() != ""]
        except ValueError:
            raise ValueError("Sequence values must be numbers separated by commas, "
                             "e.g. 1, 2, 3, 4, 5")
        if len(x) == 0:
            raise ValueError("Please enter at least one sequence value.")

        try:
            n0 = int(self.n0_entry.get().strip())
        except ValueError:
            raise ValueError("Starting index n0 must be an integer.")

        try:
            k = int(self.k_entry.get().strip())
        except ValueError:
            raise ValueError("Shift amount k must be an integer.")

        return x, n0, k

    def compute_and_plot(self):
        try:
            x, n0, k = self._parse_inputs()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        results = compute_all_operations(x, n0, k)
        self._update_text(results, x, n0, k)
        self._update_plots(results, k)

    # ---------------- Rendering ----------------
    def _update_text(self, results, x, n0, k):
        n_x, x_arr = results["original"]
        n_delay, y_delay = results["delay"]
        n_adv, y_adv = results["advance"]
        n_rev, y_rev = results["reversal"]
        n_c1, y_c1 = results["composite1_final"]
        n_c2, y_c2 = results["composite2_final"]

        lines = []
        lines.append(f"Original Sequence: x[n] over n = {n0}..{n0 + len(x) - 1}")
        lines.append("  x[n] = " + "{" + ", ".join(f"{v:g}" for v in x_arr) + "}")
        lines.append("")
        lines.append(format_sequence(n_delay, y_delay, f"1) Time Delay: y[n] = x[n - {k}]"))
        lines.append("")
        lines.append(format_sequence(n_adv, y_adv, f"2) Time Advance: y[n] = x[n + {k}]"))
        lines.append("")
        lines.append(format_sequence(n_rev, y_rev, "3) Time Reversal: y[n] = x[-n]"))
        lines.append("")
        lines.append(format_sequence(
            n_c1, y_c1,
            f"4) Composite Method 1 (Shift then Fold): y[n] = x[-n + {k}]"))
        lines.append("")
        lines.append(format_sequence(
            n_c2, y_c2,
            f"5) Composite Method 2 (Fold then Shift): y[n] = x[-n + {k}]"))
        lines.append("")
        same = np.array_equal(n_c1, n_c2) and np.allclose(y_c1, y_c2)
        lines.append(f"Note: Method 1 and Method 2 give the SAME result "
                     f"({'confirmed match' if same else 'MISMATCH - check inputs'}), "
                     f"as expected since both compute y[n] = x[-n + {k}].")
        lines.append("")
        lines.append(f"Energy of original sequence: E = sum |x[n]|^2 = {results['energy']:g}")

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "\n".join(lines))

    def _update_plots(self, results, k):
        specs = [
            ("original", "Original: x[n]"),
            ("delay", f"1) Time Delay: y[n] = x[n - {k}]"),
            ("advance", f"2) Time Advance: y[n] = x[n + {k}]"),
            ("reversal", "3) Time Reversal: y[n] = x[-n]"),
            ("composite1_final", f"4) Composite M1 (Shift->Fold): y[n]=x[-n+{k}]"),
            ("composite2_final", f"5) Composite M2 (Fold->Shift): y[n]=x[-n+{k}]"),
        ]

        for ax, (key, title) in zip(self.axes.flat, specs):
            n, y = results[key]
            ax.clear()
            ax.stem(n, y, basefmt=" ")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("n", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.4)
            ax.axhline(0, color="black", lw=0.6)
            ax.axvline(0, color="black", lw=0.6)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = SequenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()