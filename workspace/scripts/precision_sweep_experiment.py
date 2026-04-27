import csv
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import utils as utl


MACRO = "jia_jssc_2020"
DNN = "resnet18"
LAYER_FILE = "01.yaml"
BITS = [1, 2, 4, 8]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_sweep():
    layer_path = utl.path_from_model_dir("workloads", DNN, LAYER_FILE)
    out_dir = ensure_dir(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "outputs",
                "experiments",
            )
        )
    )

    rows = []
    for input_bits, weight_bits in itertools.product(BITS, BITS):
        print(f"Running INPUT_BITS={input_bits}, WEIGHT_BITS={weight_bits}", flush=True)
        result = utl.run_layer(
            macro=MACRO,
            layer=layer_path,
            variables={
                "INPUT_BITS": input_bits,
                "WEIGHT_BITS": weight_bits,
            },
            system="ws_dummy_buffer_many_macro",
        )

        rows.append(
            {
                "macro": MACRO,
                "workload": f"{DNN}/{LAYER_FILE}",
                "input_bits": input_bits,
                "weight_bits": weight_bits,
                "energy_per_mac_j": result.per_compute("energy"),
                "energy_per_mac_pj": result.per_compute("energy") * 1e12,
                "tops_per_w": result.tops_per_w,
                "tops": result.tops,
                "tops_per_mm2": result.tops_per_mm2,
            }
        )

    csv_path = os.path.join(out_dir, "precision_sweep_jia_jssc_2020_resnet18_01.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tops_per_w_grid = np.zeros((len(BITS), len(BITS)))
    energy_per_mac_grid = np.zeros((len(BITS), len(BITS)))

    for row in rows:
        i = BITS.index(row["input_bits"])
        j = BITS.index(row["weight_bits"])
        tops_per_w_grid[i, j] = row["tops_per_w"]
        energy_per_mac_grid[i, j] = row["energy_per_mac_pj"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im0 = axes[0].imshow(tops_per_w_grid, cmap="viridis", origin="lower")
    axes[0].set_title("Energy Efficiency (TOPS/W)")
    axes[0].set_xlabel("Weight Bits")
    axes[0].set_ylabel("Input Bits")
    axes[0].set_xticks(range(len(BITS)), BITS)
    axes[0].set_yticks(range(len(BITS)), BITS)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(energy_per_mac_grid, cmap="magma", origin="lower")
    axes[1].set_title("Energy per MAC (pJ)")
    axes[1].set_xlabel("Weight Bits")
    axes[1].set_ylabel("Input Bits")
    axes[1].set_xticks(range(len(BITS)), BITS)
    axes[1].set_yticks(range(len(BITS)), BITS)
    fig.colorbar(im1, ax=axes[1])

    heatmap_path = os.path.join(
        out_dir, "precision_sweep_jia_jssc_2020_resnet18_01_heatmaps.png"
    )
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    diagonal = sorted(
        [row for row in rows if row["input_bits"] == row["weight_bits"]],
        key=lambda row: row["input_bits"],
    )

    x = [row["input_bits"] for row in diagonal]
    y_eff = [row["tops_per_w"] for row in diagonal]
    y_energy = [row["energy_per_mac_pj"] for row in diagonal]

    fig2, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(x, y_eff, marker="o", color="tab:blue", label="TOPS/W")
    ax_right.plot(x, y_energy, marker="s", color="tab:red", label="pJ/MAC")

    ax_left.set_xlabel("Input Bits = Weight Bits")
    ax_left.set_ylabel("Energy Efficiency (TOPS/W)", color="tab:blue")
    ax_right.set_ylabel("Energy per MAC (pJ)", color="tab:red")
    ax_left.set_title("Precision Tradeoff (Diagonal Sweep)")
    ax_left.set_xticks(x)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    lineplot_path = os.path.join(
        out_dir, "precision_sweep_jia_jssc_2020_resnet18_01_diagonal.png"
    )
    fig2.tight_layout()
    fig2.savefig(lineplot_path, dpi=200)
    plt.close(fig2)

    print("\nExperiment complete.")
    print(f"CSV: {csv_path}")
    print(f"Heatmaps: {heatmap_path}")
    print(f"Diagonal plot: {lineplot_path}")


if __name__ == "__main__":
    run_sweep()
