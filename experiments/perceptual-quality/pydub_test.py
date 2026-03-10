from data_loader.dereverberation_dataset import DereverberationDataset
from utils.load_data import load_data
from pathlib import Path

import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = load_data(Path(sys.argv[1]))

    train_dataset = DereverberationDataset(
        data.train_files,
        segment_length=44100 * 4,
        sample_rate=44100,
    )

    i = 0
    while True:
        print(i)
        result = train_dataset.__getitem__(i)
        mask = result["mask"]
        reverb_audio = result["reverb_audio"]
        original_audio = result["original_audio"]

        print(mask.mean(dim=0))
        if mask.mean(dim=0) != 1:
            fig, ax = plt.subplots(figsize=(12, 5))

            COLOR_REVERB = "#1f77b4"
            COLOR_ORIGINAL = "#D1D5DB"
            COLOR_MASK = "#dca7ae"

            reverb_np = reverb_audio.cpu().numpy()
            original_np = original_audio.cpu().numpy()
            mask_np = mask.cpu().numpy()

            x = np.arange(len(reverb_np))

            ax.fill_between(x, mask_np, alpha=0.25, color=COLOR_MASK, zorder=0)
            ax.plot(x, mask_np, color="#9CA3AF", linewidth=0.6, alpha=0.5, zorder=1)
            ax.set_ylim(-1.05, 1.05)
            ax.spines["right"].set_color("#E5E7EB")
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))

            ax.plot(
                x,
                reverb_np,
                color=COLOR_REVERB,
                linewidth=1.6,
                alpha=1.0,
                label="Reverb Audio",
                zorder=3,
            )

            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#E5E7EB")
            ax.spines["bottom"].set_color("#E5E7EB")

            ax.tick_params(colors="#000000", labelsize=10, length=0)
            ax.set_xlabel("Sample", fontsize=11, color="#000000", labelpad=8)
            ax.set_ylabel("Amplitude", fontsize=11, color="#000000", labelpad=8)

            ax.grid(axis="y", color="#F3F4F6", linewidth=0.8, linestyle="--")

            lines1, labels1 = ax.get_legend_handles_labels()
            from matplotlib.patches import Patch

            mask_patch = Patch(
                facecolor=COLOR_MASK, edgecolor="#9CA3AF", alpha=0.6, label="Mask"
            )
            ax.legend(
                handles=lines1 + [mask_patch],
                loc="upper right",
                frameon=False,
                labelcolor="#374151",
                fontsize=10,
            )

            plt.tight_layout()
            plt.savefig("plots/mask_plot_node_test.svg")

            sys.exit()

        i += 1


if __name__ == "__main__":
    main()
