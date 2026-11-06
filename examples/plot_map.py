"""Plot the exact published Spacious 2D coordinates, colored by broad topic."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_data import Atlas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="atlas.png")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    atlas = Atlas(args.data)
    coordinates = atlas.coordinates()
    topics = atlas.topic_assignments()
    labels = sorted({row["broad_topic"] for row in topics})
    colors = {label: index for index, label in enumerate(labels)}
    figure, axes = plt.subplots(figsize=(12, 8), facecolor="#05080d")
    axes.set_facecolor("#05080d")
    axes.scatter(coordinates[:, 0], coordinates[:, 1],
                 c=[colors[row["broad_topic"]] for row in topics],
                 cmap="turbo", s=0.5, alpha=0.65, linewidths=0, rasterized=True)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.savefig(args.output, dpi=250, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
