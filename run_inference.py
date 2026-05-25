import argparse
import csv
from pathlib import Path

from inference.detector import RFRDetector


def save_results_csv(results, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_idx",
        "frame_name",
        "object_id",
        "x_center",
        "y_center",
        "x",
        "y",
        "width",
        "height",
        "area",
        "mask_path",
        "overlay_path",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="RFR ResUNet inference app")

    parser.add_argument("--input_dir", required=True, help="Folder with input frames")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth.tar checkpoint")
    parser.add_argument("--config", default="configs/resunet_rfr.json", help="Path to config json")
    parser.add_argument("--output_dir", default="outputs", help="Folder for output masks and csv")

    args = parser.parse_args()

    detector = RFRDetector(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    results = detector.predict_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    output_csv = Path(args.output_dir) / "results.csv"
    save_results_csv(results, output_csv)

    print("Done")
    print("Objects:", len(results))
    print("CSV:", output_csv)


if __name__ == "__main__":
    main()