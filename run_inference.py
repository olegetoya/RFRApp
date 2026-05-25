import argparse
import csv
from pathlib import Path

from inference.detector import RFRDetector


FIELDNAMES = [
    "frame_idx",
    "frame_name",
    "object_id",
    "x_center",
    "y_center",
    "width",
    "height",
    "area",
    "mask_path",
    "overlay_path",
    "model_name",
]


def save_results_csv(results, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(FIELDNAMES)

    for row in results:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run RFR inference on a folder with image sequence."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to folder with input frames."
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint."
    )

    parser.add_argument(
        "--config",
        default="configs/rfr_models.json",
        help="Path to RFR models config."
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/test_resunet",
        help="Path to output folder."
    )

    parser.add_argument(
        "--model_name",
        default="ResUNet_RFR",
        help="Model name from config, for example: ResUNet_RFR."
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda or cpu. If not set, device is selected automatically."
    )

    args = parser.parse_args()

    detector = RFRDetector(
        config_path=args.config,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    results = detector.predict_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    output_csv = Path(args.output_dir) / "results.csv"
    save_results_csv(results, output_csv)

    print()
    print("Done")
    print("Detected objects:", len(results))
    print("Output folder:", args.output_dir)
    print("CSV:", output_csv)


if __name__ == "__main__":
    main()