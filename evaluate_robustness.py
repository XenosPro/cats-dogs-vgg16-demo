"""Stage 2: offline robustness audit for the Cats vs Dogs VGG16 demo.

Example:
    python evaluate_robustness.py --model models/cats_dogs_vgg16.keras --images robustness_images

Expected folders: robustness_images/{cat,dog,lion,wolf,other}/
This script does NOT train a model or implement unknown-image detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

KNOWN_CLASSES = {"cat", "dog"}
UNKNOWN_CLASSES = {"lion", "wolf", "other"}
ALL_CATEGORIES = KNOWN_CLASSES | UNKNOWN_CLASSES
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_IMAGE_EDGE = 1600
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def prepare_image(path: Path) -> Image.Image:
    """Match the Streamlit app's decoding, EXIF fix and thumbnail behavior."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("file exceeds 10 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as check:
                if check.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("unsupported format")
                if check.width * check.height > MAX_IMAGE_PIXELS:
                    raise ValueError("image exceeds 32 megapixels")
                check.verify()
            with Image.open(path) as source:
                source.draft("RGB", (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
                corrected = ImageOps.exif_transpose(source)
                corrected.thumbnail(
                    (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS
                )
                rgb = corrected.convert("RGB")
                rgb.load()
                return rgb
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError(f"cannot safely decode: {exc}") from exc


def predict_dog_score(model, image: Image.Image, tf) -> float:
    """Same TensorFlow resize + VGG16 preprocess_input as deployed app.py."""
    pixels = tf.convert_to_tensor(np.asarray(image, dtype=np.uint8))
    resized = tf.image.resize(pixels, (224, 224), method="bilinear")
    preprocessed = tf.keras.applications.vgg16.preprocess_input(resized)
    batch = tf.expand_dims(preprocessed, axis=0)
    score = float(np.asarray(model(batch, training=False)).reshape(-1)[0])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("model returned an invalid probability")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("models/cats_dogs_vgg16.keras"))
    parser.add_argument("--images", type=Path,
                        default=Path("robustness_images"))
    parser.add_argument("--output", type=Path,
                        default=Path("robustness_report"))
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    if not args.images.is_dir():
        parser.error(f"Image directory not found: {args.images}")

    import tensorflow as tf  # Imported only after argument validation

    model = tf.keras.models.load_model(str(args.model), compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"Unexpected model input: {model.input_shape}")

    paths = sorted(
        path for category in sorted(ALL_CATEGORIES)
        for path in (args.images / category).rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        parser.error("No supported image files found. Populate the category folders.")

    rows = []
    for path in paths:
        category = path.relative_to(args.images).parts[0]
        try:
            image = prepare_image(path)
            dog_score = predict_dog_score(model, image, tf)
            cat_score = 1.0 - dog_score
            prediction = "dog" if dog_score >= 0.5 else "cat"
            selected_score = max(cat_score, dog_score)
            rows.append({
                "file": str(path), "true_category": category,
                "in_domain": category in KNOWN_CLASSES,
                "prediction": prediction,
                "correct_if_cat_or_dog": (
                    prediction == category if category in KNOWN_CLASSES else ""
                ),
                "cat_score": round(cat_score, 8),
                "dog_score": round(dog_score, 8),
                "selected_score": round(selected_score, 8),
                "error": "",
            })
        except (ValueError, OSError) as exc:
            rows.append({
                "file": str(path), "true_category": category,
                "in_domain": category in KNOWN_CLASSES,
                "prediction": "", "correct_if_cat_or_dog": "",
                "cat_score": "", "dog_score": "", "selected_score": "",
                "error": str(exc),
            })
        print(f"{len(rows):3d}/{len(paths):3d}  {category:5s}  {path.name}")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    usable = [row for row in rows if not row["error"]]
    known = [row for row in usable if row["in_domain"]]
    unknown = [row for row in usable if not row["in_domain"]]
    summary = {
        "tested_images": len(rows),
        "successfully_processed": len(usable),
        "failed_images": len(rows) - len(usable),
        "known_cat_dog_n": len(known),
        "known_cat_dog_accuracy": (
            round(sum(row["correct_if_cat_or_dog"] for row in known) / len(known), 4)
            if known else None
        ),
        "unsupported_n": len(unknown),
        # Descriptive statistics only: this is NOT an OOD detection result.
        "unsupported_fraction_score_at_least_90pct": (
            round(sum(row["selected_score"] >= 0.9 for row in unknown) / len(unknown), 4)
            if unknown else None
        ),
        "unsupported_fraction_score_at_least_99pct": (
            round(sum(row["selected_score"] >= 0.99 for row in unknown) / len(unknown), 4)
            if unknown else None
        ),
        "per_category": {},
    }
    for category in sorted(ALL_CATEGORIES):
        items = [row for row in usable if row["true_category"] == category]
        summary["per_category"][category] = {
            "n": len(items),
            "cat_predictions": sum(row["prediction"] == "cat" for row in items),
            "dog_predictions": sum(row["prediction"] == "dog" for row in items),
            "mean_selected_score": (
                round(sum(row["selected_score"] for row in items) / len(items), 4)
                if items else None
            ),
        }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nResults:\n" + json.dumps(summary, indent=2))
    print(f"\nSaved: {csv_path} and {summary_path}")
    print("Note: unsupported-image high scores do not measure detection accuracy.")


if __name__ == "__main__":
    main()
