"""Validation tool for face detection quality.

Generates a visual HTML report showing original images and detected face crops
for manual review of detection accuracy.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from tqdm import tqdm
import base64
import io

load_dotenv()

from config import StashDBConfig
from stashdb_client import StashDBClient
from embeddings import FaceEmbeddingGenerator, load_image


@dataclass
class ValidationResult:
    """Result of validating face detection on one image."""
    performer_name: str
    performer_id: str
    image_url: str
    original_image: np.ndarray
    faces: list  # List of DetectedFace objects
    error: str = None


def image_to_base64(img: np.ndarray, max_size: int = 400) -> str:
    """Convert numpy image to base64 for HTML embedding."""
    pil_img = Image.fromarray(img)

    # Resize if too large
    if max(pil_img.size) > max_size:
        ratio = max_size / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def generate_html_report(results: list[ValidationResult], output_path: Path):
    """Generate HTML report from validation results."""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Face Detection Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
        h1 { color: #fff; }
        .stats { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .performer {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        .performer.no-face { border: 2px solid #ff6b6b; }
        .performer.has-face { border: 2px solid #51cf66; }
        .original { max-width: 300px; }
        .original img { max-width: 100%; border-radius: 4px; }
        .faces { display: flex; flex-wrap: wrap; gap: 10px; }
        .face {
            background: #3a3a3a;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        .face img { max-width: 150px; max-height: 200px; border-radius: 4px; }
        .confidence {
            font-size: 14px;
            margin-top: 5px;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .confidence.high { background: #51cf66; color: #000; }
        .confidence.medium { background: #ffd43b; color: #000; }
        .confidence.low { background: #ff6b6b; color: #000; }
        .info { flex: 1; }
        .name { font-size: 18px; font-weight: bold; margin-bottom: 5px; }
        .meta { font-size: 12px; color: #888; }
        .error { color: #ff6b6b; }
    </style>
</head>
<body>
    <h1>Face Detection Validation Report</h1>
"""

    # Stats
    total = len(results)
    with_faces = sum(1 for r in results if r.faces and not r.error)
    no_faces = sum(1 for r in results if not r.faces and not r.error)
    errors = sum(1 for r in results if r.error)
    total_faces = sum(len(r.faces) for r in results if r.faces)

    html += f"""
    <div class="stats">
        <strong>Summary:</strong> {total} performers processed |
        {with_faces} with faces ({with_faces/total*100:.1f}%) |
        {no_faces} no face detected |
        {errors} errors |
        {total_faces} total faces
    </div>
"""

    # Results
    for result in results:
        has_face_class = "has-face" if result.faces else "no-face"

        html += f'<div class="performer {has_face_class}">\n'

        # Original image
        if result.original_image is not None:
            orig_b64 = image_to_base64(result.original_image)
            html += f'''
            <div class="original">
                <img src="data:image/jpeg;base64,{orig_b64}" alt="Original">
            </div>
'''

        # Info and faces
        html += f'''
            <div class="info">
                <div class="name">{result.performer_name}</div>
                <div class="meta">ID: {result.performer_id}</div>
'''

        if result.error:
            html += f'<div class="error">Error: {result.error}</div>\n'
        elif result.faces:
            html += '<div class="faces">\n'
            for i, face in enumerate(result.faces):
                face_b64 = image_to_base64(face.image, max_size=200)
                conf = face.confidence
                conf_class = "high" if conf >= 0.85 else "medium" if conf >= 0.7 else "low"
                html += f'''
                <div class="face">
                    <img src="data:image/jpeg;base64,{face_b64}" alt="Face {i+1}">
                    <div class="confidence {conf_class}">{conf:.2f}</div>
                    <div class="meta">{face.bbox['w']}x{face.bbox['h']}px</div>
                </div>
'''
            html += '</div>\n'
        else:
            html += '<div class="meta">No faces detected</div>\n'

        html += '</div></div>\n'

    html += """
</body>
</html>
"""

    output_path.write_text(html)
    print(f"Report saved to {output_path}")


def validate_detections(
    num_performers: int = 50,
    output_dir: Path = Path("./validation"),
):
    """Run face detection validation on a sample of performers."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    config = StashDBConfig.from_env()
    client = StashDBClient(config.url, config.api_key, rate_limit_delay=0.3)
    generator = FaceEmbeddingGenerator()

    print(f"Fetching {num_performers} performers from StashDB...")

    # Get performers with images
    results = []
    page = 1
    per_page = 25

    with tqdm(total=num_performers, desc="Processing") as pbar:
        while len(results) < num_performers:
            _, performers = client.query_performers(page=page, per_page=per_page)

            for performer in performers:
                if len(results) >= num_performers:
                    break

                # Skip performers without images
                if not performer.image_urls:
                    continue

                # Download first image
                url = performer.image_urls[0]
                image_data = client.download_image(url)

                if not image_data:
                    results.append(ValidationResult(
                        performer_name=performer.name,
                        performer_id=performer.id,
                        image_url=url,
                        original_image=None,
                        faces=[],
                        error="Failed to download image"
                    ))
                    pbar.update(1)
                    continue

                try:
                    image = load_image(image_data)
                    faces = generator.detect_faces(image, min_confidence=0.5)  # Lower threshold for validation

                    results.append(ValidationResult(
                        performer_name=performer.name,
                        performer_id=performer.id,
                        image_url=url,
                        original_image=image,
                        faces=faces,
                    ))
                except Exception as e:
                    results.append(ValidationResult(
                        performer_name=performer.name,
                        performer_id=performer.id,
                        image_url=url,
                        original_image=None,
                        faces=[],
                        error=str(e)
                    ))

                pbar.update(1)

            page += 1

            # Safety limit
            if page > 100:
                break

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"validation_report_{timestamp}.html"
    generate_html_report(results, report_path)

    # Print summary
    print(f"\n=== Validation Summary ===")
    print(f"Performers processed: {len(results)}")
    print(f"With faces: {sum(1 for r in results if r.faces)}")
    print(f"No faces: {sum(1 for r in results if not r.faces and not r.error)}")
    print(f"Errors: {sum(1 for r in results if r.error)}")
    print(f"\nReport: {report_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate face detection quality")
    parser.add_argument("-n", "--num-performers", type=int, default=50,
                        help="Number of performers to validate")
    parser.add_argument("-o", "--output", type=str, default="./validation",
                        help="Output directory for validation report")
    args = parser.parse_args()

    validate_detections(
        num_performers=args.num_performers,
        output_dir=Path(args.output),
    )
