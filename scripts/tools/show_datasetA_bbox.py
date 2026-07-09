from __future__ import annotations

import argparse
import html
import sys
import webbrowser
from pathlib import Path
from xml.etree import ElementTree as ET


def find_label_path(image_path: Path) -> Path:
    """Infer DatasetA-Real label path from a defective image path."""
    parts_lower = [part.lower() for part in image_path.parts]

    if "image" in parts_lower:
        image_index = parts_lower.index("image")
        candidate_parts = list(image_path.parts)
        candidate_parts[image_index] = "label"
        return Path(*candidate_parts).with_suffix(".xml")

    # Fallback for direct DatasetA-Real/nondefective paths. These usually do not
    # have labels, but this gives a helpful error path.
    return image_path.with_suffix(".xml")


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[dict[str, int | str]]]:
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> tag found in {xml_path}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    boxes: list[dict[str, int | str]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "DEFECT")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))

        boxes.append(
            {
                "name": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return width, height, boxes


def build_html(image_path: Path, xml_path: Path, width: int, height: int, boxes: list[dict[str, int | str]]) -> str:
    box_html = []
    for box in boxes:
        xmin = int(box["xmin"])
        ymin = int(box["ymin"])
        xmax = int(box["xmax"])
        ymax = int(box["ymax"])
        label = html.escape(str(box["name"]))

        box_html.append(
            f"""
            <div class="box" style="left:{xmin}px; top:{ymin}px; width:{xmax - xmin}px; height:{ymax - ymin}px;">
                <span>{label}</span>
            </div>
            """
        )

    image_uri = image_path.resolve().as_uri()
    image_name = html.escape(image_path.name)
    xml_name = html.escape(str(xml_path))

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>DatasetA Bounding Box Preview</title>
    <style>
        body {{
            margin: 24px;
            font-family: Arial, sans-serif;
            background: #f3f4f6;
            color: #111827;
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: 20px;
        }}
        p {{
            margin: 0 0 16px;
            color: #4b5563;
            font-size: 14px;
        }}
        .viewer {{
            position: relative;
            width: {width}px;
            height: {height}px;
            background: #111827;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        }}
        .viewer img {{
            display: block;
            width: {width}px;
            height: {height}px;
        }}
        .box {{
            position: absolute;
            border: 3px solid #ef4444;
            box-sizing: border-box;
        }}
        .box span {{
            position: absolute;
            left: -3px;
            top: -26px;
            padding: 3px 7px;
            background: #ef4444;
            color: white;
            font-size: 13px;
            font-weight: 700;
            white-space: nowrap;
        }}
    </style>
</head>
<body>
    <h1>{image_name}</h1>
    <p>Label: {xml_name}</p>
    <div class="viewer">
        <img src="{image_uri}" alt="{image_name}">
        {"".join(box_html)}
    </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show DatasetA-Real defect bounding boxes from Pascal VOC XML labels."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a DatasetA-Real defective image. If omitted, the script will ask for it.",
    )
    parser.add_argument("--xml", help="Optional XML label path. By default it is inferred from the image path.")
    parser.add_argument(
        "--output",
        default="bbox_preview.html",
        help="HTML preview file to create. Default: bbox_preview.html",
    )
    parser.add_argument("--no-open", action="store_true", help="Create the HTML file without opening it.")
    args = parser.parse_args()

    image_input = args.image or input("Paste image path: ").strip().strip('"')
    image_path = Path(image_input).expanduser()
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    xml_path = Path(args.xml).expanduser() if args.xml else find_label_path(image_path)
    if not xml_path.exists():
        print(f"XML label not found: {xml_path}", file=sys.stderr)
        print("Tip: use an image from DatasetA-Real\\defective\\image or pass --xml manually.", file=sys.stderr)
        return 1

    width, height, boxes = parse_voc_xml(xml_path)
    if not boxes:
        print(f"No object bounding boxes found in: {xml_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output).resolve()
    output_path.write_text(build_html(image_path, xml_path, width, height, boxes), encoding="utf-8")

    print(f"Created preview: {output_path}")
    print(f"Found {len(boxes)} bounding box(es):")
    for box in boxes:
        print(f"  {box['name']}: ({box['xmin']}, {box['ymin']}) to ({box['xmax']}, {box['ymax']})")

    if not args.no_open:
        webbrowser.open(output_path.as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
