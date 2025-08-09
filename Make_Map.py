from PIL import Image
import csv
import json

import os

BASE_DIR = "/Users/mmv5513/Documents/Fun Projects/EU4 Viewer"
BMP_PATH = os.path.join(BASE_DIR, "provinces.bmp")
CSV_PATH = os.path.join(BASE_DIR, "definition.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "province_pixel_map.json")

# Step 1 — Load the province color mapping
color_to_province = {}
with open(CSV_PATH, newline='', encoding='latin-1') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if not row or not row[0].isdigit():
            continue
        prov_id = int(row[0])
        r, g, b = int(row[1]), int(row[2]), int(row[3])
        color_to_province[(r, g, b)] = prov_id

# Step 2 — Open the BMP map
img = Image.open(BMP_PATH).convert("RGB")
width, height = img.size
pixels = img.load()

province_pixels = {}

# Step 3 — Map each pixel to its province ID
for y in range(height):
    for x in range(width):
        rgb = pixels[x, y]
        prov_id = color_to_province.get(rgb)
        if prov_id:
            province_pixels.setdefault(prov_id, []).append((x, y))

# Step 4 — Save as JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(province_pixels, f)

print(f"✅ Saved province pixel map to {OUTPUT_JSON}")
