import json
import os
from base64 import b64decode

notebooks = [
    'notebooks/02_pluto_dataset_analysis.ipynb',
    'notebooks/03_mta_ridership_analysis.ipynb',
    'notebooks/05_acris_transactions_analysis.ipynb'
]

out_dir = 'figures'
os.makedirs(out_dir, exist_ok=True)

saved = []
for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"Notebook not found: {nb_path}")
        continue
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    base = os.path.splitext(os.path.basename(nb_path))[0]
    for i, cell in enumerate(nb.get('cells', [])):
        outputs = cell.get('outputs', [])
        for j, out in enumerate(outputs):
            data = out.get('data', {})
            if 'image/png' in data:
                img_b64 = data['image/png']
                # image/png may be a list of lines
                if isinstance(img_b64, list):
                    img_b64 = ''.join(img_b64)
                try:
                    img_bytes = b64decode(img_b64)
                    filename = f"{base}_cell{i+1}_out{j+1}.png"
                    path = os.path.join(out_dir, filename)
                    with open(path, 'wb') as imgf:
                        imgf.write(img_bytes)
                    saved.append(path)
                    print(f"Saved {path}")
                except Exception as e:
                    print(f"Failed to save image from {nb_path} cell {i+1} output {j+1}: {e}")

if saved:
    print('\nSummary: saved images:')
    for s in saved:
        print(' -', s)
else:
    print('No images found in notebooks outputs.')
