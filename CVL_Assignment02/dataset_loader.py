import os
import csv
from PIL import Image
import numpy as np

# ── 1. Parse anotasi dari Train.csv (Format Kaggle) ──────────────────────────
def load_class_annotations(csv_path, target_class_id):
    """
    Membaca Train.csv dan mengambil data hanya untuk kelas tertentu.
    Kaggle GTSRB punya kolom: Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
    """
    samples = []
    
    with open(csv_path, newline='') as f:
        # File Kaggle menggunakan pemisah koma
        reader = csv.DictReader(f, delimiter=',')
        
        for row in reader:
            # Hanya ambil baris yang ClassId-nya sesuai target (misal: 14)
            if int(row['ClassId']) == target_class_id:
                # Kolom 'Path' isinya sudah seperti 'Train/14/00000_00000.png'
                # Kita tinggal gabungkan dengan root folder './gtsrb'
                img_path = os.path.join('./gtsrb', row['Path'])
                
                samples.append({
                    'filepath' : img_path,
                    'width'    : int(row['Width']),
                    'height'   : int(row['Height']),
                    'x1'       : int(row['Roi.X1']),
                    'y1'       : int(row['Roi.Y1']),
                    'x2'       : int(row['Roi.X2']),
                    'y2'       : int(row['Roi.Y2']),
                    'class_id' : int(row['ClassId']),
                })
    return samples

# ── 2. Load image + konversi ke numpy array ────────────────────────
def load_image(filepath):
    img = Image.open(filepath).convert('RGB')
    return np.array(img)          # shape: (H, W, 3)

# ── 3. Convert ke format {x, y, w, h} untuk sliding window ───────────────────
def gt_box_from_annotation(ann):
    return {
        'x': ann['x1'],
        'y': ann['y1'],
        'w': ann['x2'] - ann['x1'],
        'h': ann['y2'] - ann['y1'],
    }

# ── 4. Jalankan Testing ────────────────────────────────────────────────
if __name__ == "__main__":
    # Path langsung menuju file Train.csv
    CSV_PATH = './gtsrb/Train.csv'

    print("Mencoba load anotasi kelas 14 (Stop Sign) dari Train.csv...")
    stop_sign_samples = load_class_annotations(CSV_PATH, target_class_id=14)
    
    if len(stop_sign_samples) > 0:
        sample = stop_sign_samples[0]
        image_array = load_image(sample['filepath'])
        gt_box = gt_box_from_annotation(sample)

        print("\nBerhasil!")
        print(f"Jumlah sampel kelas 14: {len(stop_sign_samples)} gambar")
        print(f"Path gambar contoh  : {sample['filepath']}")
        print(f"Image shape         : {image_array.shape}")
        print(f"GT box              : {gt_box}")
    else:
        print("Gagal menemukan sampel. Coba cek path CSV-nya.")