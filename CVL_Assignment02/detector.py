import os
import cv2
import numpy as np
import dataset_loader as dl
import evaluation as ev

def get_red_mask(image):
    # 1. Blur untuk menghaluskan noise piksel
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    
    # 2. Range merah yang LEBIH LONGGAR (Lower Saturation & Value)
    lower_red1, upper_red1 = np.array([0, 50, 30]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 50, 30]), np.array([180, 255, 255])
    
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                          cv2.inRange(hsv, lower_red2, upper_red2))
    
    # 3. Dilation sedikit lebih kuat (2 iterasi) agar bentuk tersambung
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

def get_templates_per_class(samples, num=5): # Tambah jadi 5 template biar lebih variatif
    templates = []
    for i in range(min(num, len(samples))):
        img = dl.load_image(samples[i]['filepath'])
        gt = dl.gt_box_from_annotation(samples[i])
        crop = img[gt['y']:gt['y']+gt['h'], gt['x']:gt['x']+gt['w']]
        templates.append(get_red_mask(crop))
    return templates

def detect_multi_template(image, templates, threshold=0.3):
    target_mask = get_red_mask(image)
    best_det = None
    # 4. Skala lebih rapat (20 langkah) untuk menangkap ukuran yang presisi
    scales = np.linspace(0.4, 1.6, 20)
    for template_mask in templates:
        t_h, t_w = template_mask.shape
        for scale in scales:
            rw, rh = int(target_mask.shape[1] * scale), int(target_mask.shape[0] * scale)
            if rw < t_w or rh < t_h: continue
            
            resized_target = cv2.resize(target_mask, (rw, rh))
            res = cv2.matchTemplate(resized_target, template_mask, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= threshold:
                curr_w, curr_h = int(t_w / scale), int(t_h / scale)
                x_start, y_start = int(max_loc[0] / scale), int(max_loc[1] / scale)
                
                # --- FILTER GEOMETRI YANG LEBIH TOLERAN ---
                # Aspect ratio lingkaran bisa jadi agak lonjong karena perspektif kamera
                aspect_ratio = max(curr_w, curr_h) / min(curr_w, curr_h)
                if aspect_ratio > 1.6: continue
                
                # Density diturunkan sedikit lagi (10%) karena banyak rambu yang blur
                crop_area = target_mask[y_start:y_start+curr_h, x_start:x_start+curr_w]
                if crop_area.size > 0:
                    density = np.sum(crop_area > 0) / crop_area.size
                    if density < 0.10: continue

                if best_det is None or max_val > best_det['score']:
                    best_det = {'x': x_start, 'y': y_start, 'w': curr_w, 'h': curr_h, 'score': max_val}
                    
    return [best_det] if best_det else []

def visualize_and_save(image_rgb, gt_box, det_box, save_path):
    """Menggambar kotak Ground Truth (Hijau) dan Deteksi (Biru) lalu menyimpannya."""
    # OpenCV menyimpan gambar dalam format BGR, jadi kita convert dulu dari RGB
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Gambar Ground Truth (Kotak Hijau)
    cv2.rectangle(img_bgr, (gt_box['x'], gt_box['y']), 
                  (gt_box['x'] + gt_box['w'], gt_box['y'] + gt_box['h']), 
                  (0, 255, 0), 2)
    cv2.putText(img_bgr, "Asli", (gt_box['x'], gt_box['y'] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 2. Gambar Deteksi (Kotak Biru) - Jika terdeteksi
    if det_box is not None:
        cv2.rectangle(img_bgr, (det_box['x'], det_box['y']), 
                      (det_box['x'] + det_box['w'], det_box['y'] + det_box['h']), 
                      (255, 0, 0), 2)
        cv2.putText(img_bgr, f"Det: {det_box['score']:.2f}", (det_box['x'], max(15, det_box['y'] - 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
    cv2.imwrite(save_path, img_bgr)

if __name__ == "__main__":
    CSV_PATH = './gtsrb/Train.csv'
    OUTPUT_DIR = './output_visuals'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    classes_to_test = {
        1:  {"name": "Speed 30", "thresh": 0.35},
        14: {"name": "Stop",     "thresh": 0.35},
        17: {"name": "No Entry", "thresh": 0.25} # Turunkan biar lebih sensitif
    }

    final_report = []

    for cid, info in classes_to_test.items():
        print(f"Mengoptimalkan Kelas {info['name']}...")
        samples = dl.load_class_annotations(CSV_PATH, target_class_id=cid)
        
        # Pakai 5 template agar mencakup berbagai kondisi cahaya
        templates = get_templates_per_class(samples, num=5)
        test_data = samples[15:65] # Tambah jumlah data tes jadi 50 

        metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'iou': []}
        saved_count = 0

        for i, s in enumerate(test_data):
            img = dl.load_image(s['filepath'])
            gt = dl.gt_box_from_annotation(s)
            dets = detect_multi_template(img, templates, threshold=info['thresh'])
            res = ev.evaluate_detections(dets, [gt])

            if saved_count < 5:
                # Ambil deteksi terbaik (jika ada) untuk digambar
                best_det = dets[0] if len(dets) > 0 else None
                
                # Buat nama file yang unik: misal "Stop_sample_0.jpg"
                filename = os.path.join(OUTPUT_DIR, f"{info['name']}_sample_{i}.jpg")
                visualize_and_save(img, gt, best_det, filename)
                saved_count += 1

            metrics['TP'] += res['TP']
            metrics['FP'] += res['FP']
            metrics['FN'] += res['FN']
            if res['TP'] > 0: metrics['iou'].append(res['avg_iou'])

        prec = metrics['TP'] / (metrics['TP'] + metrics['FP'] + 1e-9)
        rec = metrics['TP'] / (metrics['TP'] + metrics['FN'] + 1e-9)

        final_report.append({
            'name': info['name'], 'total': len(test_data), 'TP': metrics['TP'],
            'FP': metrics['FP'], 'FN': metrics['FN'], 'prec': prec, 'rec': rec,
            'iou': np.mean(metrics['iou']) if metrics['iou'] else 0
        })

    print("\n" + "="*60)
    header = f"{'Class Name':<12} | {'Total':<5} | {'TP':<3} | {'FP':<3} | {'FN':<3} | {'Prec':<5} | {'Rec':<5} | {'IoU':<5}"
    print(header)
    print("-" * 60)
    for r in final_report:
        print(f"{r['name']:<12} | {r['total']:<5} | {r['TP']:<3} | {r['FP']:<3} | "
              f"{r['FN']:<3} | {r['prec']:.2f}  | {r['rec']:.2f}  | {r['iou']:.2f}")
    print("="*60)