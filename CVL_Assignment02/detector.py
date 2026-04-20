import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import dataset_loader as dl
import evaluation as ev

# ═══════════════════════════════════════════════════════════════
#  IMPLEMENTASI MATEMATIS (pengganti cv2) — NumPy murni
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────
# 1. GAUSSIAN BLUR
#    Kernel Gaussian 1D: G(x) = exp(-x² / 2σ²)
#    Kernel 2D dibuat dari outer product dua kernel 1D (separable filter)
#    lalu dikonvolusi manual dengan sliding window (stride-tricks)
# ───────────────────────────────────────────────
def make_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Buat kernel Gaussian 2D ternormalisasi."""
    ax = np.arange(ksize) - (ksize - 1) / 2.0          # pusat = 0
    g1d = np.exp(-ax**2 / (2 * sigma**2))
    kernel = np.outer(g1d, g1d)                          # G(x,y) = G(x)*G(y)
    return kernel / kernel.sum()                         # normalisasi agar jumlah = 1


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Konvolusi 2D manual dengan zero-padding.
    image : (H, W)  float32
    kernel: (kH, kW) float32
    """
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2

    # Zero-padding
    padded = np.pad(image, ((pH, pH), (pW, pW)), mode='constant', constant_values=0)

    # Sliding window menggunakan stride tricks (efisien, tanpa Python loop per piksel)
    H, W = image.shape
    shape   = (H, W, kH, kW)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Hadamard product + sum → satu nilai per piksel
    return np.einsum('hwkl,kl->hw', windows, kernel).astype(np.float32)


def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Gaussian blur per-channel.
    σ dihitung dengan rumus OpenCV: σ = 0.3·((ksize-1)·0.5 - 1) + 0.8
    """
    sigma  = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    kernel = make_gaussian_kernel(ksize, sigma)
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        result[:, :, c] = convolve2d(image[:, :, c].astype(np.float32), kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────────
# 2. KONVERSI RGB → HSV
#    Rumus standar ITU / Wikipedia:
#      V = max(R,G,B)
#      S = (V - min) / V   jika V≠0, else 0
#      H = tergantung channel mana yang max
#    Output: H∈[0,180], S∈[0,255], V∈[0,255]  (skala OpenCV)
# ───────────────────────────────────────────────
def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Konversi gambar RGB uint8 → HSV float32.
    H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255]  (sama dengan OpenCV).
    """
    img = image.astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    Vmax = np.max(img, axis=2)
    Vmin = np.min(img, axis=2)
    diff = Vmax - Vmin                                   # chroma

    # Value
    V = Vmax

    # Saturation: S = diff/V jika V>0, else 0
    S = np.where(Vmax > 0, diff / (Vmax + 1e-9), 0.0)

    # Hue (dalam derajat 0–360, lalu dibagi 2 → 0–180 seperti OpenCV)
    H = np.zeros_like(V)

    # Kondisi: max = R
    mask_R = (Vmax == R) & (diff > 0)
    H[mask_R] = (60.0 * ((G[mask_R] - B[mask_R]) / diff[mask_R])) % 360

    # Kondisi: max = G
    mask_G = (Vmax == G) & (diff > 0)
    H[mask_G] = 60.0 * ((B[mask_G] - R[mask_G]) / diff[mask_G]) + 120

    # Kondisi: max = B
    mask_B = (Vmax == B) & (diff > 0)
    H[mask_B] = 60.0 * ((R[mask_B] - G[mask_B]) / diff[mask_B]) + 240

    # Skala ke OpenCV: H/2, S*255, V*255
    hsv = np.stack([H / 2.0, S * 255.0, V * 255.0], axis=2).astype(np.float32)
    return hsv


# ───────────────────────────────────────────────
# 3. inRange — threshold HSV
#    Cek apakah setiap piksel masuk rentang [lower, upper]
#    untuk ketiga channel (H, S, V) sekaligus
# ───────────────────────────────────────────────
def in_range(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Hasilkan mask uint8 (0 / 255).
    lower, upper: array [H, S, V] dengan skala OpenCV.
    """
    mask = np.all((hsv >= lower) & (hsv <= upper), axis=2)
    return (mask * 255).astype(np.uint8)


# ───────────────────────────────────────────────
# 4. MORPHOLOGICAL DILATION
#    Dilation: output(x,y) = max nilai piksel dalam
#    neighborhood yang didefinisikan oleh structuring element (kernel kotak).
#    Implementasi dengan stride tricks (sama seperti konvolusi, tapi pakai max).
# ───────────────────────────────────────────────
def dilate(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Morfologi dilation dengan structuring element kotak.
    mask: (H, W) uint8 biner (0/255)
    """
    current = mask.copy().astype(np.float32)
    k = kernel_size
    p = k // 2

    for _ in range(iterations):
        padded  = np.pad(current, p, mode='constant', constant_values=0)
        H, W    = current.shape
        shape   = (H, W, k, k)
        strides = (padded.strides[0], padded.strides[1],
                   padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        current = windows.max(axis=(2, 3))               # max dalam neighborhood

    return (current > 0).astype(np.uint8) * 255


# ───────────────────────────────────────────────
# 5. RESIZE BILINEAR
#    Untuk setiap piksel output (x', y'), hitung posisi
#    float di gambar input: x = x' * (W_src/W_dst), idem y.
#    Interpolasi bilinear dari 4 tetangga terdekat:
#      f(x,y) = (1-dx)(1-dy)·f00 + dx(1-dy)·f10
#             + (1-dx)dy·f01  + dx·dy·f11
# ───────────────────────────────────────────────
def resize_bilinear(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Resize gambar (2D mask atau 3D RGB) dengan interpolasi bilinear murni NumPy.
    """
    src_h, src_w = image.shape[:2]

    # Grid koordinat output → koordinat float di ruang input
    # Gunakan transformasi: x_src = (x_dst + 0.5) * (src_w / new_w) - 0.5
    # agar area-align seperti OpenCV INTER_LINEAR
    x_dst = np.arange(new_w, dtype=np.float32)
    y_dst = np.arange(new_h, dtype=np.float32)
    x_src = (x_dst + 0.5) * (src_w / new_w) - 0.5
    y_src = (y_dst + 0.5) * (src_h / new_h) - 0.5

    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp ke batas gambar
    x0c = np.clip(x0, 0, src_w - 1)
    x1c = np.clip(x1, 0, src_w - 1)
    y0c = np.clip(y0, 0, src_h - 1)
    y1c = np.clip(y1, 0, src_h - 1)

    # Bobot interpolasi
    dx = (x_src - x0).astype(np.float32)    # fraksi horizontal
    dy = (y_src - y0).astype(np.float32)    # fraksi vertikal

    # Broadcast: dx → (1, new_w),  dy → (new_h, 1)
    dx = dx[np.newaxis, :]
    dy = dy[:, np.newaxis]

    def interp_channel(ch):
        f00 = ch[np.ix_(y0c, x0c)].astype(np.float32)
        f10 = ch[np.ix_(y0c, x1c)].astype(np.float32)
        f01 = ch[np.ix_(y1c, x0c)].astype(np.float32)
        f11 = ch[np.ix_(y1c, x1c)].astype(np.float32)
        return ((1 - dy) * (1 - dx) * f00 +
                (1 - dy) * dx       * f10 +
                dy       * (1 - dx) * f01 +
                dy       * dx       * f11)

    if image.ndim == 2:
        out = interp_channel(image)
        # Untuk mask biner: threshold 127 agar tetap biner
        return (out > 127).astype(np.uint8) * 255
    else:
        channels = [np.clip(interp_channel(image[:, :, c]), 0, 255).astype(np.uint8)
                    for c in range(image.shape[2])]
        return np.stack(channels, axis=2)


# ───────────────────────────────────────────────
# 6. TEMPLATE MATCHING — TM_CCOEFF_NORMED
#    Rumus:
#      T'(x,y)  = T(x,y) - mean(T)
#      I'(x,y)  = I(x,y) - mean(I dalam window)
#      R(u,v)   = Σ T'·I' / sqrt(Σ T'² · Σ I'²)
#    Implementasi dengan FFT-based cross-correlation (O(N log N)):
#      cross_corr = IFFT( FFT(I) · conj(FFT(T_padded)) )
#    Mean window dihitung dengan summed-area table (integral image).
# ───────────────────────────────────────────────
def integral_image(arr: np.ndarray) -> np.ndarray:
    """Summed-area table untuk query sum O(1)."""
    return arr.cumsum(axis=0).cumsum(axis=1)


def window_sum(integral: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Hitung sum setiap window (h x w) dari integral image.
    Output shape: (H-h+1, W-w+1)
    """
    r = integral
    # Gunakan inklusi-eksklusi pada integral image
    out = (r[h:,  w:]
         - r[h:,  :r.shape[1]-w]
         - r[:r.shape[0]-h, w:]
         + r[:r.shape[0]-h, :r.shape[1]-w])
    return out


def match_template_normed(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Template matching TM_CCOEFF_NORMED menggunakan FFT cross-correlation.
    image, template: 2D float32, nilai 0–255.
    Return: result map shape (H-th+1, W-tw+1), nilai ∈ [-1, 1].
    """
    iH, iW = image.shape
    tH, tW = template.shape

    if iH < tH or iW < tW:
        return np.array([[-1.0]])

    # Zero-mean template
    T_mean = template.mean()
    T_norm = template - T_mean
    sum_T2 = np.sum(T_norm ** 2)

    if sum_T2 == 0:
        return np.zeros((iH - tH + 1, iW - tW + 1), dtype=np.float32)

    # FFT-based cross-correlation antara image dan template
    # Pad ke ukuran yang sama untuk FFT circular
    fft_h = iH
    fft_w = iW

    # Pad template ke ukuran gambar, letakkan di pojok kiri atas
    T_padded         = np.zeros((fft_h, fft_w), dtype=np.float32)
    T_padded[:tH, :tW] = T_norm

    F_image    = np.fft.rfft2(image.astype(np.float32), s=(fft_h, fft_w))
    F_template = np.fft.rfft2(T_padded,                 s=(fft_h, fft_w))

    # Cross-correlation: I ⊗ T = IFFT(FFT(I) · conj(FFT(T)))
    cross = np.fft.irfft2(F_image * np.conj(F_template), s=(fft_h, fft_w))

    # Kita butuh nilai pada posisi (v, u) = Σ_{x,y} I(u+x, v+y)·T'(x,y)
    # Dari FFT circular, posisi valid ada di cross[0:iH-tH+1, 0:iW-tW+1]
    cross_valid = cross[:iH - tH + 1, :iW - tW + 1]

    # Mean window menggunakan integral image
    integral = np.pad(integral_image(image.astype(np.float64)),
                      ((1, 0), (1, 0)), mode='constant')
    win_sum  = window_sum(integral, tH, tW).astype(np.float32)
    win_mean = win_sum / (tH * tW)

    # Koreksi cross-correlation: kurangi efek mean image
    # Σ T'·I' = cross - mean_I·Σ T'  (karena Σ T' = 0 setelah zero-mean)
    # Σ T' = 0 → suku koreksi = 0, jadi cross_valid sudah benar
    numerator = cross_valid

    # Σ I'² dalam window = Σ I² - n·mean_I²
    I2_integral = np.pad(integral_image((image.astype(np.float64))**2),
                         ((1, 0), (1, 0)), mode='constant')
    win_sum_I2  = window_sum(I2_integral, tH, tW).astype(np.float32)
    n           = tH * tW
    sum_I2      = win_sum_I2 - n * (win_mean ** 2)
    sum_I2      = np.maximum(sum_I2, 0)

    denominator = np.sqrt(sum_T2 * sum_I2)
    result      = np.where(denominator > 1e-9, numerator / denominator, 0.0)
    return result.astype(np.float32)


def find_max(result: np.ndarray):
    """
    Pengganti cv2.minMaxLoc.
    Return: (max_val, (x, y)) — x=kolom, y=baris, seperti OpenCV.
    """
    idx     = np.unravel_index(result.argmax(), result.shape)  # (row, col)
    max_val = float(result[idx])
    max_loc = (int(idx[1]), int(idx[0]))                        # (x, y)
    return max_val, max_loc


# ───────────────────────────────────────────────
# 7. GAMBAR BOUNDING BOX & SIMPAN GAMBAR
#    Menggunakan Pillow hanya untuk I/O gambar dan
#    menggambar kotak (tidak ada komputasi CV di sini)
# ───────────────────────────────────────────────
def draw_and_save(image_rgb: np.ndarray,
                  gt_box: dict,
                  det_box,
                  save_path: str) -> None:
    """Gambar kotak GT (hijau) dan deteksi (biru), simpan ke file."""
    img_pil = Image.fromarray(image_rgb.astype(np.uint8), mode='RGB')
    draw    = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Ground Truth → Hijau
    gx, gy, gw, gh = gt_box['x'], gt_box['y'], gt_box['w'], gt_box['h']
    draw.rectangle([(gx, gy), (gx + gw, gy + gh)], outline=(0, 200, 0), width=2)
    draw.text((gx, max(0, gy - 16)), "Asli", fill=(0, 200, 0), font=font)

    # Deteksi → Biru
    if det_box is not None:
        dx, dy, dw, dh = det_box['x'], det_box['y'], det_box['w'], det_box['h']
        draw.rectangle([(dx, dy), (dx + dw, dy + dh)], outline=(0, 80, 220), width=2)
        draw.text((dx, max(0, dy - 20)),
                  f"Det: {det_box['score']:.2f}", fill=(0, 80, 220), font=font)

    img_pil.save(save_path)


# ═══════════════════════════════════════════════════════════════
#  PIPELINE DETEKSI (identik dengan versi cv2)
# ═══════════════════════════════════════════════════════════════

def get_red_mask(image: np.ndarray) -> np.ndarray:
    """Hasilkan mask merah dari gambar RGB menggunakan HSV threshold."""
    blurred = gaussian_blur(image, ksize=5)
    hsv     = rgb_to_hsv(blurred)

    lower_red1, upper_red1 = np.array([0,   50,  30]), np.array([10,  255, 255])
    lower_red2, upper_red2 = np.array([160, 50,  30]), np.array([180, 255, 255])

    mask = np.where(
        (in_range(hsv, lower_red1, upper_red1) > 0) |
        (in_range(hsv, lower_red2, upper_red2) > 0),
        np.uint8(255), np.uint8(0)
    )
    return dilate(mask, kernel_size=3, iterations=2)


def get_templates_per_class(samples: list, num: int = 5) -> list:
    templates = []
    for i in range(min(num, len(samples))):
        img  = dl.load_image(samples[i]['filepath'])
        gt   = dl.gt_box_from_annotation(samples[i])
        crop = img[gt['y']:gt['y'] + gt['h'], gt['x']:gt['x'] + gt['w']]
        templates.append(get_red_mask(crop))
    return templates


def detect_multi_template(image: np.ndarray,
                           templates: list,
                           threshold: float = 0.3) -> list:
    target_mask = get_red_mask(image)
    best_det    = None
    scales      = np.linspace(0.4, 1.6, 20)

    for template_mask in templates:
        t_h, t_w = template_mask.shape

        for scale in scales:
            rw = int(target_mask.shape[1] * scale)
            rh = int(target_mask.shape[0] * scale)
            if rw < t_w or rh < t_h:
                continue

            resized_target = resize_bilinear(target_mask, rw, rh)
            result         = match_template_normed(
                                resized_target.astype(np.float32),
                                template_mask.astype(np.float32))
            max_val, max_loc = find_max(result)

            if max_val >= threshold:
                curr_w  = int(t_w / scale)
                curr_h  = int(t_h / scale)
                x_start = int(max_loc[0] / scale)
                y_start = int(max_loc[1] / scale)

                aspect_ratio = max(curr_w, curr_h) / max(min(curr_w, curr_h), 1)
                if aspect_ratio > 1.6:
                    continue

                crop_area = target_mask[y_start:y_start + curr_h,
                                        x_start:x_start + curr_w]
                if crop_area.size > 0:
                    density = np.sum(crop_area > 0) / crop_area.size
                    if density < 0.10:
                        continue

                if best_det is None or max_val > best_det['score']:
                    best_det = {'x': x_start, 'y': y_start,
                                'w': curr_w,  'h': curr_h,
                                'score': max_val}

    return [best_det] if best_det else []


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    CSV_PATH   = './gtsrb/Train.csv'
    OUTPUT_DIR = './output_visuals'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    classes_to_test = {
        1:  {"name": "Speed 30", "thresh": 0.35},
        14: {"name": "Stop",     "thresh": 0.35},
        17: {"name": "No Entry", "thresh": 0.25},
    }

    final_report = []

    for cid, info in classes_to_test.items():
        print(f"Mengoptimalkan Kelas {info['name']}...")
        samples     = dl.load_class_annotations(CSV_PATH, target_class_id=cid)
        templates   = get_templates_per_class(samples, num=5)
        test_data   = samples[15:65]
        metrics     = {'TP': 0, 'FP': 0, 'FN': 0, 'iou': []}
        saved_count = 0

        for i, s in enumerate(test_data):
            img  = dl.load_image(s['filepath'])
            gt   = dl.gt_box_from_annotation(s)
            dets = detect_multi_template(img, templates, threshold=info['thresh'])
            res  = ev.evaluate_detections(dets, [gt])

            if saved_count < 5:
                filename = os.path.join(OUTPUT_DIR, f"{info['name']}_sample_{i}.jpg")
                draw_and_save(img, gt, dets[0] if dets else None, filename)
                saved_count += 1

            metrics['TP'] += res['TP']
            metrics['FP'] += res['FP']
            metrics['FN'] += res['FN']
            if res['TP'] > 0:
                metrics['iou'].append(res['avg_iou'])

        prec = metrics['TP'] / (metrics['TP'] + metrics['FP'] + 1e-9)
        rec  = metrics['TP'] / (metrics['TP'] + metrics['FN'] + 1e-9)
        final_report.append({
            'name': info['name'], 'total': len(test_data),
            'TP': metrics['TP'],  'FP': metrics['FP'],  'FN': metrics['FN'],
            'prec': prec,         'rec': rec,
            'iou': np.mean(metrics['iou']) if metrics['iou'] else 0,
        })

    print("\n" + "=" * 60)
    print(f"{'Class Name':<12} | {'Total':<5} | {'TP':<3} | {'FP':<3} | "
          f"{'FN':<3} | {'Prec':<5} | {'Rec':<5} | {'IoU':<5}")
    print("-" * 60)
    for r in final_report:
        print(f"{r['name']:<12} | {r['total']:<5} | {r['TP']:<3} | {r['FP']:<3} | "
              f"{r['FN']:<3} | {r['prec']:.2f}  | {r['rec']:.2f}  | {r['iou']:.2f}")
    print("=" * 60)
