import numpy as np

def compute_iou(det, gt):
    """det & gt: {'x', 'y', 'w', 'h'}"""
    ix1 = max(det['x'], gt['x'])
    iy1 = max(det['y'], gt['y'])
    ix2 = min(det['x'] + det['w'], gt['x'] + gt['w'])
    iy2 = min(det['y'] + det['h'], gt['y'] + gt['h'])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = det['w']*det['h'] + gt['w']*gt['h'] - inter
    return inter / (union + 1e-9)

def evaluate_detections(detections, gt_boxes, iou_threshold=0.5):
    matched = [False] * len(gt_boxes)
    results = []
    for det in sorted(detections, key=lambda d: -d['score']):
        best_iou, best_idx = 0, -1
        for i, gt in enumerate(gt_boxes):
            v = compute_iou(det, gt)
            if v > best_iou:
                best_iou, best_idx = v, i
        is_tp = best_iou >= iou_threshold and not matched[best_idx]
        if is_tp:
            matched[best_idx] = True
        results.append({**det, 'iou': best_iou, 'tp': is_tp})
    
    TP = sum(r['tp'] for r in results)
    FP = len(results) - TP
    FN = sum(1 for m in matched if not m)
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    avg_iou   = np.mean([r['iou'] for r in results if r['tp']]) if TP else 0
    return {'precision': precision, 'recall': recall,
            'avg_iou': avg_iou, 'TP': TP, 'FP': FP, 'FN': FN}