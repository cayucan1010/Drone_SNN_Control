import torch
import numpy as np

def compute_iou(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """
    IoU between predicted and ground truth bounding boxes.
    Boxes: [xcenter, ycenter, w, h] normalized.
    Detection successful if IoU > 0.5.
    """
    def to_corners(box):
        xc, yc, w, h = box
        return xc - w/2, yc - h/2, xc + w/2, yc + h/2

    px1, py1, px2, py2 = to_corners(pred_box)
    gx1, gy1, gx2, gy2 = to_corners(gt_box)

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area  = (px2 - px1) * (py2 - py1)
    gt_area    = (gx2 - gx1) * (gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    return inter_area / (union_area + 1e-8)

def mean_absolute_error_center(preds: np.ndarray, gts: np.ndarray) -> float:
    """MAE on gate center coordinates (xcenter, ycenter)."""
    return float(np.mean(np.abs(preds[:, :2] - gts[:, :2])))

def evaluate(model, test_loader):
    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.permute(1, 0, 2, 3, 4)
            preds = model(x_batch).numpy()
            all_preds.append(preds)
            all_gts.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_gts   = np.concatenate(all_gts)

    ious = [compute_iou(p, g) for p, g in zip(all_preds, all_gts)]
    detection_rate = np.mean([iou > 0.5 for iou in ious])
    center_mae = mean_absolute_error_center(all_preds, all_gts)

    print(f"Detection Rate (IoU>0.5): {detection_rate:.2%}")
    print(f"Center MAE: {center_mae:.4f}")
    return detection_rate, center_mae