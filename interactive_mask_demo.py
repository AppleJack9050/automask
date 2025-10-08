import cv2
import numpy as np
import tkinter as tk
import time

# ======================
# Config
# ======================
IMAGE_PATH = "sample.jpg"  # change to your own image
MASK_RADIUS = 60           # base radius for demo mask
ALPHA_SCALE = 0.5          # max transparency (0=fully transparent, 1=opaque)
TARGET_FPS = 30            # limit refresh rate (frames per second)


# ======================
# Get screen size
# ======================
def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h


# ======================
# Mouse state
# ======================
mouse_pos = None  # store in original image coordinates
scale = 1.0       # global scale for mapping


def on_mouse(event, x, y, flags, param):
    """Mouse callback: update mouse position"""
    global mouse_pos, scale
    if event == cv2.EVENT_MOUSEMOVE:
        # Map display coords back to original image coords
        x_orig = int(x / scale)
        y_orig = int(y / scale)
        mouse_pos = (x_orig, y_orig)


# ======================
# Mask generation
# ======================
def generate_mask(h, w, center, r):
    """
    Generate a soft circular mask.
    Replace with your own algorithm output.
    """
    if center is None:
        return np.zeros((h, w), np.uint8)

    cx, cy = center
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    mask = np.zeros((h, w), dtype=np.float32)
    inner = dist <= r
    feather = (dist > r) & (dist < r * 1.5)

    mask[inner] = 1.0
    mask[feather] = 1.0 - (dist[feather] - r) / (0.5 * r)

    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5)
    return mask


def apply_overlay(base_bgr, mask, alpha_scale=0.5):
    """Apply semi-transparent overlay"""
    alpha = (mask.astype(np.float32) / 255.0) * alpha_scale
    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)

    overlay_color = np.zeros_like(base_bgr, dtype=np.float32)
    base_f = base_bgr.astype(np.float32)
    out = base_f * (1 - alpha_3) + overlay_color * alpha_3
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================
# Main
# ======================
def main():
    global scale

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")
    h, w = img.shape[:2]

    # Get screen size and compute scale
    screen_w, screen_h = get_screen_size()
    scale = min(screen_w * 0.8 / w, screen_h * 0.8 / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)

    # Prepare window
    cv2.namedWindow("Interactive Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interactive Mask", disp_w, disp_h)
    cv2.setMouseCallback("Interactive Mask", on_mouse)

    print("Move mouse over the window, press ESC to quit.")

    # Throttle control
    min_interval = 1.0 / TARGET_FPS
    last_time = 0.0

    while True:
        now = time.time()
        if now - last_time < min_interval:
            # Skip this loop iteration to maintain FPS cap
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        last_time = now

        frame = img.copy()

        # Generate mask in original resolution
        mask = generate_mask(h, w, mouse_pos, MASK_RADIUS)

        # Apply overlay
        frame = apply_overlay(frame, mask, ALPHA_SCALE)

        # Draw mouse dot (mapped back to original)
        if mouse_pos is not None:
            cv2.circle(frame, mouse_pos, 3, (0, 255, 255), -1)

        # Resize for display
        disp_frame = cv2.resize(frame, (disp_w, disp_h))

        cv2.imshow("Interactive Mask", disp_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
