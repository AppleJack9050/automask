import time
import tkinter as tk
from pathlib import Path
import os
from dotenv import load_dotenv

import numpy as np
import torch
import kornia
from kornia.geometry.transform import resize
from kornia.utils import image_to_tensor, tensor_to_image
from PIL import Image, ImageTk
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ======================
# Config
# ======================
IMAGE_PATH = "sample.jpg"  # change to your own image
MASK_RADIUS = 60           # base radius for demo mask
ALPHA_SCALE = 0.5          # max transparency (0=fully transparent, 1=opaque)
TARGET_FPS = 30            # limit refresh rate (frames per second)
GAUSSIAN_KERNEL = (21, 21)
GAUSSIAN_SIGMA = (5.0, 5.0)


# ======================
# Screen size helpers
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
mouse_clicked = False # global var to say if mouse clicked

# ======================
# Mask generation
# ======================
def generate_mask(h, w, center, radius, device, blur_module):
    """Generate a soft circular mask with Kornia filters."""
    mask_layer = torch.zeros((h, w), dtype=torch.float32, device=device)
    if center is None:
        return mask_layer.unsqueeze(0).unsqueeze(0)

    cx, cy = center
    cx_t = torch.tensor(float(cx), device=device)
    cy_t = torch.tensor(float(cy), device=device)
    x_coords = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
    y_coords = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
    dist = torch.sqrt((x_coords - cx_t) ** 2 + (y_coords - cy_t) ** 2)

    inner = dist <= radius
    feather = (dist > radius) & (dist < radius * 1.5)
    mask_layer[inner] = 1.0
    mask_layer[feather] = 1.0 - (dist[feather] - radius) / (0.5 * radius)

    mask = mask_layer.unsqueeze(0).unsqueeze(0)
    return blur_module(mask).clamp(0.0, 1.0)


def apply_overlay(base_tensor, mask_tensor, alpha_scale):
    """Apply a semi-transparent overlay using Kornia tensors."""
    alpha = mask_tensor * alpha_scale
    return base_tensor * (1.0 - alpha)


def add_highlight(frame_tensor, center, device, radius=3):
    """Highlight the current cursor position."""
    if center is None:
        return frame_tensor

    frame = frame_tensor.clone()
    x, y = center
    h = frame.shape[-2]
    w = frame.shape[-1]
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return frame

    highlight_color = torch.tensor([0.0, 1.0, 1.0], device=device).view(1, 3, 1, 1)
    frame[:, :, y0:y1, x0:x1] = highlight_color
    return frame

def generate_sam2_masking(image, device):
    """Generates the object masks from sam2"""
    image = Image.open(image)
    image = image.resize((1024, 1024))
    image = np.array(image.convert("RGB"))

    os.chdir(os.getenv("SAM2_FULL_PATH"))
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    masks = mask_generator.generate(image)
    
    return masks

def render_current_object_mask(image_tensor, mask, remove_object_from_image=False):
    """When an object is hovered over/clicked this will either highlight or remove the object from the image"""

    if mask is None:
        return image_tensor
    
    device = image_tensor.device
    mask_tensor = image_to_tensor(mask["segmentation"]).to(device)

    if image_tensor.shape[1] < 4:
        image_tensor = torch.cat([image_tensor, torch.ones_like(image_tensor[:, :1, :, :])], dim=1)

    overlay = torch.zeros(mask_tensor.shape, device=device)

    if remove_object_from_image:
        image_with_overlay = image_tensor.float() * ~mask_tensor
        return image_with_overlay.clamp(0)
    
    coloured_mask = torch.tensor([1.0, 0.0, 0.0, ALPHA_SCALE], device=device).view(4, 1, 1)
    mask = mask_tensor.float().unsqueeze(0).repeat(1, 4, 1, 1)
    overlay = mask * coloured_mask 

    image_with_overlay = image_tensor.float() + overlay * (ALPHA_SCALE * mask_tensor)
    return image_with_overlay.clamp(0)

def coords_in_mask(mask, x, y) -> bool:
    if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]:
        return mask[x, y] != 0 
    return False

def find_mask_by_coord(masks, x, y):
    for mask in masks:
        if coords_in_mask(mask["segmentation"], x, y):
            return mask
    return None

# ======================
# Main
# ======================
def main():
    load_dotenv() 
    global scale

    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    pil_image = Image.open(image_path).convert("RGB")
    # Arbitrary resizing to speed up processing time
    pil_image = pil_image.resize((1024, 1024))

    w, h = pil_image.size

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    base_tensor = image_to_tensor(np.array(pil_image)).float().unsqueeze(0).to(device) / 255.0
    shown_tensor = base_tensor

    screen_w, screen_h = get_screen_size()
    scale = min(screen_w * 0.8 / w, screen_h * 0.8 / h, 1.0)
    disp_w = max(1, int(w * scale))
    disp_h = max(1, int(h * scale))

    blur_module = kornia.filters.GaussianBlur2d(GAUSSIAN_KERNEL, GAUSSIAN_SIGMA).to(device)
    blur_module.eval()

    root = tk.Tk()
    root.title("Interactive Mask (Kornia)")
    root.geometry(f"{disp_w}x{disp_h}")

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, highlightthickness=0)
    canvas.pack()
    canvas_image = canvas.create_image(0, 0, anchor="nw")

    min_interval = 1.0 / TARGET_FPS
    last_frame_time = 0.0

    def handle_mouse_move(event):
        global mouse_pos, scale
        if scale <= 0:
            return
        x_orig = int(event.x / scale)
        y_orig = int(event.y / scale)
        x_orig = max(0, min(w - 1, x_orig))
        y_orig = max(0, min(h - 1, y_orig))
        mouse_pos = (x_orig, y_orig)

    def handle_mouse_leave(_event):
        global mouse_pos
        mouse_pos = None

    def handle_mouse_click(_event):
        global mouse_clicked
        mouse_clicked = True

    canvas.bind("<Motion>", handle_mouse_move)
    canvas.bind("<Leave>", handle_mouse_leave)
    canvas.bind("<Button>", handle_mouse_click)
    root.bind("<Escape>", lambda _e: root.destroy())

    sam2_object_masks = generate_sam2_masking(image_path, device)

    def get_current_object_mask(mouse_pos):
        nonlocal sam2_object_masks
        if mouse_pos is None:
            return None
        x, y = mouse_pos
        return find_mask_by_coord(sam2_object_masks, x, y)

    def remove_mask(mask_to_remove):
        if mask_to_remove or sam2_object_masks is None:
            return

        index_to_remove = None
        for i, mask in enumerate(sam2_object_masks):
            if np.array_equal(mask_to_remove["segmentation"], mask["segmentation"]):
                index_to_remove = i
                break

        if index_to_remove is not None:
            sam2_object_masks.pop(index_to_remove)
        else:
            pass

    def render_frame():
        global mouse_clicked
        nonlocal last_frame_time
        nonlocal sam2_object_masks
        nonlocal shown_tensor
        now = time.time()

        if now - last_frame_time >= min_interval:
            mask_tensor = generate_mask(h, w, mouse_pos, MASK_RADIUS, device, blur_module)
            frame_tensor = apply_overlay(shown_tensor, mask_tensor, ALPHA_SCALE)
            frame_tensor = add_highlight(frame_tensor, mouse_pos, device)
            disp_tensor = resize(frame_tensor, (h, w), align_corners=False)
            disp_tensor_with_mask_overlay = disp_tensor

            current_mask = get_current_object_mask(mouse_pos)
            if mouse_clicked:
                disp_tensor_with_mask_overlay = render_current_object_mask(
                    image_tensor=disp_tensor,
                    mask=current_mask,
                    remove_object_from_image=True
                )
                mouse_clicked = False
                shown_tensor = disp_tensor_with_mask_overlay[:, :3, :, :]
                mouse_clicked = False
            else:
                disp_tensor_with_mask_overlay = render_current_object_mask(
                    image_tensor=disp_tensor,
                    mask=current_mask
                )

            disp_np = tensor_to_image(disp_tensor_with_mask_overlay.squeeze(0).cpu().clamp(0.0, 1.0))
            disp_img = Image.fromarray((disp_np * 255).astype(np.uint8))
            photo = ImageTk.PhotoImage(image=disp_img)
            canvas.itemconfig(canvas_image, image=photo)
            canvas.image = photo
            last_frame_time = now
        root.after(5, render_frame)

    print("Move mouse over the window, press ESC to quit.")
    print("Click on higlighted mask to remove it from base image")
    render_frame()
    root.mainloop()


if __name__ == "__main__":
    main()
