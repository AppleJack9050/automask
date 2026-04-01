import time
from pathlib import Path
import os
from dotenv import load_dotenv

import random
import numpy as np
import torch
from kornia.utils import image_to_tensor
from PIL import Image

class FileEditor:
    def __init__(self, processed_directory, logger):
        self.processed_directory = processed_directory
        self.logger = logger

    def edit_image(self, image_name, target_dir, positive_prompt, highlight):
        image = Image.open(Path(target_dir / "original" / image_name)).convert("RGB")
        if highlight:
            self.highlight_mask_edit(
                target_dir=target_dir,
                image_name=image_name,
                image_tensor=image_to_tensor(np.array(image))                
            )
            return

        if positive_prompt:
            return self.positive_prompt_edit(
                target_dir=target_dir,
                image_name=image_name,
                image_tensor=image_to_tensor(np.array(image))
            )

        else:
            return self.negative_prompt_edit(
                target_dir=target_dir,
                image_name=image_name,
                image_tensor=image_to_tensor(np.array(image))
            )

    def highlight_mask_edit(self, target_dir, image_name, image_tensor):
        mask_files = os.listdir(f'{target_dir}/masks')
        self.__make_edited_image_directory(target_dir)

        for mask in mask_files:
            mask = Image.open(target_dir / "masks" / mask)
            image_tensor = self.__highlight_mask(image_tensor, image_to_tensor(np.array(mask)))  

        self.save_edited_image(image_tensor, target_dir, image_name)

    def positive_prompt_edit(self, target_dir, image_name, image_tensor):
        mask_files = os.listdir(f'{target_dir}/masks')
        self.__make_edited_image_directory(target_dir)

        combined_mask = torch.zeros_like(image_tensor, dtype=torch.bool)
        for mask in mask_files:
            mask = image_to_tensor(np.array(Image.open(target_dir / "masks" / mask)))
            combined_mask = torch.logical_or(combined_mask, mask)

        image_tensor = self.__keep_object_in_image(image_tensor, combined_mask)
        self.save_edited_image(image_tensor, target_dir, image_name)

    def negative_prompt_edit(self, target_dir, image_name, image_tensor):
        mask_files = os.listdir(f'{target_dir}/masks')                
        self.__make_edited_image_directory(target_dir)

        for mask in mask_files:
            mask = Image.open(target_dir / "masks" / mask)
            image_tensor = self.__remove_mask_from_image(image_tensor, image_to_tensor(np.array(mask)))

        self.save_edited_image(image_tensor, target_dir, image_name)

    def __remove_mask_from_image(self, image_tensor, mask_tensor) -> torch.Tensor:
        device = image_tensor.device
        mask_tensor = mask_tensor.to(device)
        mask_tensor = mask_tensor > 0

        edited_image =  image_tensor.float() * ~mask_tensor
        return edited_image

    def __keep_object_in_image(self, image_tensor, mask_tensor) -> torch.Tensor:
        device = image_tensor.device
        mask_tensor = mask_tensor.to(device)
        mask_tensor = mask_tensor > 0

        edited_image = image_tensor.float() * mask_tensor
        return edited_image.clamp(0)

    def __highlight_mask(self, image_tensor, mask_tensor) -> torch.Tensor:
        alpha = 0.5

        coloured_mask = torch.rand(4, device=image_tensor.device)
        coloured_mask[3] = alpha
        coloured_mask = coloured_mask.view(1, 4, 1, 1)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
 
        if image_tensor.shape[1] == 3:
            image_tensor = torch.cat(
                [
                    image_tensor,
                    torch.ones_like(image_tensor[:, :1, :, :])
                ],
                dim=1
            )

        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)

        mask_tensor = mask_tensor.float().repeat(1, 4, 1, 1)
        overlay = mask_tensor * coloured_mask 
        image_tensor_overlay = image_tensor.float() + overlay * alpha

        return image_tensor_overlay.clamp(0)

    def __make_edited_image_directory(self, target_directory):
        edited_dir = Path(target_directory) / "edited"
        edited_dir.mkdir(parents=True, exist_ok=True)

    def save_edited_image(self, image_tensor, target_dir, image_name):
        image_tensor = (image_tensor.clamp(0, 255)).to(torch.uint8)

        if image_tensor.dim() == 3:
            image = image_tensor.permute(1, 2, 0).cpu().numpy()

        else:
            image = (
                image_tensor
                .squeeze(0)
                .permute(1, 2, 0) 
                .cpu()
                .numpy()
            )

        image = Image.fromarray(image)
        image.save(Path(target_dir / "edited" / image_name).with_suffix(".png"))
