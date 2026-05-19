from pathlib import Path
from PIL import Image
import numpy as np
import gc
import os
import torch

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model, load_image, predict, box_convert
except:
    print("Grounded SAM2 Dependencies not installed. skipping...")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

class SAM2Processor():
    def __init__(self, device, upload_directory):
        self.device = device
        print("Building SAM2...")
        self.sam2_model = self.__build_sam2()
        print("SAM2 Built")
        print("Building DINO...")
        self.dino = self.__build_dino()
        self.upload_directory = upload_directory

    def process_text_prompt(self, user: str, mask_dir: str, file: str, prompt: str):
        masks = self.__generate_grounded_sam2_mask(prompt, str(Path(self.upload_directory) / user / file))
        
        for index, mask in enumerate(masks):
            mask_image = np.asarray(mask)
            if mask.dtype != np.uint8:
                mask = np.clip(mask, 0, 1) * 255
                mask = mask.astype(np.uint8)

            if mask.ndim == 3:
                mask = mask.squeeze()

            mask_image = Image.fromarray(mask, mode="L")
            mask_image.save(mask_dir / f"mask_{index}.png")
        del masks
        gc.collect()

    def process_empty_prompt(self, user: str, mask_dir: str, file: str):
        masks = self.__generate_sam2_masking(str(Path(self.upload_directory) / user / file))

        for index, mask in enumerate(masks):
            mask_image = mask["segmentation"]

            if mask_image.dtype == bool:
                mask_image = mask_image.astype(np.uint8) * 255

            mask_image = Image.fromarray(mask["segmentation"])
            mask_image.save(mask_dir / f"mask_{index}.png")
        
        del masks
        gc.collect()

    def __generate_sam2_masking(self, image: str):
        image = Image.open(image)

        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
        masks = mask_generator.generate(image_np)
        del mask_generator
        gc.collect()
        
        return masks

    def __build_sam2(self):
        cwd = os.getcwd()
        os.chdir(os.getenv("SAM2_FULL_PATH"))
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        os.chdir(cwd)
        return sam2_model

    def __build_dino(self):
        return load_model(
            model_config_path=os.getenv("GROUNDING_DINO_CONFIG"),
            model_checkpoint_path=os.getenv("GROUNDING_DINO_CHECKPOINT"),
            device=self.device
        )

    def __generate_grounded_sam2_mask(self, prompt: str, image: str):
        sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        image_source, image = load_image(image)
        sam2_predictor.set_image(image_source)
        prompt = prompt.lower()
        if not prompt.endswith('.'):
            prompt = prompt + '.'

        box_prompts, _, _ = predict(
            model=self.dino,
            image=image,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.device
        )

        h, w, _ = image_source.shape
        box_prompts = box_prompts* torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=box_prompts, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        masks = []
        for box in input_boxes:
            box_masks, scores, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            masks.append(box_masks)

        return masks
