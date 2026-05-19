try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except Exception as e:
    print("SAM3 not installed. Skipping...")
from huggingface_hub import hf_hub_download
import traceback
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import gc

class SAM3Processor():
    def __init__(self, device, upload_directory):
        print("Building SAM3...")
        self.upload_directory = upload_directory
        checkpoint_path = self.download_checkpoints()
        self.device = device
        self.model = build_sam3_image_model(
            device=device,
            bpe_path="/image-processor/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=checkpoint_path
        )

        self.processor = Sam3Processor(
            model=self.model,
            device=self.device    
        )
        print("SAM3 Built")

    def process_text_prompt(self, user: str, mask_dir: str, file: str, prompt: str):
        image = Image.open(str(Path(self.upload_directory) / user / file)).convert("RGB")
        individual_prompts = prompt.split(".")
        all_masks = []
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

        for p in individual_prompts:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                output = self.processor.set_text_prompt(state=inference_state, prompt=p)
                all_masks.extend(output["masks"])

        self.__save_masks(all_masks, mask_dir)

    def process_empty_prompt(self, user: str, mask_dir: str, file: str):
        image = Image.open(str(Path(self.upload_directory) / user / file)).convert("RGB")

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            output = self.processor.add_geometric_prompt(
                box=[0.5, 0.5, 1.0, 1.0],
                label=True,
                state=inference_state
            )
        self.__save_masks(output, mask_dir)
    
    def __save_masks(self, masks: dict, mask_dir: str):
        for index, mask in enumerate(masks):
            mask_np = mask.squeeze().cpu().numpy()
            if mask_np.dtype == bool or mask.dtype == torch.bool:
                mask_np = mask_np.astype(np.uint8) * 255

            mask= Image.fromarray(mask_np, mode="L")
            mask.save(mask_dir / f"mask_{index}.png")

        del masks
        gc.collect()

    def download_checkpoints(self):
        checkpoint_path = hf_hub_download(
            repo_id="facebook/sam3.1",
            filename="sam3.1_multiplex.pt"
        )
        return checkpoint_path
