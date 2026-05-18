try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except Exception as e:
    print("SAM3 not installed")
from huggingface_hub import hf_hub_download
from torchvision.transforms import v2
import torch
import inspect
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
            bpe_path="/image-processor/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=checkpoint_path
        )

        self.processor = Sam3Processor(self.model)
        print("SAM3 Built")

    def process_text_prompt(self, user: str, mask_dir: str, file: str, prompt: str):
        image = Image.open(str(Path(self.upload_directory) / user / file)).convert("RGB")

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        self.__save_masks(output, mask_dir)

    def process_empty_prompt(self, user: str, mask_dir: str, file: str):
        image = np.array(Image.open((str(Path(self.upload_directory) / user / file)).convert("RGB")))
        self.processor.set_image(image)
        height, width = image.shape[:2]
        masks = []
        batch_size = 64

        points, labels = self.__make_point_predictions(height, width)

        for i in range(0, len(points), batch_size):

            batch_points = points[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            masks, _, _ = self.processor.predict(
                point_coords=batch_points[:, None, :],
                point_labels=batch_labels[:, None],
                multimask_output=True,
            )

            masks.extend(masks)

        self.__save_masks(masks, mask_dir)

    def __make_point_predictions(self, height, width) -> tuple:
        grid_size = 32

        xs = np.linspace(0, width - 1, grid_size)
        ys = np.linspace(0, height - 1, grid_size)
        points = np.array([
            [x, y]
            for y in ys
            for x in xs
        ])

        labels = np.ones(len(points))

        return points, labels

    def __save_masks(self, masks: list, mask_dir: str):
        for index, mask in enumerate(masks["masks"]):
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            
            mask.save(mask_dir / f"mask_{index}.png")

        del masks
        gc.collect()

    def download_checkpoints(self):
        checkpoint_path = hf_hub_download(
            repo_id="facebook/sam3.1",
            filename="sam3.1_multiplex.pt"
        )
        return checkpoint_path
