import { useUndoStore } from '@/stores/undostore';
import { useRedoStore } from "@/stores/redostore";
import pixelRemovingTools from './pixelRemovingTools';

export default class MaskSelector {
  constructor(fileName) {
    this.undoStore = useUndoStore();
    this.redoStore = useRedoStore();
    this.fileName = fileName;
  }
  async removeObject(id, undo = false, image, mask, transparentBackground, basePixels = null) {
    if (undo) {
      this.redoStore.addFileFuture(this.fileName, {id:[id], action:'remove', undo:true});
    } else {
      this.undoStore.addFileHistory(this.fileName, {id:[id], action:'remove', undo:false});
    }

    const editSetup = await this.editSetup(mask, image);

    for (let i = 0; i < editSetup.maskData.length; i += 4) {
      const r = editSetup.maskData[i];
      const g = editSetup.maskData[i + 1];
      const b = editSetup.maskData[i + 2];
      if (r > 0 && g > 0 && b > 0) {
        if (undo && basePixels) {
          pixelRemovingTools.restorePixel(editSetup.shownPixels, basePixels, i);
        } else {
          transparentBackground ? 
            pixelRemovingTools.turnPixelTransparent(editSetup.shownPixels, i) :
            pixelRemovingTools.turnPixelBlack(editSetup.shownPixels, i);

            editSetup.shownPixels[i + 3] = 0;
        }
      }
    }

    editSetup.ctx.putImageData(editSetup.shownImageData, 0, 0);
    return editSetup.canvas.toDataURL('image/png').split(',')[1];
  }

  async selectOnlyObject(id, undo = false, image, mask, transparentBackground, basePixels = null) {
    if (undo) {
      this.redoStore.addFileFuture(this.fileName, {id:[id], action:'select', undo:true});
    } else {
      this.undoStore.addFileHistory(this.fileName, {id:[id], action:'select', undo:false});
    }
    const editSetup = await this.editSetup(mask, image);

    for (let i = 0; i < editSetup.maskData.length; i += 4) {
      const r = editSetup.maskData[i];
      const g = editSetup.maskData[i + 1];
      const b = editSetup.maskData[i + 2];

      if (r == 0 && g == 0 && b == 0) {
        if (undo && basePixels) {
          pixelRemovingTools.restorePixel(editSetup.shownPixels, basePixels, i);
        } else {
          transparentBackground ? 
            pixelRemovingTools.turnPixelTransparent(editSetup.shownPixels, i) :
            pixelRemovingTools.turnPixelBlack(editSetup.shownPixels, i);
        }
      }
    }

    editSetup.ctx.putImageData(editSetup.shownImageData, 0, 0);
    return editSetup.canvas.toDataURL('image/png').split(',')[1];
  }

  async editSetup(mask, image) {
    const maskImg = new Image();
    maskImg.src = `data:image/png;base64,${mask}`;
    await maskImg.decode();

    const shownImage = new Image();
    shownImage.src = `data:image/png;base64,${image}`;
    await shownImage.decode();
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = shownImage.width;
    canvas.height = shownImage.height;

    ctx.drawImage(shownImage, 0, 0);

    const shownImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let shownPixels = shownImageData.data;

    const maskCanvas = document.createElement('canvas');
    const maskCtx = maskCanvas.getContext('2d');
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;

    maskCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;

    return {
      shownPixels:shownPixels,
      maskData:maskData,
      shownImageData:shownImageData,
      ctx:ctx,
      canvas:canvas
    };
  }
}
