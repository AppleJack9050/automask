import { useUndoStore } from '@/stores/undostore';
import { useRedoStore } from "@/stores/redostore";
import pixelRemovingTools from './pixelRemovingTools';

export default class TouchUp {
  constructor(radius, scaleX, scaleY, fileName) {
    this.touchUpRadius = radius;
    this.scaleX = scaleX;
    this.scaleY = scaleY;
    this.fileName = fileName;
    this.undoStore = useUndoStore();
    this.redoStore = useRedoStore();
  }
  updateTouchUpRadius(value) {
    this.touchUpRadius = value;
  }
  removePixels(svgX, svgY, shownData, transparentBackground, restoreBackground, basePixels) {
    const imgX = Math.round(svgX * this.scaleX);
    const imgY = Math.round(svgY * this.scaleY);
    const radiusSq = this.touchUpRadius ** 2;
    const width = shownData.width;

    const startX = Math.max(0, imgX - this.touchUpRadius);
    const endX = Math.min(width, imgX + this.touchUpRadius);
    const startY = Math.max(0, imgY - this.touchUpRadius);
    const endY = Math.min(shownData.height, imgY + this.touchUpRadius);

    for (let y = startY; y < endY; y++) {
      const rowOffset = y * width;
      for (let x = startX; x < endX; x++) {
        if ((x - imgX) ** 2 + (y - imgY) ** 2 <= radiusSq) {
          const idx = (rowOffset + x) * 4;

          if (restoreBackground) {
            pixelRemovingTools.restorePixel(shownData.data, basePixels, x);
          }
          transparentBackground ? 
            pixelRemovingTools.turnPixelTransparent(shownData.data, idx) :
            pixelRemovingTools.turnPixelBlack(shownData.data, idx);
        }
      }
    }

    return shownData;
  }
  resetTouchUp(canvas, ctx, preSnapshot, postSnapshot, undo, restore = false) {
    if (undo) {
      this.redoStore.addFileFuture(
        this.fileName,
        {
          action:'touchUp',
          undo:true,
          preTouchUpSnapshot:preSnapshot,
          postTouchUpSnapshot:postSnapshot
        });
    } else {
      this.undoStore.addFileHistory(
      this.fileName,
      {
        action:'touchUp',
        undo:false,
        preTouchUpSnapshot:preSnapshot,
        postTouchUpSnapshot:postSnapshot
      });
    }

    if (restore) {
      ctx.putImageData(postSnapshot, 0, 0);
      return canvas.toDataURL('image/png').split(',')[1];
    } else {
      ctx.putImageData(preSnapshot, 0, 0);
      return canvas.toDataURL('image/png').split(',')[1];
    }
  }
}
