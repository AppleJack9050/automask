export default class MaskSetup {
  static async convertMaskTransparant(mask) {
    const img = new Image();
    img.src = `data:image/png;base64,${mask.mask}`;
    await img.decode();

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      if (r > 0 || g > 0 || b > 0) {
        data[i + 3] = 255;
      } else {
        data[i + 3] = 0;
      }
    }
    ctx.putImageData(imageData, 0, 0);
    const newBase64 = canvas.toDataURL('image/png').split(',')[1];
    return newBase64;
  }
  static async createSVGLayers(masks) {
    const layers = [];

    for (const layer of masks) {
      const base64 = await MaskSetup.convertMaskTransparant(layer);
      const byteString = atob(base64);
      const bytes = new Uint8Array(byteString.length);
      for (let i = 0; i < byteString.length; i++) {
        bytes[i] = byteString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'image/png' });
  
      const img = await createImageBitmap(blob);
  
      const canvas = new OffscreenCanvas(img.width, img.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
  
      layers.push({
        width: img.width,
        height: img.height,
        data: new Uint8ClampedArray(imageData.data)
      });
      img.close();
    }
    return layers;
  }
}
