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
    let layers = []
    for (const layer of masks) {
      const img = new Image();
      img.src = `data:image/png;base64,${await MaskSetup.convertMaskTransparant(layer)}`;
      await img.decode();

      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      layers.push({
        img,
        width: img.width,
        height: img.height,
        data: ctx.getImageData(0, 0, img.width, img.height).data
      });
    }
    return layers;
  }
}