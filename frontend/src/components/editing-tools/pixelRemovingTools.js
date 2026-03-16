export default class pixelRemovingTools {
  static restorePixel(shownPixels, basePixels, i){
    shownPixels[i] = basePixels[i]; 
    shownPixels[i + 1] = basePixels[i + 1];
    shownPixels[i + 2] = basePixels[i + 2];
    shownPixels[i + 3] = 255;  
  }
  static turnPixelTransparent(shownPixels, i) {
    shownPixels[i + 3] = 0;
  }
  static turnPixelBlack(shownPixels, i) {
    shownPixels[i] = 0; 
    shownPixels[i + 1] = 0;
    shownPixels[i + 2] = 0;
    shownPixels[i + 3] = 255;  
  }    
}