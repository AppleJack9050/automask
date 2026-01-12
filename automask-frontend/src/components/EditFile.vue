<template>
  <div >
    <div
      class="svg-mask-viewer"
      v-if="!loading"
    >
      <svg
        :width="this.imageWidth"
        :height="this.imageHeight"
        :viewBox="`0 0 ${this.imageWidth} ${this.imageHeight}`"
        @mousemove="highlight"
        @mouseleave="this.hoveredMask = null; this.editingPixels = false"
        @mousedown.left="this.editingPixels = true"
        @mouseup.left="this.editingPixels = false"
        @mouseup.right="this.touchingUp = false"
      >
        <image
          :href="`data:image/png;base64,${shownImage}`"
          width="100%"
          height="100%"
        />
        <g v-for="(mask, index) in masks" :key="index">
          <image
            :ref="el => maskRefs[index] = el"
            :href="`data:image/png;base64,${mask.mask}`"
            :x="mask.x || 0"
            :y="mask.y || 0"
            :width="mask.width"
            :height="mask.height"
            :opacity="hoveredMask === index ? 0.5 : 0"
            class="mask"
            @contextmenu="handleRightClick($event, index)"
          />
        </g>
        <circle
          :opacity="touchingUp ? 0.5 : 0"
          :cx="highlightX"
          :cy="highlightY"
          :r="touchUpRadius"
          fill="red"
          class="highlight"
          @contextmenu="handleRightClick($event, index)"
        >
        </circle>
      </svg>
      <ContextMenu
        v-if="contextId !== null"
        :id="contextId"
        :usingTouchUp="touchingUp"
        @select="selectOnlyObject"
        @remove="removeObject"
        @save="save"
        @start="showTouchUp"
        @end="hideTouchUp"
        :style="{ top: contextY + 'px', left: contextX + 'px' }"
        ></ContextMenu>
        <br></br>
        <div>
          <label for="slider" class="form-label">Toggle Original Image</label>
          <input
            type="checkbox"
            v-model="showOriginalImage"
          />
        </div>
        <div v-if="touchingUp">
          <label for="slider" class="form-label">Select Brush Size: {{ touchUpRadius }} px</label>
          <input
            type="range"
            class="form-range"
            min="5"
            max="100"
            step="5"
            v-model="touchUpRadius"
          />
        </div>
    </div>
    <div v-else>
      <loading></loading>
    </div>
  </div>
</template>

<script>
import ContextMenu from './ContextMenu.vue';
import Loading from './Loading.vue';
export default {
  components: {
    ContextMenu,
    Loading
  },
  props: {
    baseImage:{
      type:String,
      required:true
    },
    editedImage:{
      type:String,
      required:false
    },
    masks:{
      type: Array,
      default: () => []
    }
  },
  data() {
    return {
      hoveredMask: null,
      maskRefs: [],
      layers: [],
      loading:true,
      contextId: null,
      contextX: null,
      contextY: null,
      shownImage: null,
      touchingUp: false,
      touchUpRadius:10,
      highlightX: null,
      highlightY: null,
      editingPixels: false,
      touchUpCtx: null,
      touchUpCanvas: null,
      imageHeight: null,
      imageWidth: null,
      showOriginalImage: false
    }
  },
  computed: {
  },
  methods: {
    highlight(e) {
      if (this.touchingUp) {
        if (this.editingPixels) {
          this.handleTouchUp(e);
          return this.removePixels(e);
        }
        return this.handleTouchUp(e);
      }

      const svg = e.currentTarget;
      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());
      const masks = svg.querySelectorAll('.mask');

      if (masks.length === 0 ) {
        return;
      }

      const x = parseFloat(masks[0].getAttribute('x'));
      const y = parseFloat(masks[0].getAttribute('y'));

      const imgX = Math.floor((svgPoint.x - x) * (this.layers[0].width / this.imageWidth));
      const imgY = Math.floor((svgPoint.y - y) * (this.layers[0].height / this.imageHeight));

      if (imgX < 0 || imgY < 0 || imgX >= masks[0].width || imgY >= masks[0].height) {
        this.hoveredMask = null
        return
      }

      for (let i = this.layers.length - 1; i >= 0; i--) {
        const { width, data } = this.layers[i];
        const idx = ((imgY * width) + imgX) * 4 + 3
        const alpha = data[idx];
        if (alpha > 0) {
          this.hoveredMask = i;
          return;
        }
      }
    },
    async convertMaskTransparant(mask) {
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
    },
    handleRightClick(event) {
      event.preventDefault();
      this.contextX = event.clientX;
      this.contextY = event.clientY;
      this.contextId = this.hoveredMask;
    },
    async removeObject(id) {
      const maskToRemove = this.masks[id].mask;
      const maskImg = new Image();
      maskImg.src = `data:image/png;base64,${maskToRemove}`;
      await maskImg.decode();

      const shownImage = new Image();
      shownImage.src = `data:image/png;base64,${this.shownImage}`;
      await shownImage.decode();

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = shownImage.width;
      canvas.height = shownImage.height;

      ctx.drawImage(shownImage, 0, 0);

      const shownImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const shownPixels = shownImageData.data;

      const maskCanvas = document.createElement('canvas');
      const maskCtx = maskCanvas.getContext('2d');
      maskCanvas.width = canvas.width;
      maskCanvas.height = canvas.height;
      maskCtx.imageSmoothingEnabled = false;
      maskCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;

      for (let i = 0; i < maskData.length; i += 4) {
        const r = maskData[i];
        const g = maskData[i + 1];
        const b = maskData[i + 2];
        if (r > 0 && g > 0 && b > 0) {
          shownPixels[i + 3] = 0;
        }
      }

      ctx.putImageData(shownImageData, 0, 0);
      this.shownImage = canvas.toDataURL('image/png').split(',')[1];
      this.contextId = null;
    },
    async selectOnlyObject(id) {
      const maskToRemove = this.masks[id].mask;
      const maskImg = new Image();
      maskImg.src = `data:image/png;base64,${maskToRemove}`;
      await maskImg.decode();

      const shownImage = new Image();
      shownImage.src = `data:image/png;base64,${this.shownImage}`;
      await shownImage.decode();

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = shownImage.width;
      canvas.height = shownImage.height;

      ctx.drawImage(shownImage, 0, 0);

      const shownImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const shownPixels = shownImageData.data;

      const maskCanvas = document.createElement('canvas');
      const maskCtx = maskCanvas.getContext('2d');
      maskCanvas.width = canvas.width;
      maskCanvas.height = canvas.height;

      maskCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;

      for (let i = 0; i < maskData.length; i += 4) {
        const r = maskData[i];
        const g = maskData[i + 1];
        const b = maskData[i + 2];
        if (r < 255 && g < 255 && b < 255) {
          shownPixels[i + 3] = 0;
        }
      }

      ctx.putImageData(shownImageData, 0, 0);
      this.shownImage = canvas.toDataURL('image/png').split(',')[1];
      this.contextId = null;
    },
    save() {
      this.$emit('saveImage', this.shownImage);
    },
    async removePixels(e) {
      const svg = e.currentTarget;
      if (!this.touchingUp || !svg) {
        return;
      }
      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());

      const shownImage = new Image();
      shownImage.src = `data:image/png;base64,${this.shownImage}`;
      await shownImage.decode();

      const svgInternalWidth = svg.viewBox.baseVal.width || svg.clientWidth;
      const svgInternalHeight = svg.viewBox.baseVal.height || svg.clientHeight;

      const scaleX =  shownImage.width / svgInternalWidth;
      const scaleY = shownImage.height / svgInternalHeight;

      const imgX = Math.floor(svgPoint.x * scaleX);
      const imgY = Math.floor(svgPoint.y * scaleY);

      this.touchUpCanvas.width = shownImage.width;
      this.touchUpCanvas.height = shownImage.height;
      this.touchUpCtx.drawImage(shownImage, 0, 0);

      const shownData = this.touchUpCtx.getImageData(0, 0, this.touchUpCanvas.width, this.touchUpCanvas.height);
      const shownPixels = shownData.data;

      const startX = Math.max(0, Math.floor(imgX - this.touchUpRadius));
      const endX = Math.min(this.touchUpCanvas.width, Math.ceil(imgX + this.touchUpRadius));
      const startY = Math.max(0, Math.floor(imgY - this.touchUpRadius));
      const endY = Math.min(this.touchUpCanvas.height, Math.ceil(imgY + this.touchUpRadius));

      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const dx = Math.pow((x - imgX), 2);
          const dy = Math.pow((y - imgY), 2);
          if (dx + dy <= this.touchUpRadius * this.touchUpRadius) {
            const pixelIndex = ((y* this.touchUpCanvas.width) + x) * 4;
            if (shownPixels[pixelIndex + 3] > 0) {
              shownPixels[pixelIndex + 3] = 0;
            }
          }
        }
      }

      this.touchUpCtx.putImageData(shownData, 0, 0);
      this.shownImage = this.touchUpCanvas.toDataURL('image/png').split(',')[1];
    },
    handleTouchUp(e) {
      const svg = e.currentTarget;
      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());

      const imgX = Math.floor((svgPoint.x));
      const imgY = Math.floor((svgPoint.y));

      if (imgX < 0 || imgY < 0 ) {
        return;
      }

      this.highlightX = imgX;
      this.highlightY = imgY;
    },
    hideTouchUp() {
      this.touchingUp = false;
      this.contextId = null;
    },
    showTouchUp() {
      this.touchingUp = true;
      this.contextId = null;
    }
  },
  watch: {
    baseImage: {
      immediate: true,
      async handler(newImage) {
        if (newImage) {
          try {
            this.touchUpCanvas = document.createElement('canvas');
            this.touchUpCtx = this.touchUpCanvas.getContext('2d');

            this.shownImage = this.editedImage != null ? this.editedImage : this.baseImage;
            const image = new Image();
            image.src = `data:image/png;base64,${this.baseImage}`;

            for (const layer of this.masks) {
              const img = new Image();
              img.src = `data:image/png;base64,${await this.convertMaskTransparant(layer)}`;
              await img.decode();
              this.imageWidth = img.width;
              this.imageHeight = img.height;
              const canvas = document.createElement('canvas')
              const ctx = canvas.getContext('2d')
              canvas.width = img.width
              canvas.height = img.height
              ctx.drawImage(img, 0, 0)

              this.layers.push({
                img,
                width: img.width,
                height: img.height,
                data: ctx.getImageData(0, 0, img.width, img.height).data
              });
            }
          } catch (e) {
            this.$notify({
            title:'Error',
            text:e.message,
            type:'error'
          })
          } finally {
            this.loading = false;
          }
        }
      }
    },
    showOriginalImage: {
      handler(showOriginal) {
        if(showOriginal) {
          this.shownImage = this.baseImage;
        } else {
          this.shownImage = this.editedImage;
        }
      }
    } 
  }
};
</script>
