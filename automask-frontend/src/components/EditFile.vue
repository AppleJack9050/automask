<template>
  <div >
    <div
      class="svg-mask-viewer"
      v-if="!loading"
    >
      <svg
        :width="1200"
        :height="1200"
        viewBox="0 0 1200 1200"
        @mousemove="highlightMask" 
        @mouseleave="this.hoveredMask = null" 
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
            :width="mask.width || 1200"
            :height="mask.height || 1200"
            :opacity="hoveredMask === index ? 0.5 : 0"
            class="mask"
            @contextmenu="handleRightClick($event, index)"
          />
        </g>
      </svg>
      <ContextMenu
        v-if="contextId"
        :id="contextId"
        @select="selectOnlyObject"
        @remove="removeObject"
        @save="save"
        :style="{ top: contextY + 'px', left: contextX + 'px' }"    
        ></ContextMenu>
    </div>
    <div v-else class="center">
      Loading Image
      <i class="bi bi-arrow-clockwise spin" style="font-size: 4rem;"></i>
    </div>
  </div>
</template>

<script>
import ContextMenu from './ContextMenu.vue';

export default {
  components: {
    ContextMenu
  },
  props: {
    baseImage:{
      type:String,
      required:true
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
      maskColours: [
      'hue-rotate(0deg) brightness(1.2)',
      'hue-rotate(90deg) brightness(1.2)',
      'hue-rotate(180deg) brightness(1.2)',
      'hue-rotate(270deg) brightness(1.2)'
    ],
    loading:true,
    contextId: null,
    contextX: null,
    contextY: null,
    shownImage: null
    }
  },
  computed: {
  },
  methods: {
    highlightMask(e) {

      const svg = e.currentTarget;
      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());
      const masks = svg.querySelectorAll('.mask');

      if (masks.length === 0 ) {
        return
      }

      const w = parseFloat(masks[0].getAttribute('width'));
      const h = parseFloat(masks[0].getAttribute('height'));
      const x = parseFloat(masks[0].getAttribute('x'));
      const y = parseFloat(masks[0].getAttribute('y'));

      const imgX = Math.floor((svgPoint.x - x) * (this.layers[0].width / w));
      const imgY = Math.floor((svgPoint.y - y) * (this.layers[0].height / h));

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
        if (r > 128) {
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

      const shownData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const shownPixels = shownData.data;

      const maskCanvas = document.createElement('canvas');
      const maskCtx = maskCanvas.getContext('2d');
      maskCanvas.width = maskImg.width;
      maskCanvas.height = maskImg.height;
      maskCtx.drawImage(maskImg, 0, 0);
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;

      for (let i = 0; i < maskData.length; i += 4) {
        const r = maskData[i];
        const g = maskData[i + 1];
        const b = maskData[i + 2];
        const alpha = maskData[i + 3];

        if (r > 200 && g > 200 && b > 200 && alpha > 0) {
          shownPixels[i + 3] = 0;
        }
      }

      ctx.putImageData(shownData, 0, 0);
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

      const shownData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const shownPixels = shownData.data;

      const maskCanvas = document.createElement('canvas');
      const maskCtx = maskCanvas.getContext('2d');
      maskCanvas.width = maskImg.width;
      maskCanvas.height = maskImg.height;
      maskCtx.drawImage(maskImg, 0, 0);
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;

      for (let i = 0; i < maskData.length; i += 4) {
        const r = maskData[i];
        const g = maskData[i + 1];
        const b = maskData[i + 2];
        if (r < 255 && g < 255 && b < 255) {
          shownPixels[i + 3] = 0;
        }
      }

      ctx.putImageData(shownData, 0, 0);
      this.shownImage = canvas.toDataURL('image/png').split(',')[1];
      this.contextId = null;
    },
    save() {
      this.$emit('saveImage', this.shownImage);
    }
  },
  async created() {
    if (this.baseImage) {
      try {
        this.shownImage = this.baseImage;
        const image = new Image();
        image.src = `data:image/png;base64,${this.baseImage}`;
        for (const layer of this.masks) {
          console.log('a');
          const img = new Image();
          img.src = `data:image/png;base64,${await this.convertMaskTransparant(layer)}`;
          await image.decode();

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
          console.log('edit me ');
        }
      } catch (e) {
        // fail 
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>

<style>
image {
  pointer-events: visiblePainted;
}

.center {
  margin: 50%;
  width: 100%;
}

.spin {
  display: inline-block;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>