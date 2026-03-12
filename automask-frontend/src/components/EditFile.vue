<template>
  <div>
    <div
      class="svg-mask-viewer"
      v-if="!loading"
    >
      <edit-toolbar
        :touching-up="usingTouchUp"
        @restore="this.restoreImage"
        @undo="handleUndo"
        @redo="handleRedo"
        @update-touch-up-radius="updateTouchUpRadius"
        @transparentRemoval="toggleTransparent"
      ></edit-toolbar>
      <svg
        :width="this.imageWidth"
        :height="this.imageHeight"
        :viewBox="`0 0 ${this.imageWidth} ${this.imageHeight}`"
        @mousemove="highlight"
        @mouseleave="this.hoveredMask = null; this.editingPixels = false"
        @mousedown.left="startTouchUpEditing"
        @mouseup.left="this.editingPixels = false"
        @mouseup.right="this.usingTouchUp = false"
      >
        <image
          :href="`data:image/png;base64,${shownImage}`"
          width="100%"
          height="100%"
          v-show="!showCanvas"
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
        <foreignObject
          x="0"
          y="0"
          :width="imageWidth"
          :height="imageHeight"
          @contextmenu="handleRightClick($event, index)"
        >
          <canvas
            ref="touchUpCanvas"
            v-show="showCanvas"
            style="width: 100%; height: 100%;"
          ></canvas>
        </foreignObject>
        <circle
          :opacity="usingTouchUp ? 0.5 : 0"
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
        :usingTouchUp="usingTouchUp"
        @select="selectOnlyObject"
        @remove="removeObject"
        @save="save"
        @start="showTouchUp"
        @end="hideTouchUp"
        @cancel="cancel"
        :style="{ top: contextY + 'px', left: contextX + 'px' }"
        ></ContextMenu>
        <br />
    </div>
    <div v-else>
      <loading></loading>
    </div>
  </div>
</template>

<script>
import ContextMenu from './ContextMenu.vue';
import Loading from './Loading.vue';
import EditToolbar from './EditToolbar.vue';
import { useRedoStore } from '@/stores/redostore';
import { useUndoStore } from '@/stores/undostore';
import { mapState, mapActions } from 'pinia';

export default {
  components: {
    ContextMenu,
    Loading,
    EditToolbar
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
    },
    fileName:{
      type:String,
      required:true
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
      usingTouchUp: false,
      touchUpRadius:10,
      highlightX: null,
      highlightY: null,
      editingPixels: false,
      touchUpCtx: null,
      touchUpCanvas: null,
      imageHeight: null,
      imageWidth: null,
      showOriginalImage: false,
      basePixels: null,
      undoTouchUp: false,
      preTouchUpSnapshot: false,
      ctm: null,
      shownData: null,
      shownImageData: null,
      showCanvas: false,
      transparentBackground: true
    }
  },
  computed: {
    ...mapState(useUndoStore, ['getFileHistory', 'getLastFileState']),
    ...mapState(useRedoStore, ['getFileFuture', 'getFutureFileState'])
  },
  methods: {
    ...mapActions(useUndoStore, ['addFileHistory', 'updateAfterUndo']),
    ...mapActions(useRedoStore, ['addFileFuture', 'updateAfterRedo']),
    highlight(e) {
      if (this.usingTouchUp) {
        const svg = e.currentTarget;
        const point = svg.createSVGPoint();
        point.x = e.clientX;
        point.y = e.clientY;
        const svgPoint = point.matrixTransform(this.ctm);
        
        if (this.editingPixels) {
          this.handleTouchUp(svgPoint);
          return this.removePixels(svgPoint.x, svgPoint.y);
          
        }
        return this.handleTouchUp(svgPoint);
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
    async removeObject(id, undo = false) {

      if (undo) {
        this.addFileFuture(this.fileName, {id:[id], action:'remove', undo:true});
      } else {
        this.addFileHistory(this.fileName, {id:[id], action:'remove', undo:false})      
      }

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
          if (undo) {
            shownPixels[i] = this.basePixels[i];
            shownPixels[i + 1] = this.basePixels[i + 1];
            shownPixels[i + 2] = this.basePixels[i + 2];
            shownPixels[i + 3] = 255;
          } else {
            // here just follow the  undo logic but kind of just in reverse.
            this.transparentBackground ? 
              this.turnPixelsTransparent(shownPixels, i) :
              this.turnPixelsBlack(shownPixels, i);

            shownPixels[i + 3] = 0;
          }
        }
      }

      ctx.putImageData(shownImageData, 0, 0);
      this.shownImage = canvas.toDataURL('image/png').split(',')[1];
      this.contextId = null;
    },
    async selectOnlyObject(id, undo = false) {

      if (undo) {
        this.addFileFuture(this.fileName, {id:[id], action:'select', undo:true});
      } else {
        this.addFileHistory(this.fileName, {id:[id], action:'select', undo:false});
      }

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

        if (r == 0 && g == 0 && b == 0) {
          if (undo) {
            shownPixels[i] = this.basePixels[i]; 
            shownPixels[i + 1] = this.basePixels[i + 1];
            shownPixels[i + 2] = this.basePixels[i + 2];
            shownPixels[i + 3] = 255;
          } else {
            // here just follow the  undo logic but kind of just in reverse.
            this.transparentBackground ? 
              this.turnPixelsTransparent(shownPixels, i) :
              this.turnPixelsBlack(shownPixels, i);

              shownPixels[i + 3] = 0;
          }
        }
      }

      ctx.putImageData(shownImageData, 0, 0);
      this.shownImage = canvas.toDataURL('image/png').split(',')[1];
      this.contextId = null;
    },
    cancel() {
      this.contextId = null;
    },
    save() {
      this.$emit('saveImage', this.shownImage);
    },
    removePixels(svgX, svgY) {
      const imgX = Math.round(svgX * this.scaleX);
      const imgY = Math.round(svgY * this.scaleY);
      const radiusSq = this.touchUpRadius ** 2;
      const width = this.shownData.width;

      const startX = Math.max(0, imgX - this.touchUpRadius);
      const endX = Math.min(width, imgX + this.touchUpRadius);
      const startY = Math.max(0, imgY - this.touchUpRadius);
      const endY = Math.min(this.shownData.height, imgY + this.touchUpRadius);

      for (let y = startY; y < endY; y++) {
        const rowOffset = y * width;
        for (let x = startX; x < endX; x++) {
          if ((x - imgX) ** 2 + (y - imgY) ** 2 <= radiusSq) {
            const idx = rowOffset + x;
            if (this.pixelBuffer[idx] !== 0) {
              this.pixelBuffer[idx] = 0;
            }
          }
        }
      }

      this.touchUpCtx.putImageData(this.shownData, 0, 0);
    },
    handleTouchUp(svgPoint) {
      const imgX = Math.floor((svgPoint.x));
      const imgY = Math.floor((svgPoint.y));

      if (imgX < 0 || imgY < 0 ) {
        return;
      }

      this.highlightX = imgX;
      this.highlightY = imgY;
    },
    async showTouchUp() {
      this.contextId = null;
      this.usingTouchUp = true;
      this.showCanvas = false;

      const canvas = Array.isArray(this.$refs.touchUpCanvas) 
        ? this.$refs.touchUpCanvas[0] 
        : this.$refs.touchUpCanvas;

      const img = new Image();
      img.src = `data:image/png;base64,${this.shownImage}`;
      await img.decode();

      canvas.width = img.width;
      canvas.height = img.height;

      this.touchUpCanvas = canvas;
      this.touchUpCtx = canvas.getContext('2d', { willReadFrequently: true });

      this.touchUpCtx.drawImage(img, 0, 0);
      this.shownData = this.touchUpCtx.getImageData(0, 0, img.width, img.height);
      this.pixelBuffer = new Uint32Array(this.shownData.data.buffer);
      
      this.preTouchUpSnapshot = this.touchUpCtx.getImageData(0, 0, img.width, img.height);

      this.showCanvas = true;
      this.editingPixels = true;
    },
    hideTouchUp() {
      this.usingTouchUp = false;
      this.contextId = null;
      this.showCanvas = false;
      this.addFileHistory(
        this.fileName,
        {
          action:'touchUp',
          undo:false,
          preTouchUpSnapshot: this.preTouchUpSnapshot,
          postTouchUpSnapshot: this.touchUpCtx.getImageData(
            0,
            0,
            this.touchUpCanvas.width,
            this.touchUpCanvas.height
      )});
      this.shownImage = this.touchUpCanvas.toDataURL('image/png').split(',')[1];
      this.contextId = null;
    },
    startTouchUpEditing(e) {
      if (!this.usingTouchUp) return;
      this.editingPixels = true;

      const svg = e.currentTarget;
      this.ctm = svg.getScreenCTM().inverse();
      
      const svgWidth = svg.viewBox.baseVal.width || svg.clientWidth;
      this.scaleX = this.touchUpCanvas.width / svgWidth;
      this.scaleY = this.touchUpCanvas.height / (svg.viewBox.baseVal.height || svg.clientHeight);
      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(this.ctm);
      this.removePixels(svgPoint.x, svgPoint.y);
    },
    updateTouchUpRadius(value) {
      this.touchUpRadius = value;
    },
    resetTouchUp(preSnapshot, postSnapshot, undo, restore = false) {
      if (undo) {
        this.addFileFuture(
          this.fileName,
          {
            action:'touchUp',
            undo:true,
            preTouchUpSnapshot:preSnapshot,
            postTouchUpSnapshot:postSnapshot
          });
      } else {
        this.addFileHistory(
        this.fileName,
        {
          action:'touchUp',
          undo:false,
          preTouchUpSnapshot:preSnapshot,
          postTouchUpSnapshot:postSnapshot
        });
      }

      if (restore) {
        this.touchUpCtx.putImageData(postSnapshot, 0, 0);
        this.shownImage = this.touchUpCanvas.toDataURL('image/png').split(',')[1];
      } else {
        this.touchUpCtx.putImageData(preSnapshot, 0, 0);
        this.shownImage = this.touchUpCanvas.toDataURL('image/png').split(',')[1];
      }
    },
    restoreImage() {
      this.shownImage = this.baseImage;
    },
    handleUndo() {
      const lastAction = this.getLastFileState(this.fileName);
      
      if (!lastAction) {
        return;
      }
      switch(lastAction.action) {
        case 'remove':
          this.removeObject(lastAction.id, true);
          break;
        case 'select':
          this.selectOnlyObject(lastAction.id, true);
          break;
        case 'touchUp':
          this.resetTouchUp(lastAction.preTouchUpSnapshot, lastAction.postTouchUpSnapshot, true);
          break;
      };
      this.updateAfterUndo(this.fileName)
    },
    handleRedo() {
      const nextAction = this.getFutureFileState(this.fileName);

      if (!nextAction) {
        return;
      }

      switch(nextAction.action) {
        case 'remove':
          nextAction.undo ?
            this.removeObject(nextAction.id, false) :
            this.removeObject(nextAction.id, false);
          break;
        case 'select':
          nextAction.undo ?
            this.selectOnlyObject(nextAction.id, false) :
            this.selectOnlyObject(nextAction.id, true);
          break;
        case 'touchUp':
          nextAction.undo ?
            this.resetTouchUp(nextAction.preTouchUpSnapshot, nextAction.postTouchUpSnapshot, false, true) :
            this.resetTouchUp(nextAction.preTouchUpSnapshot, nextAction.postTouchUpSnapshot, true, true);
          break;
      };
      this.updateAfterRedo(this.fileName)
    },
    toggleTransparent(newValue) {
      this.transparentBackground = newValue;
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
            await image.decode();

            this.touchUpCanvas.width = image.width;
            this.touchUpCanvas.height = image.height;

            this.touchUpCtx.drawImage(image, 0, 0)

            this.basePixels = this.touchUpCtx.getImageData(0, 0, image.width, image.height).data;
  
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
    },
    usingTouchUp: {
      async handler(newVal, oldVal) {
        if (!newVal && oldVal) {
          this.hideTouchUp();
        } else if (newVal && !oldVal) {
          await this.showTouchUp();
        }
      }
    }
  }
};
</script>
