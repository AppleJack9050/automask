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
import MaskSelector from './editing-tools/maskSelector';
import TouchUp from './editing-tools/touchUp';
import MaskSetup from './editing-tools/maskSetup';

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
      basePixels: null,
      preTouchUpSnapshot: null,
      ctm: null,
      shownData: null,
      showCanvas: false,
      transparentBackground: true,
      maskSelector: null,
      touchUpTool: null
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
    handleRightClick(event) {
      event.preventDefault();
      this.contextX = event.clientX;
      this.contextY = event.clientY;
      this.contextId = this.hoveredMask;
    },
    async removeObject(id, undo = false) {
      this.shownImage = await this.maskSelector.removeObject(
        id,
        undo,
        this.shownImage,
        this.masks[id].mask,
        this.transparentBackground,
        this.basePixels
      );
      this.contextId = null;
    },
    async selectOnlyObject(id, undo = false) {
      this.shownImage = await this.maskSelector.selectOnlyObject(
        id,
        undo,
        this.shownImage,
        this.masks[id].mask,
        this.transparentBackground,
        this.basePixels
      );
      this.contextId = null;
    },
    cancel() {
      this.contextId = null;
    },
    save() {
      this.$emit('saveImage', this.shownImage);
    },
    removePixels(svgX, svgY) {
      this.shownData = this.touchUpTool.removePixels(
        svgX,
        svgY,
        this.shownData,
        this.transparentBackground
      );
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
      this.touchUpTool.scaleX = this.touchUpCanvas.width / svgWidth;
      this.touchUpTool.scaleY = this.touchUpCanvas.height / (svg.viewBox.baseVal.height || svg.clientHeight);
      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(this.ctm);

      this.shownData = this.touchUpTool.removePixels(
        svgPoint.x,
        svgPoint.y,
        this.shownData,
        this.transparentBackground
      );
    },
    updateTouchUpRadius(value) {
      this.touchUpRadius = value;
    },
    resetTouchUp(preSnapshot, postSnapshot, undo, restore = false) {
      this.shownImage = this.touchUpTool.resetTouchUp(
        this.touchUpCanvas,
        this.touchUpCtx,
        preSnapshot,
        postSnapshot,
        undo,
        restore
      );
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
          this.removeObject(lastAction.id, true, this.baseImage);
          break;
        case 'select':
          this.selectOnlyObject(lastAction.id, true, this.baseImage);
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
            this.removeObject(nextAction.id, true);
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
            this.imageWidth = image.width;
            this.imageHeight = image.height;

            this.touchUpCanvas.width = image.width;
            this.touchUpCanvas.height = image.height;
            this.touchUpCtx.drawImage(image, 0, 0)

            this.basePixels = this.touchUpCtx.getImageData(0, 0, image.width, image.height).data;
  
            this.layers = await MaskSetup.createSVGLayers(this.masks);
            this.maskSelector = new MaskSelector(this.fileName);
            this.touchUpTool = new TouchUp(this.touchUpRadius, this.scaleX, this.scaleY, this.fileName);
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
