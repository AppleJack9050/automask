<template>
  <div>
    <div
      class="svg-mask-viewer"
      v-if="!loading"
    >
      <svg
        :width="this.imageWidth"
        :height="this.imageHeight"
        :viewBox="`0 0 ${this.imageWidth} ${this.imageHeight}`"
      >
        <image
          :href="`data:image/png;base64,${baseImage}`"
          width="100%"
          height="100%"
          v-show="!showCanvas"
        />
      </svg>
    </div>
    <div v-else>
      <loading></loading>
    </div>
  </div>
</template>
  
<script>
import Loading from './Loading.vue';

export default {
  components: {
    Loading,
  },
  props: {
    baseImage:{
      type:String,
      required:true
    },
    fileName:{
      type:String,
      required:true
    }
  },
  data() {
    return {
      loading:true,
    }
  },
  methods: {
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
          } catch (error) {
            this.$notify({
              title:'Error',
              text:error.message,
              type:'error'
            })
          } finally {
            this.loading = false;
          }
        }
      }
    }
  }
};
</script>
