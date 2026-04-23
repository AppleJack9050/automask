<template>
  <div>
    <div
      class="svg-canvas-area d-flex align-items-center justify-content-center flex-grow-1 position-relative overflow-hidden"
      v-if="!loading"
    >
      <svg
        class="svg-wrapper"
        width="100%"
        height="100%"
        viewBox="0 0 1200 1200"
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

<style>
.svg-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}
</style>