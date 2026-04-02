<template>
  <div>
    <button
      class="btn btn-sm btn-outline-secondary"
      @click="restore"
    >Restore</button>
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
    <button
      class="btn btn-sm btn-outline-secondary"
      @click="undo"
    >Undo</button>
    <button
      class="btn btn-sm btn-outline-secondary"
      @click="redo"
    >Redo</button>
    <select v-model="transparent">
      <option :value="'transparent'">Transparent</option>
      <option :value="'black'">Black</option>
      <option :value="'restore'">Restore</option>
    </select>
  </div>
</template>

<script>
export default {
  props: {
    touchingUp: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      touchUpRadius: 10,
      showOriginalImage: false,
      transparent: 'transparent'
    };
  },
  methods: {
    undo() {
      this.$emit('undo');
    },
    redo() {
      this.$emit('redo');
    },
    restore() {
      this.$emit('restore');
    }
  },
  watch: {
    touchUpRadius: {
      immediate:true,
      handler(newValue) {
        this.$emit('updateTouchUpRadius', newValue);
      }
    },
    transparent: {
      immediate: true,
      handler(newValue) {
        this.$emit('transparentRemoval', newValue);
      }
    }
  }
};
</script>
