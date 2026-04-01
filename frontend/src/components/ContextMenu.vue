<template>
  <div class="dropdown position-absolute" :id="`context_menu_${id}`">
    <ul class="dropdown-menu show">
      <li>
        <a class="dropdown-item" @click="selectObject">Select Object</a>
      </li>
      <li>
        <a class="dropdown-item" @click="removeObject">Remove Object from Image</a>
      </li>
      <li>
        <a class="dropdown-item" @click="save">Save Current Image</a>
      </li>
      <li>
        <a class="dropdown-item" @click="startTouchUp">Use Touch Up Tool</a>
      </li>
      <li>
        <a class="dropdown-item" @click="cancel">Cancel</a>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  props: {
    id: String,
    usingTouchUp: {
      type: Boolean,
      default: false
    }
  },
  methods: {
    selectObject() {
      this.$emit('select', this.id); 
    },
    removeObject() {
      this.$emit('remove', this.id);
    },
    save() {
      this.$emit('save');
    },
    startTouchUp() {
      this.$emit('start');
    },
    cancel() {
      this.$emit('cancel');
    }
  }
}
</script>
