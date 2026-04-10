<template>
  <div class="dropdown position-absolute" :id="`context_menu_${id}`">
    <ul class="dropdown-menu show">
      <li>
        <a class="dropdown-item" @click="selectObject">Apply to Rest Of Image</a>
      </li>
      <li>
        <a class="dropdown-item" @click="removeObject">Apply to Highlighted Area Only</a>
      </li>
      <li>
        <a class="dropdown-item" @click="startTouchUp">Use Touch Up Tool</a>
      </li>
      <li>
        <a class="dropdown-item" @click="save">Save Current Image</a>
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
