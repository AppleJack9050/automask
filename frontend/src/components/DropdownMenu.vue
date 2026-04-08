<template>
  <div class="dropdown" ref="dropdown">
    <button class="btn btn-link p-0" @click="toggle">
      <i class="bi bi-three-dots-vertical" />
    </button>
    <ul class="dropdown-menu" :class="{ show: isOpen }">
      <li v-for="(option, index) in options" :key="index">
        <a class="dropdown-item" @click="handleEmit(option)">{{ option }}</a>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  props: {
    options: {
      type: Array,
      required: true
    },
    item: {
      type: String,
      required: true
    },
    status: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      isOpen: false
    };
  },
  methods: {
    toggle() {
      this.isOpen = !this.isOpen;
    },
    handleEmit(option) {
      this.$emit(
        'option-click',
        option,
        this.item,
        this.status
      );
    }
  }
}
</script>
