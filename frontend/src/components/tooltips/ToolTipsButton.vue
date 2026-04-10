<template>
  <button
    ref="tooltipBtn"
    type="button"
    class="btn btn-primary rounded-circle d-flex align-items-center justify-content-center position-fixed bottom-0 end-0 m-3"
    style="width: 40px; height: 40px; font-style: italic; font-weight: bold;"
    @click="openModal"
  >
    i
  </button>
</template>
<script>
import { Modal } from 'bootstrap';
export default {
  props: {
    infoType: {
      type: String
    }
  },
  methods: {
    openModal() {
      const el = document.getElementById(`${this.infoType}InfoModal`);
      const modal = Modal.getOrCreateInstance(el);
      modal.show();
    }
  }
}
</script>
