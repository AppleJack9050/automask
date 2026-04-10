<template>
  <Teleport to="body">
    <div
      class="modal fade"
      :id="`${infoType}InfoModal`"
      data-bs-backdrop="static"
      tabindex="-1"
    >
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Share File</h5>
            <button
              type="button"
              class="btn-close"
              data-bs-dismiss="modal"
              aria-label="Close"
            >
            </button>
          </div>
          <div class="modal-body">
            <div
              class="mb-3 signin-form card border-0 shadow-sm rounded-4 p-4"
              style="max-width: 420px; margin: 0 auto;"
              v-html="content"
            >
            </div>
          </div>
          <div class="modal-footer">
            <button
              type="button"
              class="btn btn-secondary"
              data-bs-dismiss="modal"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>
<script>
export default {
  props: {
    content: {
      type: String,
      required: true
    },
    infoType: {
      type: String,
      required: true
    }
  },
}
</script>
