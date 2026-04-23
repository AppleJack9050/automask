<template>
  <div class="modal fade" id="promptModal" tabindex="-1" aria-labelledby="promptModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content border-0 shadow">
        <div class="modal-header bg-dark text-white">
          <h5 class="modal-title fw-semibold" id="promptModalLabel">
            Editing Prompt
          </h5>
          <button
            type="button"
            class="btn-close"
            data-bs-dismiss="modal"
            aria-label="Close"
          ></button>
        </div>
        <div class="modal-body p-4">
          <div class="alert alert-secondary d-flex gap-2 py-2 px-3 mb-4">
            <p class="mb-1 small fw-semibold">Example</p>
            <p>
              Prompts should be entered in lowercase and end with a full stop, at both the last prompt and between prompts.
              Prompts are also allowed contain multiple words
              <br/>
              Here is an example:          
              1. sky 2. the grey clouded sky.
            </p>
          </div>
          <div class="mb-3">
            <label for="pm-prompt" class="form-label fw-semibold small">
              Prompt
            </label>
            <input
              id="pm-prompt"
              type="text"
              class="form-control"
              v-model="prompt"
              placeholder="prompt"
            />
          </div>
          <div class="row g-3">
            <div class="col-6">
              <label for="pm-positive" class="form-label fw-semibold small">
                Action
              </label>
              <select id="pm-positive" class="form-select" v-model="positive">
                <option :value="true">Keep</option>
                <option :value="false">Remove</option>
              </select>
            </div>
            <div class="col-6">
              <label for="pm-highlight" class="form-label fw-semibold small">
                Display
              </label>
              <select id="pm-highlight" class="form-select" v-model="highlight">
                <option :value="true">Highlight</option>
                <option :value="false">No Highlight</option>
              </select>
            </div>
          </div>
        </div>
        <div class="modal-footer bg-light">
          <span v-if="status" class="text-muted small me-auto fst-italic">{{ status }}</span>
          <button type="button" class="btn btn-dark px-4" @click="process">
            Process
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  props:{
  },
  data() {
    return {
    prompt: '',
    positive: true,
    highlight: false
    }
  },
  methods: {
    process() {
      this.$emit(
        'process',
        this.prompt,
        this.positive,
        this.highlight
      );
    }
  }
}
</script>
