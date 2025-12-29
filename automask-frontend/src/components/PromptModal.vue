<template>
  <div class="modal fade" :id="id">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title">
            Enter The Prompt
          </h5>
          <button>
            type="button"
            class="btn-close"
            data-bs-dismiss="modal"
            aria-label="Close"
          </button>
        </div>
      ``<div class="mb3">
          <form>
            <label class="form-label" text="Enter The Prompt For Segmentation:"/>
            <input
              type="text"
              v-model="prompt"
            >
            </input>
            <select
              v-model="positive"
            >
              <option :key="Positive" :value=true></option>
              <option :key="Negative" :value=false></option>
            </select>
          <form>
        </div>
        <div class="modal-footer">
          <button
            type="button"
            @click="process"
            class="btn btn-primary"
          >
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
  },,
  data() {
    return {
    prompt: '',
    positive: true
    }
  },
  methods: {
    process() {
      this.emit('process', this.prompt, this.positive);
    }
  }
}
</script>
