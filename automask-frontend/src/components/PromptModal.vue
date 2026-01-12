<template>
  <div class="modal fade" id="promptModal">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title">
            Enter The Prompt:
          </h5>
          <button
            type="button"
            class="btn-close"
            data-bs-dismiss="modal"
            aria-label="Close"
          >
          </button>
        </div>
        <div class="mb-3">
            <div>
              <p>
                Prompts should be entered in lowercase and end with a full stop, at both the last prompt and between prompts.
                Prompts are also allowed contain multiple words
                <br/>
                Here is an example: 

                object 1. more complex object 2.
              </p>
            </div>
            <form>
              <label class="form-label" text="Enter The Prompt For Segmentation:"></label>
              <input
                type="text"
                v-model="prompt"
              >
              </input>
              <br></br>
              <select
                v-model="positive"
              >
                <option :value=true>Keep</option>
                <option :value=false>Remove</option>
              </select>
              <select
                v-model="highlight"
              >
                <option :value=true>Highlight</option>
                <option :value=false>Don't Highlight</option>
              </select>
            </form>
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
