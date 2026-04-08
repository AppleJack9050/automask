<template>
  <div class="modal fade" id="downloadModal">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title">
            Downlaod
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
          <form>
            <label class="form-label" text="Enter The Name For The File:"></label>
            <input
              type="text"
              v-model="name"
            >
            </input>
            <br></br>
            <select
              v-model="zip"
            >
              <option :value=true>Zip File</option>
              <option :value=false>Tar File</option>
            </select>
          </form>
        </div>
        <div class="modal-footer">
          <button
            type="button"
            @click="download"
            class="btn btn-primary"
          >
            Download
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
  <script>
  export default {
    data() {
      return {
      name: '',
      zip: true
      }
    },
  methods: {
    download() {
      this.$emit(
        'download',
        this.name,
        this.zip
      );
    }
  }
}
</script>
