<template>
  <div class="modal fade" id="saveModal">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title">Save</h5>
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
            <label class="form-label">Select File Type</label>
            <select
              v-model="selectedFileType"
              class="form-select"
            >
              <option
                v-for="extension in fileOptions"
                :key="extension"
                :value="extension"
              >
                {{ extension }}
              </option>
            </select>
          </form>
        </div>

        <div class="modal-footer">
          <button
            type="button"
            class="btn btn-secondary"
            data-bs-dismiss="modal"
          >
            Close
          </button>
          <button
            type="button"
            @click="save"
            class="btn btn-primary"
            :disabled="extensionNull"
          >
            Save Image
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import fileservice from '@/services/fileservice';

export default {
  props: {
    file: {
      type: String,
      required: true
    },
    fileName: {
      type: String,
      required: true
    },
    visible: {
      type:Boolean,
      default:false
    }
  },
  data() {
    return {
      fileOptions: ['.jpg', '.jpeg', '.png', '.tiff', '.tif'],
      selectedFileType: null
    }
  },
  computed: {
    extensionNull() {
      return this.selectedFileType === null;
    }
  },
  methods: {
    downloadToUser() {
      const link = document.createElement("a");

      link.download = `${this.fileName}${this.selectedFileType}`;

      link.href = `data:image/png;base64,${this.file}`;

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    },
    async saveToBackend() {
      fileservice.saveImage(this.file, this.fileName, this.selectedFileType);
    },
    async save() {
      if (!this.extensionEntered) {
        await this.saveToBackend(this.file);
        this.downloadToUser(this.file);
      }
    }
  }
}

</script>
