<template>
  <div class="home container py-4">
  <div>
    <button
      class="btn btn-sm btn-outline-secondary"
      @click="loadFiles"
    >
      Refresh Files
    </button>
  </div>
    <div
      v-if="filesLoaded"
      v-for="(statusObj, index) in files"
      :key="index"
      class="mb-3"
    >
      <div
        v-for="(fileList, status) in statusObj"
        :key="status"
        class="card"
      >
        <div
          class="card-header d-flex justify-content-between align-items-center"
        >
          <h5 class="mb-0">{{ status }}</h5>
          <button
            class="btn btn-sm btn-outline-secondary"
            type="button"
            data-bs-toggle="collapse"
            :data-bs-target="'#collapse-' + index + '-' + status"
            aria-expanded="false"
            :aria-controls="'collapse-' + index + '-' + status"
          >
            ☰
          </button>
        </div>
        <div
          :id="'collapse-' + index + '-' + status"
          class="collapse show"
        >
          <div
            v-if="status === 'Unprocessed'"          
          >
            <button
              class="btn btn-secondary"
              @click="openPromptModal()"
            >
              Process All
            </button>
            <button
              v-if="filesSelected"
              class="btn btn-primary"
              @click="openPromptModal()"
            >
              Process Selected
            </button>
          </div>
          <div class="card-body">
            <ul class="list-group list-group-flush">
              <li
                v-for="file in fileList"
                :key="file"
                class="list-group-item d-flex justify-content-between align-items-center"
              >
                {{ file }}
                <button
                  v-if="status === 'Processed'"
                  class="btn btn-sm btn-secondary"
                  @click="editFile(file)"
                >
                  Edit
                </button>
                <button
                  v-if="status === 'Saved'"
                  class="btn btn-sm btn-secondary"
                  @click="downloadFile(file)"
                >
                  Download
                </button>
                <div  v-if="status === 'Unprocessed'">
                  <input
                    v-model="selectedFiles"
                    type="checkbox"
                    :value="file"
                  >
                  </input>
                </div>
                <div>
                  <input
                    v-model="selectedFilesForDownload"
                    type="checkbox"
                    :value="file"
                  >
                  </input>
                  </div>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    <div v-else>
      <loading></loading>
    </div>
    <prompt-modal
      @process="this.processFiles"
      ></prompt-modal>
  </div>
</template>

<script>
import Loading from "@/components/Loading.vue";
import fileService from "@/services/fileservice"
import 'bootstrap/dist/js/bootstrap.bundle.min';
import PromptModal from "@/components/PromptModal.vue";
import { Modal } from 'bootstrap';

export default {
  components:{
    Loading,
    PromptModal
  },
  data() {
    return {
      files: [],
      showModal: false,
      selectedFiles: [],
      selectedFilesForDownload: []
    }
  },
  computed: {
    filesLoaded() {
      return this.files.length !== 0;
    },
    filesSelected() {
      return this.selectedFiles.length > 0;
    }
  },
  methods: {
    async downloadFile(fileName) {
      try {
        const file = (await fileService.downloadImage(fileName)).data;
        const link = document.createElement("a");
        link.download = fileName;

        link.href = file.data;

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        this.$notify({
          title:'Success',
          text:'File Downlaoded Successfully',
          type:'success'
        })
      } catch (e) {
        this.$notify({
          title:'Error',
          text:e,
          type:'error'
        })
      }
    },
    downloadZip() {
      for (const fileName in this.selectedFilesForDownload) {
        // get the file and add it to the blob
        
      }
    },
    downloadTar() {

    },
    editFile(file) {
      this.$router.push({ name: 'Edit', params: { imageTitle: file } });
    },
    async processFiles(prompt, positive, highlight) {
      try {
        await fileService.processFiles(
          this.selectedFiles.length === 0 ?
            this.files
              .filter(f => f.hasOwnProperty('Unprocessed'))
              .map(f => f.Unprocessed)
              .flat() :
            this.selectedFiles,
          prompt,
          positive,
          highlight
        );
        this.closeModal();
        this.$notify({
          title:'Success',
          text:'Files Processed Successfully',
          type:'success'
        });
      } catch (error) {
        this.$notify({
          title:'Error',
          text:error,
          type:'error'
        });
      }
    },
    async loadFiles() {
      try {
        const response = await fileService.getFiles();
        this.files = response.data;
      } catch (e) {
        this.$notify({
          title:'Failed to get Files',
          text:e.message,
          type:'error'
        });
      }
    },
    openPromptModal() {
      const el = document.getElementById('promptModal');
      const modal = Modal.getOrCreateInstance(el);
      modal.show();
    },
    closeModal() {
      const el = document.getElementById('promptModal');
      const modal = Modal.getOrCreateInstance(el);
      modal.hide();
    }
  },
  created() {
    this.loadFiles();
  }
}
</script>
