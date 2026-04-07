<template>
  <div class="home container-lg py-5">
    <div class="d-flex justify-content-between align-items-center mb-4">
      <button
        class="btn btn-outline-secondary btn-sm d-flex align-items-center gap-2"
        @click="loadFiles"
      >
        <span>↻</span> Refresh
      </button>
    </div>
    <div v-if="filesLoaded">
      <div
        v-for="(statusObj, index) in files"
        :key="index"
        class="mb-4"
      >
        <div
          v-for="(fileList, status) in statusObj"
          :key="status"
          class="card border-0 shadow-sm rounded-3"
        >
          <div class="card-header bg-white border-bottom d-flex justify-content-between align-items-center px-4 py-3 rounded-top-3">
            <div class="d-flex align-items-center gap-2">
              <span
                class="badge rounded-pill"
                :class="{
                  'bg-warning text-dark': status === 'Unprocessed',
                  'bg-success': status === 'Saved',
                  'bg-secondary': status !== 'Unprocessed' && status !== 'Saved'
                }"
              >
                {{ fileList.length }}
              </span>
              <h6 class="mb-0 fw-semibold">{{ status }}</h6>
            </div>
            <button
              class="btn btn-sm btn-light border"
              type="button"
              data-bs-toggle="collapse"
              :data-bs-target="'#collapse-' + index + '-' + status"
              aria-expanded="true"
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
              v-if="status === 'Unprocessed' || status === 'Saved'"
              class="px-4 py-2 bg-light border-bottom d-flex flex-wrap gap-2 align-items-center"
            >
              <button
                class="btn btn-sm btn-outline-primary"
                @click="openPromptModal(status === 'Saved')"
              >
                Process All
              </button>
              <button
                v-if="filesSelectedUpload || filesSelectedSaved"
                class="btn btn-sm btn-primary"
                @click="openPromptModal()"
              >
                Process Selected
              </button>
              <template v-if="status === 'Saved'">
                <div class="vr mx-1"></div>
                <button
                  class="btn btn-sm btn-outline-success"
                  @click="openDownloadModal()"
                >
                  Download All
                </button>
                <button
                  v-if="filesSelectedSaved"
                  class="btn btn-sm btn-success"
                  @click="openDownloadModal()"
                >
                  Download Selected
                </button>
              </template>
            </div>

            <!-- File List -->
            <ul class="list-group list-group-flush">
              <li
                v-for="file in fileList"
                :key="file"
                class="list-group-item list-group-item-action d-flex justify-content-between align-items-center px-4 py-3"
              >
                <span class="text-truncate me-3 text-body-secondary small fw-medium">
                  {{ file }}
                </span>

                <div class="d-flex align-items-center gap-3 flex-shrink-0">
                  <div v-if="status === 'Unprocessed'">
                    <input
                      v-model="selectedFiles"
                      type="checkbox"
                      class="form-check-input"
                      :value="file"
                    />
                  </div>
                  <div v-if="status === 'Saved'">
                    <input
                      v-model="selectedFilesSaved"
                      type="checkbox"
                      class="form-check-input"
                      :value="file"
                    />
                  </div>
                  <dropdown-menu
                    :options="options"
                    :item="file"
                    :status="status"
                    @option-click="handleAction"
                  />
                </div>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    <div v-else class="d-flex justify-content-center align-items-center py-5">
      <loading />
    </div>
    <prompt-modal @process="processFiles" :saved="saved"/>
    <download-modal @download="handleDownload" :saved="saved"/>
  </div>
</template>

<script>
import Loading from "@/components/Loading.vue";
import fileService from "@/services/fileservice"
import 'bootstrap/dist/js/bootstrap.bundle.min';
import PromptModal from "@/components/PromptModal.vue";
import { Modal } from 'bootstrap';
import DownloadModal from "@/components/DownloadModal.vue";
import DropdownMenu from "@/components/DropdownMenu.vue";

export default {
  components:{
    Loading,
    PromptModal,
    DownloadModal,
    DropdownMenu
  },
  data() {
    return {
      files: [],
      showProcessModal: false,
      saved: false,
      selectedFiles: [],
      selectedFilesSaved: [],
      options: ["View", "Delete"]
    }
  },
  computed: {
    filesLoaded() {
      return this.files.length !== 0;
    },
    filesSelectedUpload() {
      return this.selectedFiles.length > 0;
    },
    filesSelectedSaved() {
      return this.selectedFilesSaved.length > 0;
    }
  },
  methods: {
    async handleDownload(fileName, zip) {
      return zip ?
        this.downloadZip(fileName) :
        this.downloadTar(fileName);
    },
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
    async downloadZip(fileName) {
      try {
        const zip = (
          await fileService.downloadZip(this.selectedFilesSaved, fileName)
        );
        const url = URL.createObjectURL(zip.data);

        const link = document.createElement("a");
          link.download = `${fileName}.zip`;

          link.href = url;

          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
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
    async downloadTar(fileName) {
      try {
        const tar = (
          await fileService.downloadTar(this.selectedFilesSaved, fileName)
        );

        const url = URL.createObjectURL(tar.data);
        const link = document.createElement("a");
          link.download = `${fileName}.tar.gz`;
          link.href = url;

          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
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
    editFile(file, saved) {
      this.$router.push({ name: 'Edit', params: { imageTitle:file, saved:saved, uploadOnly: false } });
    },
    async processFiles(prompt, positive, highlight) {
      try {
        if (this.selectedFilesSaved) {
          await fileService.processFiles(
          this.selectedFiles.length === 0 ?
            this.files
              .filter(f => f.hasOwnProperty('Saved'))
              .map(f => f.Saved)
              .flat() :
            this.selectedFiles,
            prompt,
            positive,
            highlight,
            true
          );          
        } 

        if (this.filesSelectedUpload) {
          await fileService.processFiles(
          this.selectedFiles.length === 0 ?
            this.files
              .filter(f => f.hasOwnProperty('Unprocessed'))
              .map(f => f.Unprocessed)
              .flat() :
            this.selectedFiles,
          prompt,
          positive,
          highlight,
          false
        );
        }

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
    },
    openDownloadModal() {
      this.saved = true;
      const el = document.getElementById('downloadModal');
      const modal = Modal.getOrCreateInstance(el);
      modal.show();
    },
    closeDownloadModal() {
      const el = document.getElementById('downloadModal');
      const modal = Modal.getOrCreateInstance(el);
      modal.hide();
    },
    deleteFile(file, section) {
    },
    viewFile(file) {
      this.$router.push({ name: 'Edit', params: { imageTitle:file, saved: false, uploadOnly: true } });
    },
    handleAction(action, item, section) {
      switch (action) {
        case 'View':
          return section === 'Unprocessed' ?
            this.viewFile(
              item
            ) :
            this.editFile(
              item,
              section === 'Saved'
            );
        case 'Delete':
          this.deleteFile(item, section);
      }
    }
  },
  created() {
    this.loadFiles();
  }
}
</script>
