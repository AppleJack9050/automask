<template>
  <div class="home container py-4">
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
                  @click="saveFile(file)"
                >
                  Download
                </button>
                <button
                  v-if="status === 'Unprocessed'"
                  class="btn btn-sm btn-secondary"
                  @click="processFiles()"
                >
                  Process
                </button>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    <div v-else>
      <loading></loading>
    </div>
  </div>
</template>

<script>
import Loading from "@/components/Loading.vue";
import fileService from "@/services/fileservice"
import 'bootstrap/dist/js/bootstrap.bundle.min.js';

export default {
  components:{
    Loading
  },
  data() {
    return {
      files: []
    }
  },
  computed: {
    filesLoaded() {
      return this.files.length !== 0;
    }
  },
  methods: {
    async saveFile(fileName) {
      try {
        const file = await fileService.downloadImage(fileName);
        const link = document.createElement("a");

        link.download = fileName;

        link.href = file;

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
          text:error,
          type:'error'
        }) 
      }
    },
    editFile(file) {
      this.$router.push({ name: 'Edit', params: { imageTitle: file } });
    },
    async processFiles() {
      try {
        await fileService.processFiles();
        this.$notify({
          title:'Success',
          text:'Files Processed Successfully',
          type:'success'
        });
      } catch (e) {
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
    }
  },
  created() {
    this.loadFiles();
  }
}


</script>