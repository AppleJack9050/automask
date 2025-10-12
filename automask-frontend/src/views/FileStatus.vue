<script setup>
import fileService from "@/services/fileservice"
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
const router = useRouter()
const files = ref([])

onMounted(async () => {
  files.value = (await fileService.getFiles()).data
});

function editFile(file) {
  router.push({ name: 'Edit', params: { imageTitle: file } })
}

async function saveFile(fileName) {
  const file = await fileService.downloadImage(fileName);
  const link = document.createElement("a");

  link.download = fileName;

  link.href = file;

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

async function processFiles() {
  await fileService.processFiles();
}

</script>

<template>
  <div class="home container py-4">
    <div v-for="(statusObj, index) in files" :key="index" class="mb-3">
      <div v-for="(fileList, status) in statusObj" :key="status" class="card">
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
  </div>
</template>