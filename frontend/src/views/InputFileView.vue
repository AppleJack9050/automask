<script setup>
import InputFiles from '@/components/input-components/InputFiles.vue';
import fileService from "@/services/fileservice";
import { useNotification } from "@kyvg/vue3-notification"
import { useFileStore } from '@/stores/filestore';
const filesStore = useFileStore();
const { notify } = useNotification()

async function uploadFiles() {
  try {
    await fileService.putFiles(filesStore.getFiles);
    notify({
      type:'success',
      title: 'Success',
      text:'Uploaded Successfully'
    });
    filesStore.empty();
  } catch (e) {
    notify({
      text: e.message, 
      title: 'Error',
      type: 'error'
    });
  }
}
</script>
<template>
  <input-files></input-files>
  <button 
    @click="uploadFiles"
    class="btn btn-primary"
    style="width: auto; display: inline-block; height: 50px; margin: auto;"
  >
    Upload Files
  </button>
</template>
