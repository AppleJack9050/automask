<template>
  <div class="drag-and-drop-input" @dragover.prevent @drop="handleDrop">
    <input
      type="file"
      multiple
      id="file-input"
      class="hidden-input"
      accept=".zip, .tar, .tar.gz, .gz, .tgz, application/gzip, application/x-tar, .png, .jpg, .jpeg, .bmp, .tiff, .tif"
      @change="onChange"
    />
    <label for="file-input" class="file-label">Drop or Click to input files</label>
  </div>
</template>

<script setup>
import { useFileStore } from '@/stores/filestore';
const filesStore = useFileStore();

function onChange(event) {
  filesStore.addFile(...event.target.files);
}

function handleDrop(event) {
  filesStore.addFile(...event.dataTransfer.files);
}
</script>

<style>
.drag-and-drop-input {
  padding: 2rem;
  border: 2px dashed #ccc;
  text-align: center;
  width: 400px;
  height: auto;
  margin: auto;
}
.hidden-input {
  display: none;
}
.file-label {
  cursor: pointer;
  font-size: 1.2rem;
}
</style>
