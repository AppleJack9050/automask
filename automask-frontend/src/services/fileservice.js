import axios from 'axios'
const apiClient = axios.create({
    baseURL: 'http://localhost:8000'
});
const basicHeaders = {
  headers: {"Content-Type": "application/json"}
}
async function getFiles() {
    return await apiClient.get(
      '/files',
      basicHeaders
    );
}
async function getFile(id) {
    return await apiClient.get(
      `/files/${id}`,
     basicHeaders
    );
}
async function putFiles(files) {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    return await apiClient.post("/files", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
}
async function showEditor(fileName) {
    return await apiClient.get(
      `/show-file/${fileName}`,
      basicHeaders
    )
}
async function processFiles(files, prompt, positive) {
  return await apiClient.post(
    '/process-files/',
    {
      headers: basicHeaders
    },
    {
      files: files,
      prompt: prompt,
      positive: positive
    };
  )
}
async function saveImage(file, fileName, fileType) {
  try {
    const response = await apiClient.post(
      `/save-image/${fileName}`,
      {
        file: file,
        file_type: fileType,
      },
      {
        headers: { "Content-Type": "application/json" },
      }
    );
    return response;
  } catch (error) {
    this.$notify({
      title:'Error',
      text:error.message,
      type:'error'
    })
  }
}
async function downloadImage(fileName) {
  try {
    const response = await apiClient.post(
      `/save-image/${fileName}`,
      {
        headers: { "Content-Type": "application/json" },
      }
    );
    return response;
  } catch (error) {
    this.$notify({
      title:'Error',
      text:error.message,
      type:'error'
    })
  }
}

export default {
    getFiles,
    getFile,
    putFiles,
    showEditor,
    processFiles,
    saveImage,
    downloadImage
  }
