import apiClient from './apiClient';

const basicHeaders = {
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${localStorage.getItem('token')}`  
}}
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
      formData.append('files', file);
    }

    return await apiClient.post('/files', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
    });
}
async function showEditor(fileName) {
    return await apiClient.get(
      `/show-file/${fileName}`,
      basicHeaders
    )
}
async function processFiles(files, prompt, positive, highlight) {
  return await apiClient.post(
    '/process-files',
    {
      files: files,
      prompt: prompt,
      positive: positive,
      highlight: highlight
    },
    {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
       }
    }
  );
}
async function saveImage(file, fileName, fileType) {
  const response = await apiClient.post(
    `/save-image/${fileName}`,
    {
      file: file,
      file_type: fileType,
    },
    {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${localStorage.getItem('token')}`
  }});
  return response;
}
async function downloadImage(fileName) {
  const response = await apiClient.get(
    `/download/${fileName}`,
    {
      headers: 
      {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }
  });
  return response;

}
async function downloadZip(files, name) {
  const response = await apiClient.post(
    `/download-zip`,
    {
      files: files,
      name: name,
    },
    {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
      },
      responseType: 'blob'
    }
);
  return response;
}
async function downloadTar(files, name) {
  const response = await apiClient.post(
    `/download-tar`,
    {
      files: files,
      name: name,
    },
    {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`
        },
      responseType: 'blob'
    });
    return response;
}
export default {
    getFiles,
    getFile,
    putFiles,
    showEditor,
    processFiles,
    saveImage,
    downloadImage,
    downloadZip,
    downloadTar
  }
