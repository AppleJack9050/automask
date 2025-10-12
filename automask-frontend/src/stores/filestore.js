import { defineStore } from 'pinia'

export const useFileStore = defineStore('files',  {
  state: () => ({
    files: []
  }),

  getters: {
    getFiles: (state) => state.files
  },

  actions: {
    addFile(files) {
      this.files.push(files)
    },
    removeFile(file) {
      const index = this.files.indexOf(file)
      if (index > -1) {
        this.files.splice(index, 1)
      } 
    }
  }
})
