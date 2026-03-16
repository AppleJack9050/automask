import { defineStore } from 'pinia'

const MAXSIZE = 10

export const useRedoStore = defineStore('redo',  {
  state: () => ({
    files: {}
  }),

  getters: {
    getFileFuture: (state) => (file) => state.files[file],
    getFutureFileState: (state) => (file) => {
      const history = state.files[file]
      return history ? history[history.length - 1] : null
    }
  },
  actions: {
    addFileFuture(file, newState) {
      if (!this.files[file]) {
        this.files[file] = [newState];
        return;
      }

      if (this.files[file].length < MAXSIZE) {
            this.files[file].push(newState);
      } else {
        this.files[file].shift();
        this.files[file].push(newState);
      }
    },
    updateAfterRedo(file) {
      this.files[file].pop(file);
    },
  }
});
