import { defineStore } from 'pinia'

const MAXSIZE = 10

export const useUndoStore = defineStore('undo',  {
  state: () => ({
    files: {}
  }),
  getters: {
    getFileHistory: (state) => (file) => state.files[file],
    getLastFileState: (state) => (file) => {
      const history = state.files[file]
      return history ? history[history.length - 1] : null
    }
  },
  actions: {
    addFileHistory(file, newState) {
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
    updateAfterUndo(file) {
      this.files[file].pop(file);
    },
  }
});
