<template>
  <div>
    <edit-file
      v-show="imageTitle"
      :baseImage="baseImage"
      :editedImage="editedImage"
      :masks="masks"
      @saveImage="openSaveModal"
    />
    <save-modal
      :file="imageToSave"
      :fileName="imageTitle"
      :visible="showModal"
      @close="closeModal"
    >
    </save-modal>
  </div>
</template>

<script>
import EditFile from '@/components/EditFile.vue';
import SaveModal from '@/components/SaveModal.vue';
import fileService from '@/services/fileservice';
import { Modal } from 'bootstrap';

export default {
  components: {
    EditFile,
    SaveModal
  },
  props: {
    imageTitle:{
      type: String
    }
  },
  data() {
    return {
      baseImage: null,
      editedImage: null,
      masks: [],
      imageToSave: null,
      showModal: false
    }
  },
  methods: {
    openSaveModal(image) {
      this.imageToSave = image;
      this.showModal = true;
      const el = document.getElementById('saveModal');
      const modal = Modal.getOrCreateInstance(el);

      modal.show();

    },
    closeModal() {
      const el = document.getElementById('saveModal');
      const modal = Modal.getOrCreateInstance(el);
      modal.hide();
    }
  },
  async mounted() {
    if (this.imageTitle !== undefined) {
      try {
        const interval = setInterval(async () => {
          const assets = await fileService.showEditor(this.imageTitle);
          if (assets.data.masks.length > 0) {
            clearInterval(interval);
            clearTimeout(timeout);
            this.baseImage = assets.data.base_image;
            this.editedImage = assets.data.edited_image;
            this.masks = assets.data.masks;
          }
        }, 5000);

        const timeout = setTimeout(() => {
          this.$notify({
          title:'Error',
          text:'Timed out waiting for assets',
          type:'error'
        })
          clearInterval(interval);
        }, 30000);
      } catch (e) {
        this.$notify({
          title:'Error',
          text:e.message,
          type:'error'
        })
      }
    } 
  }
}
</script>