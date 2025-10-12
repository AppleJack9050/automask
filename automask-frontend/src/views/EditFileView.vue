<template>
  <div>
    <edit-file
      v-show="imageTitle"
      :baseImage="baseImage"
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
        const assets = await fileService.showEditor(this.imageTitle);
        this.baseImage = assets.data.image;
        this.masks = assets.data.masks;
      } catch (e) {
        console.log(e);
        // get notifications working 
      }
    } 
  }
}
</script>