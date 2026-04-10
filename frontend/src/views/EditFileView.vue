<template>
  <div class="container-fluid px-0 min-vh-100 d-flex flex-column">
    <div v-if="uploadOnly === 'true'">
      <div class="card border-0 shadow-sm rounded-4 p-3 w-100" style="max-width: 860px;">
        <view-file
          :baseImage="baseImage"
          :fileName="imageTitle"
        />
      </div>
    </div>
    <div v-else class="flex-grow-1 d-flex flex-column">
      <edit-file
        v-show="imageTitle"
        :baseImage="baseImage"
        :editedImage="editedImage"
        :masks="masks"
        :fileName="imageTitle"
        class="flex-grow-1"
        @saveImage="openSaveModal"
      />
      <save-modal
        :file="imageToSave"
        :fileName="imageTitle"
        :visible="showModal"
        @close="closeModal"
      />
    </div>
    <tool-tips 
      infoType="edit"
      :content="edit"
    />
    <tool-tips-button
      infoType="edit"
    />
  </div>
</template>

<script>
import EditFile from '@/components/EditFile.vue';
import SaveModal from '@/components/modals/SaveModal.vue';
import fileService from '@/services/fileservice';
import ViewFile from '@/components/ViewFile.vue';
import { Modal } from 'bootstrap';
import ToolTipsButton from '@/components/tooltips/ToolTipsButton.vue';
import ToolTips from '@/components/tooltips/ToolTips.vue';
import tooltips from '@/components/tooltips/tooltips'

export default {
  components: {
    EditFile,
    SaveModal,
    ViewFile,
    ToolTips,
    ToolTipsButton
  },
  props: {  
    imageTitle:{
      type: String
    },
    saved: {
      type: Boolean,
      required: false,
      default: false
    },
    uploadOnly: {
      type: Boolean,
      required: false,
      default: false
    }
  },
  data() {
    return {
      baseImage: null,
      editedImage: null,
      masks: [],
      imageToSave: null,
      showModal: false,
      edit:tooltips.edit
    }
  },
  computed:{
    isSaved() {
      return this.saved === 'true';
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
    },
    fetchAssetsForEdit() {
      try {
        const interval = setInterval(async () => {
          const assets = await fileService.showEditor(this.imageTitle, this.isSaved);
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
        });
          clearInterval(interval);
        }, 30000);
      } catch (e) {
        this.$notify({
          title:'Error',
          text:e.message,
          type:'error'
        })
      }
    },
    fetchFileForViewing() {
      try {
        const interval = setInterval(async () => {
          const assets = await fileService.showUploadFile(this.imageTitle);
          clearInterval(interval);
          clearTimeout(timeout);
          this.baseImage = assets.data.base_image;
        }, 5000);

        const timeout = setTimeout(() => {
          this.$notify({
          title:'Error',
          text:'Timed out waiting for assets',
          type:'error'
        });
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
  },
  async mounted() {
    if (this.imageTitle !== undefined) {
      this.uploadOnly === 'true' ?
        this.fetchFileForViewing() :
        this.fetchAssetsForEdit();
    }
  }
}
</script>
