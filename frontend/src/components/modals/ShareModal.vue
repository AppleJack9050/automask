<template>
  <div class="modal fade" id="shareModal">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title">Share File</h5>
          <button
            type="button"
            class="btn-close"
            data-bs-dismiss="modal"
            aria-label="Close"
          >
          </button>
        </div>
        <div class="modal-body">
          <div class="mb-3 signin-form card border-0 shadow-sm rounded-4 p-4" style="max-width: 420px; margin: 0 auto;">
              <Form @submit="shareFile">
                <label class="form-label small fw-medium text-secondary">Enter the File Recipients Username</label>
                <div class="mb-3">
                  <label class="form-label small fw-medium text-secondary">Username</label>
                  <Field
                    name="new_username"
                    :rules="{ required: true, alpha_num: true }"
                    v-model="username"
                    type="text"
                    placeholder="Username"
                    as="input"
                    class="form-control form-control-sm rounded-3"
                  />
                  <ErrorMessage name="new_username" class="text-danger small mt-1 d-block" />
              </div>
              <button type="submit" class="btn btn-primary w-100 rounded-3 fw-medium">
                  Share File
              </button>
            </Form>
          </div>
        </div>
        <div class="modal-footer">
          <button
            type="button"
            class="btn btn-secondary"
            data-bs-dismiss="modal"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import fileservice from '@/services/fileservice';
import { Modal } from 'bootstrap';
import { Form, Field, ErrorMessage } from 'vee-validate';
import { defineRule } from 'vee-validate';
import { required, alpha_num } from '@vee-validate/rules';

defineRule('required', required);
defineRule('alpha_num', alpha_num);

export default {
  components: {
    Form,
    Field,
    ErrorMessage
  },
  props: {
    file: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      username: null
    };
  },
  methods: {
    async shareFile() {
      try {
        await fileservice.shareFile(this.username, this.file);
        this.$notify({
            title:'Success',
            text:'Updated',
            type:'success'
        });
      } catch (error) {
        this.$notify({
          title:'Failed',
          text:error.message,
          type:'error'
        });
      } finally {
        this.username = null;
        this.closeModal();
      }
    },
    closeModal() {
      const el = document.getElementById('shareModal');
      const modal = Modal.getOrCreateInstance(el);
     modal.hide();
    }
  }
}
</script>
