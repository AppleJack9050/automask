<template>
  <div class="modal fade" id="userModal">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title">Manage Profile</h5>
          <button
            type="button"
            class="btn-close"
            data-bs-dismiss="modal"
            aria-label="Close"
          >
          </button>
        </div>
        <div class="modal-body">
          <div class="mb-3 signin-form card border-0 shadow-sm rounded-4 p-4" style="max-width: 500px; margin: 0 auto;">
            <ul class="nav nav-tabs nav-fill flex-nowrap" id="tabs" role="tablist">
              <li class="nav-item" role="presentation">
                <button
                  class="nav-link active"
                  id="username-tab"
                  data-bs-toggle="tab"
                  data-bs-target="#username"
                  type="button"
                  role="tab"
                  aria-controls="username-tab"
                  aria-selected="true"
                  >
                  Update Username
                </button>
              </li>
              <li class="nav-item" role="presentation">
                <button
                  class="nav-link"
                  id="password-tab"
                  data-bs-toggle="tab"
                  data-bs-target="#password"
                  type="button"
                  role="tab"
                  aria-controls="password-tab"
                  aria-selected="false"
                  >
                  Update Password
                </button>
              </li>
              <li class="nav-item" role="presentation">
                <button
                  class="nav-link"
                  id="delete-tab"
                  data-bs-toggle="tab"
                  data-bs-target="#delete"
                  type="button"
                  role="tab"
                  aria-controls="delete-tab"
                  aria-selected="false"
                  >
                  Delete User
                </button>
              </li>
            </ul>
            <div class="tab-content">
              <div
                id="delete"
                class="tab-pane"
                role="tabpanel"
                aria-labelledby="delete-tab"
              >
                <Form @submit="deleteAccount">
                  <label class="form-label small fw-medium text-secondary">Enter your Details to Confirm</label>
                  <div class="mb-3">
                    <label class="form-label small fw-medium text-secondary">Username</label>
                    <Field
                      name="new_username"
                      :rules="{ required: true, alpha_num: true }"
                      v-model="newUsername"
                      type="text"
                      placeholder="Username"
                      as="input"
                      class="form-control form-control-sm rounded-3"
                    />
                    <ErrorMessage name="new_username" class="text-danger small mt-1 d-block" />
                  </div>
                  <div class="mb-4">
                    <label class="form-label small fw-medium text-secondary">Password</label>
                    <Field
                      name="password"
                      :rules="{ required: true }"
                      v-model="password"
                      type="password"
                      placeholder="Enter your Password"
                      as="input"
                      class="form-control form-control-sm rounded-3"
                    />
                    <ErrorMessage name="password" class="text-danger small mt-1 d-block" />
                  </div>
                  <button type="submit" class="btn btn-danger w-100 rounded-3 fw-medium">
                    Delete Account
                  </button>
                </Form>
              </div>
              <div
                id="username"
                class="tab-pane show active"
              >
                <Form @submit="updateUsername">
                  <div class="mb-3">
                    <label class="form-label small fw-medium text-secondary">Username</label>
                    <Field
                      name="new_username"
                      :rules="{ required: true, alpha_num: true }"
                      v-model="newUsername"
                      type="text"
                      placeholder="Username"
                      as="input"
                      class="form-control form-control-sm rounded-3"
                    />
                    <ErrorMessage name="new_username" class="text-danger small mt-1 d-block" />
                  </div>
                  <div class="mb-4">
                    <label class="form-label small fw-medium text-secondary">Password</label>
                    <Field
                      name="password"
                      :rules="{ required: true }"
                      v-model="password"
                      type="password"
                      placeholder="Enter your Password"
                      as="input"
                      class="form-control form-control-sm rounded-3"
                    />
                    <ErrorMessage name="password" class="text-danger small mt-1 d-block" />
                  </div>
                  <button type="submit" class="btn btn-primary w-100 rounded-3 fw-medium">
                    Update Username
                  </button>
                </Form>
              </div>
              <div
                id="password"
                class="tab-pane"
                role="tabpanel"
                aria-labelledby="password-tab"
              >
                <Form @submit="updatePassword">
                  <div class="mb-4">
                    <label class="form-label small fw-medium text-secondary">Enter Your Old Password</label>
                      <Field
                        name="old_password"
                        :rules="{ required: true }"
                        v-model="password"
                        type="password"
                        placeholder="Enter your Password"
                        as="input"
                        class="form-control form-control-sm rounded-3"
                      />
                      <ErrorMessage name="old_password" class="text-danger small mt-1 d-block" />
                  </div>
                  <div class="mb-3">
                    <label class="form-label small fw-medium text-secondary">Enter your New Password</label>
                    <Field
                      name="new_password"
                    :rules="{ required: true, min: 8, regex: /^\S+$/ }"
                      v-model="newPassword"
                      type="password"
                      placeholder="Enter your New Password"
                      as="input"
                      class="form-control form-control-sm rounded-3"
                    />
                    <ErrorMessage name="new_password" class="text-danger small mt-1 d-block" />
                  </div>
                  <div class="mb-4">
                    <label class="form-label small fw-medium text-secondary">Confirm New Password</label>
                    <Field
                      name="new_password_repeat"
                      :rules="{ required: true, min: 8, regex: /^\S+$/ }"
                      v-model="newPasswordRepeat"
                      type="password"
                      placeholder="Enter your Password"
                      as="input"
                      class="form-control form-control-sm rounded-3"
                    />
                    <ErrorMessage name="new_password_repeat" class="text-danger small mt-1 d-block" />
                  </div>
                  <button type="submit" class="btn btn-primary w-100 rounded-3 fw-medium">
                    Update Password
                  </button>
                </Form>
              </div>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button
            type="button"
            class="btn btn-secondary"
            data-bs-dismiss="modal"
            :disabled="!passwordsMatch"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import userservice from '@/services/userservice';
import { useUserStore } from '@/stores/userroles';
import { mapActions } from 'pinia';
import { Modal } from 'bootstrap';
import { Form, Field, ErrorMessage } from 'vee-validate';
import { defineRule } from 'vee-validate';
import { required, min, alpha_num } from '@vee-validate/rules';

defineRule('required', required);
defineRule('min', min);
defineRule('alpha_num', alpha_num);

export default {
  components: {
    Form,
    Field,
    ErrorMessage
  },
  props: {
      username: {
        type: String,
        required: true
      }
    },
  data() {
    return {
      newUsername: null,
      password: null,
      newPassword: null,
      newPasswordRepeat: null,
      usernameForm: true,
      deleteForm: false
    };
  },
  computed: {
    passwordsMatch() {
      return this.newPassword !== null && this.newPassword === this.newPasswordRepeat;
    }
  },
  methods: {
    ...mapActions(useUserStore, ['setAuth', 'logout']),
    async updatePassword() {
      try {
        const response = await userservice.updateUserPassword(this.username, this.password, this.newPassword);
        const { token, username } = response.data;
        this.setAuth(token, username);
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
        this.closeModal();
        this.newPassword = null;
        this.password = null;
        this.newPasswordRepeat = null;
      }
    },
    async updateUsername() {
      try {
        const response = await userservice.updateUserName(this.username, this.newUsername, this.password);
        const { token, username } = response.data;
        this.setAuth(token, username);
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
        this.newUsername = null;
        this.password = null;
        this.closeModal();
      }
    },
    async deleteAccount() {
      try {
        await userservice.deleteUser(this.username, this.password);
        this.logout();
        this.$notify({
            title:'Success',
            text:'See you',
            type:'success'
          });
        this.closeModal();
        this.newUsername = null;
        this.password = null;
      } catch (error) {
        this.$notify({
          title:'Failed',
          text:error.message,
          type:'error'
        });
      } 
    },
    closeModal() {
      const el = document.getElementById('userModal');
      const modal = Modal.getOrCreateInstance(el);
     modal.hide();
    }
  }
}
</script>
