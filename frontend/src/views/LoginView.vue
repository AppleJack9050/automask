<template>
  <div class="signin-form card border-0 shadow-sm rounded-4 p-4" style="max-width: 420px; margin: 0 auto;">
    <ul class="nav nav-pills nav-fill mb-4 bg-light rounded-3 p-1">
      <li class="nav-item">
          <a
          class="nav-link rounded-3 fw-medium"
          :class="{ active: !signUp }"
          href="#"
          @click.prevent="signUp = false"
        >
          Log In
        </a>
      </li>
      <li class="nav-item">
        <a
          class="nav-link rounded-3 fw-medium"
          :class="{ active: signUp }"
          href="#"
          @click.prevent="signUp = true"
        >
          Sign Up
        </a>
      </li>
    </ul>
    <div v-if="signUp">
      <h5 class="fw-semibold text-dark mb-1">Create an account</h5>
      <Form @submit="create">
        <div class="mb-3">
          <label class="form-label small fw-medium text-secondary">Username</label>
          <Field
            name="username"
            rules="required|alpha_num"
            v-model="username"
            type="text"
            placeholder="Username"
            as="input"
            class="form-control form-control-sm rounded-3"
          />
          <ErrorMessage name="username" class="text-danger small mt-1 d-block" />
        </div>

        <div class="mb-4">
          <label class="form-label small fw-medium text-secondary">Password</label>
          <Field
            name="password"
            rules="required|min:8|regex:^\S+$"
            v-model="password"
            type="password"
            placeholder="Min. 8 characters"
            as="input"
            class="form-control form-control-sm rounded-3"
          />
          <ErrorMessage name="password" class="text-danger small mt-1 d-block" />
        </div>

        <button type="submit" class="btn btn-primary w-100 rounded-3 fw-medium">
          Create Account
        </button>
      </Form>
    </div>
    <div v-else>
      <p class="text-body-secondary small mb-4">Log in to continue.</p>
      <Form @submit="loginUser">
        <div class="mb-3">
          <label class="form-label small fw-medium text-secondary">Username</label>
          <Field
            name="username"
            rules="required"
            v-model="username"
            type="text"
            placeholder="Username"
            as="input"
            class="form-control form-control-sm rounded-3"
          />
        </div>
        <div class="mb-4">
          <label class="form-label small fw-medium text-secondary">Password</label>
          <Field
            name="password"
            rules="required"
            v-model="password"
            type="password"
            placeholder="Password"
            as="input"
            class="form-control form-control-sm rounded-3"
          />
        </div>
        <button type="submit" class="btn btn-primary w-100 rounded-3 fw-medium">
          Log In
        </button>
      </Form>
    </div>

  </div>
</template>
<script>
import { useUserStore } from '@/stores/userroles';
import { mapActions } from 'pinia';
import { Form, Field, ErrorMessage } from 'vee-validate';
import { defineRule } from 'vee-validate';
import { required, min, alpha_num, regex } from '@vee-validate/rules';
defineRule('required', required);
defineRule('min', min);
defineRule('alpha_num', alpha_num);
defineRule('regex', regex);

export default {
  components: {
    Form,
    Field,
    ErrorMessage
  },
  data() {
    return {
      signUp:false,
      username:'',
      password:''
      };
  },
  methods: {
    ...mapActions(useUserStore, ['login', 'createUser']),
    async loginUser() {
      await this.login(this.username, this.password);
      this.username = '';
      this.password = '';

    },
    async create() {
      await this.createUser(this.username, this.password);
      this.username = '';
      this.password = '';
      this.$router.push('/view-files');
    }
  }
};
</script>
