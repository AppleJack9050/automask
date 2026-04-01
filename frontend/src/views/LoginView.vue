<template>
  <div class="signin-form">
    <ul class="nav nav-pills nav-fill mb-4">
      <li class="nav-item">
        <a class="nav-link" :class="{ active: !signUp }" href="#" @click.prevent="signUp = false">
          Log In
        </a>
      </li>
      <li class="nav-item">
        <a class="nav-link" :class="{ active: signUp }" href="#" @click.prevent="signUp = true">
          Sign Up
        </a>
      </li>
    </ul>
    <div v-if="signUp">
      <h2>Sign Up</h2>
      <Form @submit="create">
        <Field name="username" rules="required|alpha_num" v-model="username" type="text" placeholder="Username" as="input" />
        <ErrorMessage name="username" />
        <Field name="password" rules="required|min:8|regex:^\S+$" v-model="password" type="password" placeholder="Password" as="input" />
        <ErrorMessage name="password" />
        <button type="submit">Sign Up</button>
      </Form>
    </div>
    <div v-else>
      <h2>Login</h2>
      <Form @submit="loginUser">
        <Field name="username" rules="required|alpha_num" v-model="username" type="text" placeholder="Username" as="input" />
        <ErrorMessage name="username" />
        <Field name="password" rules="required" v-model="password" type="password" placeholder="Password" as="input" />
        <ErrorMessage name="password" />
        <button type="submit">Login</button>
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
