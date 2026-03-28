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
      <form @submit.prevent="create">
        <input v-model="username" type="text" placeholder="Username" required />
        <input v-model="password" type="password" placeholder="Password" required />
        <button type="submit">Sign Up</button>
      </form>
    </div>
    <div v-else>
      <h2>Login</h2>
      <form @submit.prevent="loginUser">
        <input v-model="username" type="text" placeholder="Username" required />
        <input v-model="password" type="password" placeholder="Password" required />
        <button type="submit">Login</button>
      </form>
    </div>

  </div>
</template>

<script>
import { useUserStore } from '@/stores/userroles';
import { mapActions } from 'pinia';
export default {
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
// TODO username validation
</script>
