import { defineStore } from 'pinia'
import { useNotification } from '@kyvg/vue3-notification';
import userservice from '@/services/userservice';
const { notify } = useNotification();

export const useUserStore = defineStore('roles', {
  state: () => ({
    auth: {
      loggedIn: false,
      token: null,
      user: null
    }
  }),
  actions: {
    setAuth(token, user) {
      this.auth.loggedIn = true;
      this.auth.token = token;
      this.auth.user = user;
    },
    clearAuth() {
      this.auth.loggedIn = false;
      this.auth.token = null;
      this.auth.user = null;
    },
    async login(userName, password) {
      try {
        const response = await userservice.loginUser(userName, password);
        const { token, username } = response.data;

        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(username));

        this.setAuth(token, username);
      } catch (error) {
        notify({
          title:'Login Failed',
          text:'Username or Password Incorrect.',
          type:'error'
        });
      }
    },
    async createUser(email, password) {
      try {
        const response = await userservice.createUser(email, password);
        const { token, user } = response.data;

        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));

        this.setAuth(token, user);
      } catch (error) {
        notify({
          title:'Login Failed',
          text:error.message,
          type:'error'
        });
      }
    },
    async updateUsername(oldUsername, newUsername, password) {
      try {
        await userservice.updateUserName(oldUsername, password, newUsername);
      } catch (error) {
        notify({
          title:'Failed',
          text:error.message,
          type:'error'
        });
      }
    },
    async updatePassword(oldUsername, newUsername, password) {
      try {
        await userservice.updateUserName(oldUsername, password, newUsername);
      } catch (error) {
        notify({
          title:'Failed',
          text:error.message,
          type:'error'
        });
        this.logout();
      }
    },
    logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      this.clearAuth();
    }
  }
});
