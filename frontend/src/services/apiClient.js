import axios from 'axios'
import { useUserStore } from '@/stores/userroles';
import router from '@/router';

const apiClient = axios.create({
    baseURL: '/api'
});
apiClient.interceptors.response.use(
  response => response,
  error => {
      if (error.response?.status === 401) {
          const userStore = useUserStore()
          userStore.clearAuth()
          router.push('/login')
      }
      return Promise.reject(error)
  }
)

export default apiClient
