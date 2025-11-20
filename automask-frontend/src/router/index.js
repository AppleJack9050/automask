import { createRouter, createWebHashHistory } from 'vue-router'
import Home from '@/views/Home.vue'

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Home,
    },
    {
      path: '/input-files',
      name: 'Input',
      component: () => import('../views/InputFileView.vue')
    },
    {
      path: '/edit-file/:imageTitle',
      name: 'Edit',
      component: () => import('../views/EditFileView.vue'),
      props:true
    },
    {
      path: '/view-files',
      name: 'View Files',
      component: () => import('../views/FileStatus.vue')
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue'),
    },
  ],
})

export default router
