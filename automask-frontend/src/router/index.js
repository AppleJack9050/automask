import { createRouter, createWebHistory } from 'vue-router'
import InputFileView from '@/views/InputFileView.vue'
import EditFileView from '@/views/EditFileView.vue'
import FileStatus from '@/views/FileStatus.vue'
import Home from '@/views/Home.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Home,
    },
    {
      path: '/input-files',
      name: 'Input',
      component: () => InputFileView
    },
    {
      path: '/edit-file/:imageTitle',
      name: 'Edit',
      component: () => EditFileView,
      props:true
    },
    {
      path: '/view-files',
      name: 'View Files',
      component: () => FileStatus
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('../views/AboutView.vue'),
    },
  ],
})

export default router
