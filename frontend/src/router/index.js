import { createRouter, createWebHashHistory,} from 'vue-router'
import { useUserStore } from '@/stores/userroles';

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'Login',
      component:  () => import('../views/LoginView.vue'),
    },
    {
      path: '/input-files',
      name: 'Input',
      component: () => import('../views/InputFileView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/edit-file/:imageTitle',
      name: 'Edit',
      component: () => import('../views/EditFileView.vue'),
      meta: { requiresAuth: true },
      props:true
    },
    {
      path: '/view-files',
      name: 'View Files',
      component: () => import('../views/FileStatus.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue'),
    },
    {
      path: '/login',
      name: 'Login',
      component: () => import('../views/LoginView.vue')
    },
  ],
});

router.beforeEach((to, _, next) => {
  const roles = useUserStore();

  const loggedIn = roles.auth.loggedIn;

  if (to.matched.some(record => record.meta.requiresAuth) && !loggedIn) {
    next('/login');
  } else {
    next();
  }
});

export default router
