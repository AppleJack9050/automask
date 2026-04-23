<script setup>
import { computed } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { useUserStore } from '@/stores/userroles'
import UserModal from './modals/UserModal.vue';
import { Modal } from 'bootstrap';
const router = useRouter();
const userStore = useUserStore();

const routes = computed(() => 
  router.getRoutes().filter(route => {
    if (!route.name) return false;
    if (route.path === '/') return false;
    if (route.name === 'Edit') return false;
    if (route.meta.requiresAuth && !userStore.auth.loggedIn) return false;

    return true;
  })
);

function openUserModal() {
  const el = document.getElementById('userModal');
  const modal = Modal.getOrCreateInstance(el);
  modal.show();
}
</script>
<template>
  <div class="d-flex min-vh-100">
    <aside
      class="d-flex flex-column bg-dark text-white"
      style="position: fixed; top: 0; left: 0; width: 200px; height: 100vh;"
    >
      <div class="px-4 py-4 border-bottom border-secondary">
        <h5 class="fw-bold text-white mb-0 letter-spacing-1">Automask</h5>
      </div>
      <nav class="flex-grow-1 py-3">
        <router-link
          v-for="route in routes"
          :key="route.name"
          :to="route.path"
          class="d-flex align-items-center px-4 py-2 text-decoration-none text-secondary rounded-2 mx-2 mb-1"
          active-class="bg-primary text-white"
        >
          <span class="small fw-medium">{{ route.name }}</span>
        </router-link>
      </nav>
      <div class="px-4 py-3 border-top border-secondary">
        <div v-if="userStore.auth.loggedIn" class="d-flex align-items-center gap-2 flex-column">
          <span
            class="badge bg-success rounded-pill"
            style="width: 8px; height: 8px; padding: 0;"
          >
          </span>
          <span
            class="small text-secondary "
            @click="openUserModal"
            style="cursor: pointer;"
          >
            {{ userStore.auth.user }}
          </span>
          <br />
          <button
            class="btn btn-primary mt-2"
            @click="userStore.logout()"  
          >
            Log Out
          </button>
        </div>
        <div v-else>
          <span class="small text-secondary fst-italic">Not signed in</span>
        </div>
      </div>
    </aside>
    <user-modal :username="userStore.auth.user"></user-modal>
    <main class="flex-grow-1 bg-light">
      <slot />
    </main>
  </div>
</template>
