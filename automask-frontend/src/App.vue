<script setup>
import { RouterLink, RouterView, useRouter, useRoute } from 'vue-router'
import Sidebar from '@/components/Sidebar.vue'
const router = useRouter();
const currentRoute = useRoute()

const routes = router.getRoutes();
</script>

<template>
  <div id="app">
    <Sidebar />
    <Suspense>
      <template #default>
        <RouterView />
      </template>
      <template #fallback>
        <div class="loading-screen">Loading...</div>
      </template>
    </Suspense>
  </div>
</template>

<style scoped>
#app {
  display: flex;
  height: 100vh;
}

.main-content {
  flex: 1;
  padding: 1rem;
  background-color: #f4f4f9;
  overflow-y: auto;
}
</style>
