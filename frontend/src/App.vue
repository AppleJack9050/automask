<script setup>
import { RouterView } from 'vue-router'
import Sidebar from '@/components/Sidebar.vue'
</script>

<template>
  <div id="app">
    <notifications />
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
