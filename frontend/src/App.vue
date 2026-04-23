<template>
  <div id="app">
    <notifications />
    <Sidebar />
    <main class="main-content">
      <Suspense>
        <template #default>
          <RouterView />
        </template>
        <template #fallback>
          <div class="loading-screen">Loading...</div>
        </template>
      </Suspense>
    </main>
  </div>
</template>
<script>
import { RouterView } from 'vue-router'
import Sidebar from '@/components/Sidebar.vue'

export default {
  components: {
    RouterView,
    Sidebar
  },
  methods: {
    handleRefresh(event) {
      this.$notify({
        title:'Info',
        text:'Refreshing Will Log You Out',
        type:'warning',
        duration: 5000
      });
      event.preventDefault();
      event.returnValue = "";
    }
  },
  mounted() {
    window.addEventListener("beforeunload", this.handleRefresh);
  },
  beforeUnmount() {
    window.removeEventListener("beforeunload", this.handleRefresh);
  },
}
</script>

<style scoped>
#app {
  display: flex;
  height: 100vh;
}

.main-content {
  flex: 1;
  margin-left: 200px;
  padding: 1rem;
  background-color: #f4f4f9;
  overflow-y: auto;
}
</style>
