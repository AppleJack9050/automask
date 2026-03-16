<script setup>
import { RouterLink, useRouter } from 'vue-router'
const router = useRouter();

const routes = router.getRoutes();
</script>

<template>
  <div>
    <aside>
      <div>
        <h2>Interactive Mask Selection</h2>
      </div>
      <nav class="sidebar-nav">
          <router-link
            v-for="route in routes"
            :key="route.name"
            :to="route.path"
            class="nav-item"
            active-class="active"
          >
            <span class="label">{{ route.name }}</span>
          </router-link>
      </nav>
    </aside>
    <main class="main-content">
    </main>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  height: 100vh;
  font-family: 'Inter', sans-serif;
}

/* Sidebar styling */
.sidebar {
  width: 200px;
  background-color: #ffffff;
  border-right: 1px solid #ddd;
  display: flex;
  flex-direction: column;
  padding: 1rem;
}

.sidebar-header {
  font-size: 1.2rem;
  margin-bottom: 1rem;
  color: #333;
}

.sidebar-nav {
  display: flex;
  flex-direction: column;
}

.nav-item {
  display: flex;
  align-items: center;
  padding: 0.5rem 0.75rem;
  margin-bottom: 0.25rem;
  color: #333;
  text-decoration: none;
  border-radius: 5px;
  transition: background-color 0.2s, color 0.2s;
}

.nav-item:hover {
  background-color: #f0f0f5;
  color: #5c7cfa;
}

.nav-item.active {
  font-weight: bold;
  color: #5c7cfa;
  background-color: #e6edff;
}

.icon {
  margin-right: 0.5rem;
}

.main-content {
  flex: 1;
  padding: 1rem;
  background-color: #f4f4f9;
  overflow-y: auto;
}
</style>
