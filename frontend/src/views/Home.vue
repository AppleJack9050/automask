<script setup>
import fileService from "@/services/fileservice"
import { useRouter } from "vue-router";
const router = useRouter();
</script> 

<template>
  <div class="home">
    <div class="cards">
      <div class="card">
        <button 
          @click="fileService.processFiles"
          class="btn btn-warning"
        >
          Process Stored Files
        </button>
      </div>
      <div class="card">
        <button
          @click="router.push('/view-files')"
          class="btn btn-secondary"  
          >
          View Processed Files
        </button>
      </div>
    </div>
  </div>
</template>
 
<style scoped>
.home {
  padding: 1rem;
}
  
.cards {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}
  
.card {
  background-color: #fff;
  border: 1px solid #ddd;
  padding: 1rem;
  border-radius: 8px;
  flex: 1;
  transition: transform 0.2s;
}
</style>