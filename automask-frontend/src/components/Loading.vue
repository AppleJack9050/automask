<template>
  <div class="center">
    Loading Image
    <i class="bi bi-arrow-clockwise spin" style="font-size: 4rem;"></i>
  </div>
</template>

<style>
image {
  pointer-events: visiblePainted;
}

.center {
  margin: 50%;
  width: 100%;
}

.spin {
  display: inline-block;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>