<template>
  <Nav />
  <main class="common-padding min-h-[calc(100vh-5rem)]">
    <router-view />
  </main>
</template>

<script setup>
  import Nav from '@/components/Nav.vue';
</script>
