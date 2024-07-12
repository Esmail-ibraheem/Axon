<template>
  <Nav />
  <main class="common-padding">
    <router-view />
  </main>
</template>

<script setup>
  import Nav from '@/components/Nav.vue';
</script>
