<template>
  <div class="w-full min-h-[100dvh] grid grid-rows-[auto_1fr_auto]">
    <Nav />
    <main class="common-padding">
      <router-view />
    </main>

    <!-- <footer class="common-padding size-full"></footer> -->
  </div>
</template>

<script setup>
  import Nav from '@/components/Nav.vue';
</script>
