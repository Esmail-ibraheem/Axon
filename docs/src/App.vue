<template>
  <div class="w-full min-h-[100dvh]">
    <Nav />
    <main class="min-h-[80dvh] grid grid-rows-[1fr_auto]">
      <router-view />
    </main>

    <!-- <footer class="common-padding size-full"></footer> -->
  </div>
</template>

<script setup>
  import Nav from '@/components/Nav.vue';
</script>
