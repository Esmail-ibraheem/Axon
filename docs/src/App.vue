<template>
  <Cursor />
  <div class="w-full min-h-[100dvh]">
    <Nav />
    <main class="h-full">
      <router-view />
    </main>

    <!-- <footer class="common-padding size-full"></footer> -->
  </div>
</template>

<script setup>
  import Nav from '@/components/Nav.vue';
  import Cursor from '@/components/Cursor.vue';
</script>
