<template>
  <!-- ps-16 pt-10 pr-8 pb-10 -->
  <!-- h-full flex flex-col justify-center -->

  <section class="h-full flex flex-col justify-center">
    <h1 class="font-logo text-lg font-bold mb-4">
      {{ packages.length }} Packages
    </h1>

    <!-- ! -->
    <div class="w-full">
      <pre
        class="grid gap-x-4 gap-y-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"><router-link v-for="(pkg, index) in packages" :key="pkg.name" :to="`/package/${pkg.id}`" class=" opacity-60 block mr-4 hover:opacity-100 transition-opacity duration-300 ease-in-out"><span class="mr-2 opacity-75">{{ (index ).toString().padStart(2, '0') }}</span><b>{{ pkg.name }}</b></router-link></pre>
    </div>

    <boxFooter />
  </section>
</template>

<script setup>
  import { packages } from '../data';
  import boxFooter from '@/components/boxFooter.vue';
</script>
