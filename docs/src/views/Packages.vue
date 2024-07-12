<template>
  <!-- ps-16 pt-10 pr-8 pb-10 -->
  <section
    class="max-h-[100%] absolute top-1/2 left-1/2 translate-x-[-50%] translate-y-[-50%]">
    <h1 class="font-logo text-lg font-bold mb-4">
      {{ packages.length }} Packages
    </h1>

    <pre
      class="grid gap-x-4 gap-y-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"><router-link v-for="(pkg, index) in packages" :key="pkg.name" :to="`/package/${pkg.id}`" class=" opacity-60 block mr-4 hover:opacity-100 transition-opacity duration-300 ease-in-out"><span class="mr-2 opacity-75">{{ (index ).toString().padStart(2, '0') }}</span><b>{{ pkg.name }}</b></router-link></pre>

    <boxFooter />
  </section>
</template>

<script setup>
  import { packages } from '../data';
  import boxFooter from '@/components/boxFooter.vue';
</script>
