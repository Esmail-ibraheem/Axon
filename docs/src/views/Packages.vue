<template>
  <!-- ps-16 pt-10 pr-8 pb-10 -->
  <section
    class="max-h-[100%] absolute top-1/2 left-1/2 translate-x-[-50%] translate-y-[-50%]">
    <h1 class="font-logo text-lg font-bold mb-4">
      {{ packages.length }} Packages
    </h1>

    <pre
      class="grid gap-x-4 gap-y-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"><a v-for="(pkg, index) in packages" :key="pkg.name" :href="pkg.url" class=" opacity-60 block mr-4 hover:opacity-100 transition-opacity duration-300 ease-in-out"><span class="mr-2 opacity-75">{{ (index ).toString().padStart(2, '0') }}</span><b>{{ pkg.name }}</b></a></pre>

    <h1 class="font-mono mb-2 mt-8">
      <a
        class="select-none opacity-50 hover:opacity-100 transition-opacity duration-300 ease-in-out"
        href="#"
        target="_blank"
        >built with {{ mode == 'dark' ? '🤍' : '🖤' }} by 𝕏: @4Hetary</a
      >

      <span class="mx-1.5 opacity-75 inline-flex select-none animate-pulse">
        .
      </span>

      <button
        @click="toggleDark()"
        class="select-none opacity-50 hover:opacity-100 transition-opacity duration-300 ease-in-out">
        {{ mode }}
      </button>

      <span class="inline-block opacity-75"
        >first commit on <time datetime="2024/04/17"></time>2024/04/17</span
      >
    </h1>
  </section>
</template>

<script setup>
  import { onMounted, ref, watch } from 'vue';
  import { useDark, useToggle } from '@vueuse/core';
  import { packages } from '../data';

  const mode = ref('');

  const toggleDark = useToggle(useDark());

  onMounted(() => {
    mode.value = useDark().value ? 'dark' : 'light';
  });

  watch(useDark, (newVal) => {
    console.log('useDark', newVal.value);
    mode.value = newVal.value ? 'dark' : 'light';
  });
</script>
