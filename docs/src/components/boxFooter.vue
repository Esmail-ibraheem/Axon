<template>
  <h1 class="font-mono mb-2 mt-8 text-lg">
    <a
      class="select-none opacity-50 hover:opacity-100 transition-opacity duration-300 ease-in-out"
      href="https://github.com/Esmail-ibraheem/Axon"
      target="_blank"
      >source-code</a
    >

    <span class="mx-1.5 opacity-75 inline-flex select-none animate-pulse">
      .
    </span>

    <button
      @click="toggleDark()"
      class="select-none opacity-50 hover:opacity-100 transition-opacity duration-300 ease-in-out">
      {{ mode }}
    </button>

    <span class="block opacity-75"
      >first commit on <time datetime="2024/04/17"></time>2024/04/17</span
    >
  </h1>
</template>

<script setup>
  import { onMounted, ref, watch } from 'vue';
  import { useDark, useToggle } from '@vueuse/core';
  import { packages } from '../data';
  import boxFooter from '@/components/boxFooter.vue';

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
