<template>
  <section class="flex-center">
    <h1 class="text-4xl font-logo text-start w-full">Packages</h1>

    <div></div>
  </section>
</template>

<script setup>
  import { reactive, ref } from 'vue';

  const packages = ref([
    {
      name: 'Transformer',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/Transformer%20model'
    },
    {
      name: 'X-Llama',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/X-Llama'
    },
    {
      name: 'Dali',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/Dali'
    },
    {
      name: 'InstructGPT ',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/RLHF'
    }
  ]);
</script>
