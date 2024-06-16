<template>
  <section class="flex items-center justify-center h-[70vh] space-y-10">
    <div class="max-w-lg space-y-10 w-full">
      <h1 class="text-2xl !text-start w-full text-gray-500">Packages:</h1>
      <div class="grid grid-cols-4 gap-x-8 gap-y-2">
        <p
          v-for="pkg in packages"
          :key="pkg.name"
          class="text-start text-gray-500 hover:text-theme transition-[color] duration-300 ease-in-out">
          <a
            :href="pkg.url"
            class="text-gray-500">
            {{ pkg.name }}
          </a>
        </p>
      </div>
    </div>
  </section>
</template>

<script setup>
  import { reactive, ref } from 'vue';

  const packages = [
    {
      name: 'Transformer',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/Transformer%20model'
    },
    {
      name: 'X-Llama',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/X-Llama'
    },
    {
      name: 'Dali package',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/Dali'
    },
    {
      name: 'InstructGPT ',
      url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/RLHF'
    }
  ];
</script>
