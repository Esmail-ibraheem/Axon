<template>
  <!-- ps-16 pt-10 pr-8 pb-10 -->
  <section class="h-full flex flex-col justify-center w-[50%] mx-auto">
    <h1 class="font-logo text-lg font-bold mb-4">{{ p.name }}</h1>

    <div
      v-html="p.description"
      class="w-full font-mono"></div>

    <div class="mt-10 group">
      <a
        class="rounded-[5rem] h-[54px] px-8 py-4 uppercase leading-none transition-all duration-[250ms] ease-in-out hover:rounded-2xl ml-auto dark:bg-gray-100 bg-zinc-950 dark:text-gray-800 text-zinc-100"
        :href="p.url">
        Package Source Code
      </a>
    </div>

    <boxFooter />
  </section>
</template>

<script setup lang="ts">
  import boxFooter from '@/components/boxFooter.vue';
  import router from '../routes/router';
  import { onBeforeMount, Ref, ref } from 'vue';
  import { packages } from '../data';

  type pgk = { id: ''; name: ''; url: ''; description: '' };
  const p: Ref<pgk> = ref({
    id: '',
    name: '',
    url: '',
    description: ''
  });

  const { id } = router.currentRoute.value.params;

  onBeforeMount(() => {
    const filtered = packages.filter((pkg) => pkg.id == id);

    if (filtered.length == 0) {
      router.push('/packages');
      return;
    }

    p.value = filtered[0];
  });
</script>
