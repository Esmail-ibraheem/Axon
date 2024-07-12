<template>
  <!-- ps-16 pt-10 pr-8 pb-10 -->
  <section
    class="max-h-[100%] absolute top-1/2 left-1/2 translate-x-[-50%] translate-y-[-50%]">
    <h1 class="font-logo text-lg font-bold mb-4">{{ p.name }}</h1>

    <div class="w-full font-mono">
      {{ p.description }}
    </div>

    <!-- TODO: fix the transition -->
    <div class="mt-10 group">
      <a
        class="py-4 px-8 rounded-full group-hover:opacity-100 group-hover:rounded-md transition-all duration-300 ease-in-out opacity-85 dark:bg-gray-100 bg-zinc-950 dark:text-gray-800 text-zinc-100"
        :href="p.url">
        See more
      </a>
    </div>

    <boxFooter />
  </section>

  <div class="absolute bottom-10 left-10">
    {{ p.name }}
  </div>
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
