<template>
  <!-- ps-16 pt-10 pr-8 pb-10 -->

  <section
    class="relative flex flex-col justify-center h-[85dvh] lg:w-[70%] mx-auto text-lg common-padding">
    <h1 class="font-logo font-bold mb-4">Description:</h1>

    <h1
      v-html="p.description"
      class="font-mono w-full"></h1>

    <div class="mt-10 group">
      <a
        class="rounded-[5rem] h-[54px] px-8 py-4 uppercase leading-none transition-all duration-[250ms] ease-in-out hover:rounded-2xl ml-auto dark:bg-gray-100 bg-zinc-950 dark:text-gray-800 text-zinc-100 text-sm"
        :href="p.url">
        Package Source Code
      </a>
    </div>
    <boxFooter />
  </section>

  <div
    v-auto-animate
    class="opacity-50 hover:opacity-100 group hidden sm:block sm:absolute sm:bottom-7 sm:left-10 cursor-pointer transition-all duration-500 ease-in-out">
    <template
      v-for="i in packagesList"
      :key="i.id">
      <router-link
        :to="`/Axon/package/${i.id}`"
        :class="i.id >= 0 ? 'block' : 'hidden'">
        <p
          :class="i.id == p.id ? 'opacity-100' : 'opacity-0'"
          class="group-hover:opacity-50 hover:!opacity-85 transition-all duration-500 ease-in-out">
          {{ i.name }}
        </p>
      </router-link>
    </template>
  </div>
</template>

<script setup lang="ts">
  // @ts-ignore
  import boxFooter from '@/components/boxFooter.vue';
  import { onBeforeMount, Ref, ref, watch } from 'vue';
  import { packages } from '../data';
  import { useRoute, useRouter } from 'vue-router';

  const route = useRoute();
  const router = useRouter();

  const p: Ref<{ id: number; name: string; url: string; description: string }> =
    ref({
      id: -1, // or any number value
      name: '',
      url: '',
      description: ''
    });

  const before: Ref<{
    id: number;
    name: string;
    url: string;
    description: string;
  }> = ref({
    id: -1, // or any number value
    name: '',
    url: '',
    description: ''
  });

  const after: Ref<{
    id: number;
    name: string;
    url: string;
    description: string;
  }> = ref({
    id: -1, // or any number value
    name: '',
    url: '',
    description: ''
  });

  const { id } = router.currentRoute.value.params;

  const packagesList: Ref<any[]> = ref([]);

  const fetchData = (id: number) => {
    if (isNaN(id)) {
      router.push('/packages');
      return;
    }

    packagesList.value = [];

    const filtered = packages.filter((pkg) => Number(pkg.id) == id);

    if (id > 0) {
      before.value = packages.filter((pkg) => Number(pkg.id) == id - 1)[0];
      if (before.value.id > -1) {
        packagesList.value.push(before.value);
      }
    }
    if (id < packages.length - 1) {
      after.value = packages.filter((pkg) => Number(pkg.id) == id + 1)[0];
      if (after.value.id > -1) {
        packagesList.value.push(after.value);
      }
    }

    if (filtered.length == 0) {
      router.push('/packages');
      return;
    }

    packagesList.value.push(filtered[0]);

    packagesList.value.sort((a, b) => a.id - b.id);
    p.value = filtered[0];
  };

  watch(
    () => route.params.id,
    (newId) => {
      fetchData(Number(newId));
    },
    { immediate: true }
  );

  onBeforeMount(() => {
    fetchData(Number(id));
  });
</script>
