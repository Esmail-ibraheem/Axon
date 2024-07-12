<template>
  <section class="min-h-[calc(100vh-5rem)]">
    <div class="grid grid-cols-1 sm:grid-cols-2">
      <div class="">Hi</div>
      <div class="max-sm:-order-first">
        <div class="relative h-full flex items-center justify-center">
          <div
            class="absolute inset-0 bg-layers bg-scale size-64 m-auto blur-xl bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 rounded-full md:w-[30rem] md:h-[30rem] md:blur-3xl"></div>
          <Logo class="z-10 w-[70%] bg-layers" />
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
  import Logo from '@/components/Logo.vue';
</script>

<style>
  .bg-layers {
    -webkit-animation: filter-animation 4s infinite;
    animation: filter-animation 4s infinite;
  }

  .bg-scale {
    -webkit-animation: filter-animation 8s infinite;
    animation: filter-scale 8s infinite;
  }

  @-webkit-keyframes filter-animation {
    0% {
      -webkit-filter: hue-rotate(0deg);
    }
    50% {
      -webkit-filter: hue-rotate(45deg);
    }

    100% {
      -webkit-filter: hue-rotate(0deg);
    }
  }

  @keyframes filter-animation {
    0% {
      filter: hue-rotate(0deg);
    }
    50% {
      filter: hue-rotate(45deg);
    }
    100% {
      filter: hue-rotate(0deg);
    }
  }

  @keyframes filter-scale {
    0% {
      transform: scale(0.8);
    }
    50% {
      transform: scale(1.15);
    }
    100% {
      transform: scale(0.8);
    }
  }
</style>
