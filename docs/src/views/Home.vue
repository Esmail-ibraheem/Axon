<template>
  <section class="h-[80dvh]">
    <div
      class="flex items-center justify-between gap-10 max-sm:flex-col-reverse h-full">
      <div class="flex-1 space-y-10">
        <h1 class="text-3xl sm:text-4xl lg:text-5xl font-bold">
          <span
            class="inline-block bg-gradient-to-r from-cyan-500 via-blue-700 to-purple-800 bg-clip-text text-transparent"
            >Axon</span
          >
          <br />
          Your LabHub for Cutting-Edge AI Research and Implementation
        </h1>
        <p class="text-xl">
          Bridging Theory and Practice in Cutting-Edge AI Research
        </p>
      </div>
      <div class="flex-1 relative h-full flex items-center justify-center">
        <div
          class="absolute brightness-150 dark:brightness-50 inset-0 bg-layers bg-scale size-56 m-auto blur-xl bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 rounded-full md:size-96 lg:size-[30rem] md:blur-3xl"></div>
        <Logo class="z-10 w-[50%] md:w-[65%] lg:w-[50%] bg-layers" />
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
      transform: scale(0.9);
    }
    50% {
      transform: scale(1);
    }
    100% {
      transform: scale(0.9);
    }
  }
</style>
