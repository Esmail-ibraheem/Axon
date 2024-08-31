<template>
  <section class="common-padding w-screen h-[85dvh]">
    <div
      class="flex justify-between items-center max-sm:gap-4 gap-10 max-sm:flex-col-reverse h-full">
      <div class="flex-1 h-full w-full content-center space-y-10">
        <h1
          class="text-3xl sm:text-4xl lg:text-5xl font-bold w-full text-start h-1/4 max-sm:h-1/2">
          <p class="text-fancy">Axon</p>
          <p>
            <span
              id="typing"
              class="w-[90%] font-bold inline"></span>
          </p>
        </h1>

        <a
          href="https://github.com/Esmail-ibraheem/Axon"
          class="block w-fit z-50 bg-fancy text-white font-bold cursor-pointer rounded-full px-8 py-3 hover:scale-110 transition-transform duration-300 ease-in-out">
          See Source Code
        </a>
      </div>

      <div class="flex-1 relative h-full flex items-center justify-center">
        <!-- <div
          class="absolute inset-0 brightness-150 dark:brightness-50 bg-layers bg-scale size-56 m-auto blur-xl bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 rounded-full md:size-80 lg:size-[30rem] md:blur-3xl"></div> -->
        <Logo class="z-10 w-[50%] md:w-[65%] lg:w-[50%] bg-layers" />
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
  import Logo from '@/components/Logo.vue';
  import { onMounted } from 'vue';
  import Typed from 'typed.js';

  const words = [
    'Where AI Research Meets Real-World Impact',
    'From Theoretical Insights to AI Breakthroughs',
    'Pushing the Boundaries of Artificial Intelligence',
    'Innovate, Implement, Inspire in AI'
  ];
  onMounted(() => {
    new Typed('#typing', {
      typeSpeed: 50,
      loop: true,
      shuffle: true,
      backDelay: 2000,
      smartBackspace: true,
      fadeOut: true,
      strings: words
    });
  });
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
