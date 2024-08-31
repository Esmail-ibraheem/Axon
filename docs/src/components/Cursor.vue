<template>
  <div
    id="bg"
    ref="bg"
    class="dark:opacity-50 pointer-events-none fixed -z-50 size-[25vh] origin-center rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400 blur-[100px] sm:size-[50vh]"></div>
</template>

<script setup lang="ts">
  import { onMounted } from 'vue';
  import { gsap } from 'gsap';

  const gradientColors = [
    'linear-gradient(90deg, #FC466B 0%, #3F5EFB 100%)',
    'linear-gradient(45deg, #1CB5E0 0%, #000851 100%)',
    'linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%)',
    'linear-gradient(36deg, #fcff9e 0%, #c67700 100%)',
    'linear-gradient(91deg, #efd5ff 0%, #515ada 100%)',
    'linear-gradient(15deg, #d53369 0%, #daae51 100%)',
    'linear-gradient(196deg, #4b6cb7 0%, #182848 100%)'
  ];

  const animateCursor = (
    cursor: HTMLDivElement,
    innerWidth: number,
    innerHeight: number
  ) => {
    gsap.set(cursor, {
      x: innerWidth / 2 - cursor.offsetWidth / 2,
      y: innerHeight / 2 - cursor.offsetHeight / 2,

      background: gradientColors[0],
      scale: 1,
      translateX: '-50%',
      translateY: '-50%'
    });

    const tl = gsap.timeline({
      defaults: {
        ease: 'power1.inOut',
        scale: 1.2,
        rotate: 360
      },
      repeat: -1,
      yoyo: true,
      repeatRefresh: true
    });

    gradientColors.forEach((gradient) => {
      tl.to(cursor, {
        background: gradient,
        duration: 5
      });
    });

    document.addEventListener('mousemove', (e) => {
      const newX = e.clientX - cursor.offsetWidth / 2;
      const newY = e.clientY - cursor.offsetHeight / 2;
      gsap.to(cursor, {
        x: newX,
        y: newY
      });
    });
  };

  onMounted(() => {
    const bg = document.querySelector('#bg') as HTMLDivElement;
    animateCursor(bg, window.innerWidth, window.innerHeight);
  });
</script>
