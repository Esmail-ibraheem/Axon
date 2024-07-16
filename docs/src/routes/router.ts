import { createWebHistory, createRouter } from 'vue-router';
import Home from '@/views/Home.vue';
import Packages from '@/views/Packages.vue';
import Package from '@/views/Package.vue';

const routes = [
  {
    name: 'home',
    path: '/Axon/',
    component: Home
  },
  {
    name: 'packages',
    path: '/Axon/packages',
    component: Packages
  },
  {
    name: 'package',
    path: '/Axon/package/:id',
    component: Package
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes: routes
});

export default router;
