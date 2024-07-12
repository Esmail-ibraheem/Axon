import { createWebHistory, createRouter } from 'vue-router';
import Home from '@/views/Home.vue';
import Packages from '@/views/Packages.vue';
import Package from '@/views/Package.vue';

const routes = [
  {
    name: 'home',
    path: '/',
    component: Home
  },
  {
    name: 'packages',
    path: '/packages',
    component: Packages
  },
  {
    name: 'package',
    path: '/package/:id',
    component: Package
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes: routes
});

export default router;
