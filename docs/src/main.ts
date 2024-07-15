import { createApp } from 'vue';
// @ts-ignore
import App from './App.vue';
import router from './routes/router.ts';
import './style.css';
import { autoAnimatePlugin } from '@formkit/auto-animate/vue';

const app = createApp(App);
app.use(router);
app.use(autoAnimatePlugin);
app.mount('#app');
