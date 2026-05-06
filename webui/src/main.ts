import { createApp } from "vue";
import { createRouter, createWebHistory } from "vue-router";
import App from "./App.vue";
import Dashboard from "./components/Dashboard.vue";
import RequestsList from "./components/RequestsList.vue";

const routes = [
  { path: "/", name: "Dashboard", component: Dashboard },
  { path: "/requests", name: "RequestsList", component: RequestsList },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

const app = createApp(App);
app.use(router);
app.mount("#app");