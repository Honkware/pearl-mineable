import { defineConfig } from 'vite';

// GitHub Pages serves the site under the repository name.
export default defineConfig({ base: process.env.GITHUB_ACTIONS ? '/pearl-mineable/' : '/' });
