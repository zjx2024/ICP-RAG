import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
// 构建产物输出到项目根 static/dist（由 FastAPI 挂载 /static 提供）
// 注意：outDir 指向 ../static/dist，不直接清空 static 目录，避免误删旧页面
export default defineConfig({
    plugins: [react()],
    // 资源路径前缀与 FastAPI 的 /static 挂载对齐：产物在 static/dist，
    // index.html 引用 /static/dist/assets/...（/static 挂载根 = static 目录）
    base: '/static/dist/',
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url)),
        },
    },
    server: {
        port: 5174,
        host: true,
        proxy: {
            // dev 模式下代理到后端 FastAPI（ICP_RAG 热启动端口 19999）
            '/api': {
                target: 'http://localhost:19999',
                changeOrigin: true,
            },
            '/images': {
                target: 'http://localhost:19999',
                changeOrigin: true,
            },
        },
    },
    build: {
        outDir: '../static/dist',
        emptyOutDir: true,
        assetsDir: 'assets',
        chunkSizeWarningLimit: 3000,
    },
});
