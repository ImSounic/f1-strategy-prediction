/** @type {import('next').NextConfig} */
const nextConfig = {
  // Remove 'output: export' to support API calls in production
  // Vercel handles SSR/SSG automatically
  images: { unoptimized: true },
}
module.exports = nextConfig
