/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export. The dashboard reads a published JSON file and holds no
  // credentials, so there is nothing for a server to do. It also means the
  // deployment cannot possibly reach the broker, which is the point.
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
