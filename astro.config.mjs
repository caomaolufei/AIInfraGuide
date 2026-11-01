import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeExternalLinks from 'rehype-external-links';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeTableWrap from './src/utils/rehype-table-wrap.mjs';
import rehypeReferences from './src/utils/rehype-references.mjs';

export default defineConfig({
  site: 'https://caomaolufei.github.io',
  base: '/AIInfraGuide',
  integrations: [tailwind(), sitemap()],
  experimental: {
    contentLayer: true,
  },
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      defaultColor: 'light',
      wrap: false,
    },
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeSlug,
      [rehypeAutolinkHeadings, { behavior: 'wrap' }],
      [rehypeExternalLinks, { target: '_blank', rel: ['nofollow', 'noopener'] }],
      rehypeKatex,
      rehypeTableWrap,
      rehypeReferences,
    ],
  },
});
