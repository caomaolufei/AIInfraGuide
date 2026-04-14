import defaultTheme from 'tailwindcss/defaultTheme';
import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        apple: {
          blue: '#007aff',
          'blue-dark': '#0a84ff',
          cyan: '#5ac8fa',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'SF Pro Display',
          ...defaultTheme.fontFamily.sans,
        ],
        mono: ['JetBrains Mono', ...defaultTheme.fontFamily.mono],
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            '--tw-prose-links': '#007aff',
            maxWidth: 'none',
            code: {
              backgroundColor: theme('colors.blue.50'),
              borderRadius: '6px',
              padding: '2px 6px',
              fontWeight: '400',
            },
            'code::before': { content: 'none' },
            'code::after': { content: 'none' },
            img: { borderRadius: '8px' },
            pre: { borderRadius: '8px' },
          },
        },
        invert: {
          css: {
            '--tw-prose-links': '#0a84ff',
            code: {
              backgroundColor: 'rgba(59, 130, 246, 0.15)',
            },
          },
        },
      }),
    },
  },
  plugins: [typography],
};
