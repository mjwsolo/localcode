// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Preview docs site for localcode.
//
// `site` + `base` are set for a GitHub Pages preview deploy under
// /localcode/. Local `npm run dev` and `npm run preview` both honour the
// base path, so links behave the same locally and when published.
export default defineConfig({
  site: 'https://mjwsolo.github.io',
  base: '/localcode',
  trailingSlash: 'ignore',
  integrations: [
    starlight({
      title: 'localcode',
      description:
        'An open-source coding agent that runs local models on Apple Silicon. No API key. No remote inference.',
      components: {
        // Inline SVG lockup so the Finder Mark inherits the theme colour.
        // (Starlight's built-in `logo` renders an <img>, where currentColor
        // and CSS custom properties inside the SVG cannot resolve.)
        SiteTitle: './src/components/SiteTitle.astro',
      },
      favicon: '/favicon.svg',
      titleDelimiter: '·',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/mjwsolo/localcode',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/mjwsolo/localcode/edit/main/website/',
      },
      customCss: [
        '@fontsource/inter/400.css',
        '@fontsource/inter/500.css',
        '@fontsource/inter/600.css',
        '@fontsource/martian-mono/400.css',
        '@fontsource/martian-mono/600.css',
        '@fontsource/commit-mono/400.css',
        './src/styles/tokens.css',
        './src/styles/docs.css',
      ],
      // NOTE: no `og:image` is emitted. Open Graph requires an ABSOLUTE URL,
      // and this preview is not deployed anywhere yet — pointing it at
      // https://mjwsolo.github.io/localcode/ would name an origin that
      // currently serves the MkDocs site and has no social-preview.svg.
      // `public/social-preview.svg` is built and ready; wire the tag up at
      // deploy time. See website/README.md → "Deployment-time requirements".
      head: [
        {
          tag: 'meta',
          attrs: { name: 'theme-color', content: '#14110F' },
        },
      ],
      sidebar: [
        {
          label: 'Start Here',
          items: [
            { label: 'Install', slug: 'start-here/install' },
            { label: 'First Change', slug: 'start-here/first-change' },
            { label: 'Choose a Model', slug: 'start-here/choose-a-model' },
            { label: 'Permissions', slug: 'start-here/permissions' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Offline', slug: 'guides/offline' },
            { label: 'Headless', slug: 'guides/headless' },
            { label: 'MCP', slug: 'guides/mcp' },
            { label: 'Skills & Hooks', slug: 'guides/skills-and-hooks' },
            { label: 'Undo', slug: 'guides/undo' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'Architecture', slug: 'concepts/architecture' },
            { label: 'Unified Memory', slug: 'concepts/unified-memory' },
            { label: 'Verification', slug: 'concepts/verification' },
            { label: 'Network Boundary', slug: 'concepts/network-boundary' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'CLI', slug: 'reference/cli' },
            { label: 'Slash Commands', slug: 'reference/slash-commands' },
            { label: 'Configuration', slug: 'reference/configuration' },
            { label: 'JSONL Events', slug: 'reference/jsonl-events' },
            { label: 'Error Codes', slug: 'reference/error-codes' },
          ],
        },
        { label: 'Models & Performance', slug: 'models-and-performance' },
        { label: 'Contributing', slug: 'contributing' },
      ],
    }),
  ],
});
