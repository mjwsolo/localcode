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
        'An open-source coding agent that runs local models on Apple Silicon. No API key, and no remote inference unless you point it at one.',
      components: {
        // Inline SVG lockup so the House Mark inherits the theme colour.
        // (Starlight's built-in `logo` renders an <img>, where currentColor
        // and CSS custom properties inside the SVG cannot resolve.)
        SiteTitle: './src/components/SiteTitle.astro',
        // The docs render the landing page's nav bar itself, so there is
        // one header component rather than two that drift apart.
        Header: './src/components/Header.astro',
        // Dark is the default; Starlight's built-in provider follows the OS.
        ThemeProvider: './src/components/ThemeProvider.astro',
        // Renders nothing: SiteNav owns the toggle, and Starlight's built-in
        // script would otherwise re-apply 'auto' after first paint.
        ThemeSelect: './src/components/ThemeSelect.astro',
      },
      favicon: '/favicon-house.svg',
      titleDelimiter: '·',
      // No `social` block: SiteNav draws the GitHub glyph itself, at the
      // landing page's size and order. Configuring it here would render a
      // second, smaller one that then has to be hidden in CSS.
      editLink: {
        baseUrl: 'https://github.com/mjwsolo/localcode/edit/main/website/',
      },
      customCss: [
        // Must match the landing page's imports in src/pages/index.astro,
        // or the docs silently fall back to -apple-system.
        '@fontsource-variable/instrument-sans',
        '@fontsource/geist-mono/400.css',
        '@fontsource/geist-mono/500.css',
        './src/styles/tokens.css',
        './src/styles/nav.css',
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
          attrs: { name: 'theme-color', content: '#0B0E11' },
        },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Install', slug: 'start-here/first-change' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Models', slug: 'start-here/choose-a-model' },
            { label: 'Permissions', slug: 'start-here/permissions' },
            { label: 'Offline', slug: 'guides/offline' },
            { label: 'MCP', slug: 'guides/mcp' },
            { label: 'Skills & Hooks', slug: 'guides/skills-and-hooks' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'Architecture', slug: 'concepts/architecture' },
            { label: 'Unified Memory', slug: 'concepts/unified-memory' },
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
        { label: 'Contributing', slug: 'contributing' },
      ],
    }),
  ],
});
