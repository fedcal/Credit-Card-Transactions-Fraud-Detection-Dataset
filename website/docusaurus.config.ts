import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Math plugins (LaTeX)
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Credit Card Fraud — ML Pipeline',
  tagline: 'Pipeline ML per fraud detection con classificazione sbilanciata, feature temporali, ottimizzazione della soglia di decisione.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://fedcal.github.io',
  baseUrl: '/Credit-Card-Transactions-Fraud-Detection-Dataset/',

  organizationName: 'fedcal',
  projectName: 'Credit-Card-Transactions-Fraud-Detection-Dataset',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'it',
    locales: ['it'],
  },

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset/tree/main/website/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: true,
          showLastUpdateAuthor: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    metadata: [
      {name: 'keywords', content: 'machine learning, fraud detection, classificazione sbilanciata, AUC-PR, recall, precision, threshold tuning, kaggle'},
      {name: 'description', content: 'Pipeline ML per fraud detection con classificazione sbilanciata, feature temporali, ottimizzazione della soglia di decisione.'},
    ],
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Credit Card Fraud',
      logo: {
        alt: 'Credit Card Fraud Pipeline logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'mainSidebar',
          position: 'left',
          label: 'Documentazione',
        },
        {
          href: 'https://github.com/fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://federicocalo.dev',
          label: 'federicocalo.dev',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentazione',
          items: [
            {label: 'Introduzione', to: '/docs/intro'},
            {label: 'Teoria', to: '/docs/category/teoria'},
            {label: 'Scelte tecniche', to: '/docs/category/scelte-tecniche'},
          ],
        },
        {
          title: 'Codice',
          items: [
            {label: 'GitHub repo', href: 'https://github.com/fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset'},
            {label: 'Issues', href: 'https://github.com/fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset/issues'},
            {label: 'License (MIT)', href: 'https://github.com/fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset/blob/main/LICENSE'},
          ],
        },
        {
          title: 'Autore',
          items: [
            {label: 'federicocalo.dev', href: 'https://federicocalo.dev'},
            {label: 'GitHub', href: 'https://github.com/fedcal'},
            {label: 'LinkedIn', href: 'https://www.linkedin.com/in/federicocalo/'},
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Federico Calò · Progetto realizzato come parte del percorso <a href="https://datamasters.it/" target="_blank" rel="noopener noreferrer" style="color:#fff">Machine Learning Engineer di DataMasters/Skiller</a> · Distribuito sotto licenza MIT · <a href="https://federicocalo.dev" target="_blank" rel="noopener noreferrer" style="color:#fff"><strong>federicocalo.dev</strong></a>`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'yaml', 'json', 'toml', 'diff'],
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: false,
      },
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    announcementBar: {
      id: 'portfolio-banner',
      content:
        '🔗 Esplora gli altri progetti su <a target="_blank" rel="noopener noreferrer" href="https://federicocalo.dev"><strong>federicocalo.dev</strong></a>',
      backgroundColor: '#4f46e5',
      textColor: '#ffffff',
      isCloseable: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
