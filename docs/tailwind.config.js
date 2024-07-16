import plugin from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        logo: ['chillax', 'monospace']
      }
    }
  },
  plugins: [
    plugin(function ({ addUtilities }) {
      addUtilities({
        '.transition-discrete': {
          transitionBehavior: 'allow-discrete'
        }
      });
    })
  ]
};
