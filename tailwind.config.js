/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/templates/**/*.html", // Semua file HTML di folder templates
    "./app/static/**/*.js",      // Semua file JavaScript di folder static
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ['Roboto serif'],
        sans: ['DM sans']
      },
      colors: {
        'primary-green': '#64CCC5',
      },
      screens: {
        sm: '480px',
        md: '768px',
        lg: '1024px',
        xl: '1280px',
      }
    },
  },
  plugins: [],
};
