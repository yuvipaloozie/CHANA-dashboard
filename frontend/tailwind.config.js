/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bento: {
          bg: '#f4f3f6', // Very soft cool grey/lavender background
          card: '#ffffff',
          primary: '#7c5295', // Deep lavender/purple
          primaryHover: '#6a4482',
          accent: '#b298dc',  // Soft lavender
          text: '#2a2631',
          subtext: '#8a8596',
          border: '#e8e5ec'
        }
      }
    },
  },
  plugins: [],
}
