module.exports = {
  content: [
    "./pages/*.{html,js}",
    "./index.html",
    "./*.html",
    "./js/*.js"
  ],
  theme: {
    extend: {
      colors: {
        // Primary Colors - Deep Navy
        primary: {
          DEFAULT: "#1B365D", // navy-900
          50: "#EBF0F7",
          100: "#D7E1EF",
          200: "#AFC3DF",
          300: "#87A5CF",
          400: "#5F87BF",
          500: "#3769AF",
          600: "#2C5282", // navy-700
          700: "#1B365D", // navy-900
          800: "#152A4A",
          900: "#0F1E37",
        },
        // Secondary Colors - Supporting Blue
        secondary: {
          DEFAULT: "#2C5282", // blue-800
          50: "#EBF4FF",
          100: "#D6E9FF",
          200: "#ADD3FF",
          300: "#85BDFF",
          400: "#5CA7FF",
          500: "#3391FF",
          600: "#2C5282", // blue-800
          700: "#1E3A5F",
          800: "#16293D",
          900: "#0E1A26",
        },
        // Accent Colors - Gold
        accent: {
          DEFAULT: "#D69E2E", // yellow-600
          50: "#FEF9E7",
          100: "#FDF3CF",
          200: "#FBE79F",
          300: "#F9DB6F",
          400: "#F7CF3F",
          500: "#F5C30F",
          600: "#D69E2E", // yellow-600
          700: "#B8841F",
          800: "#9A6A10",
          900: "#7C5001",
        },
        // Background Colors
        background: "#FFFFFF", // white
        surface: {
          DEFAULT: "#F7FAFC", // gray-50
          100: "#EDF2F7", // gray-100
          200: "#E2E8F0", // gray-200
        },
        // Text Colors
        text: {
          primary: "#1A202C", // gray-900
          secondary: "#4A5568", // gray-600
          tertiary: "#718096", // gray-500
        },
        // Status Colors
        success: {
          DEFAULT: "#38A169", // green-600
          50: "#F0FFF4",
          100: "#C6F6D5",
          500: "#48BB78", // green-500
          600: "#38A169", // green-600
          700: "#2F855A", // green-700
        },
        warning: {
          DEFAULT: "#ED8936", // orange-500
          50: "#FFFAF0",
          100: "#FEEBC8",
          500: "#ED8936", // orange-500
          600: "#DD6B20", // orange-600
          700: "#C05621", // orange-700
        },
        error: {
          DEFAULT: "#E53E3E", // red-600
          50: "#FFF5F5",
          100: "#FED7D7",
          500: "#F56565", // red-500
          600: "#E53E3E", // red-600
          700: "#C53030", // red-700
        },
        // Border Colors
        border: {
          DEFAULT: "#E2E8F0", // gray-200
          active: "#D69E2E", // yellow-600
        },
      },
      fontFamily: {
        headline: ['Playfair Display', 'serif'],
        body: ['Inter', 'sans-serif'],
        cta: ['Inter', 'sans-serif'],
        accent: ['Source Sans Pro', 'sans-serif'],
      },
      boxShadow: {
        'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        'cta': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        'chat': '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
      },
      transitionDuration: {
        'fast': '250ms',
        'base': '300ms',
      },
      transitionTimingFunction: {
        'smooth': 'ease-in-out',
      },
      borderWidth: {
        'active': '2px',
      },
      animation: {
        'fade-in': 'fadeIn 300ms ease-in-out',
        'slide-up': 'slideUp 300ms ease-in-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
