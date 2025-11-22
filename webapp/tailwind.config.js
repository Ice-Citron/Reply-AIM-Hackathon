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
        // Liquid Glass - Primary Blues (lighter, more ethereal)
        primary: {
          DEFAULT: "#4A90E2", // soft blue
          50: "#F0F7FF",
          100: "#E0EFFF",
          200: "#B8DCFF",
          300: "#90CAFF",
          400: "#68B7FF",
          500: "#4A90E2",
          600: "#3B78C8",
          700: "#2C5FA0",
          800: "#1E4678",
          900: "#0F2D50",
        },
        // Secondary Colors - Soft Purples
        secondary: {
          DEFAULT: "#8B7EC8", // soft purple
          50: "#F5F3FF",
          100: "#EBE7FF",
          200: "#D6CFFF",
          300: "#C2B7FF",
          400: "#AD9FE8",
          500: "#8B7EC8",
          600: "#7265B0",
          700: "#594C98",
          800: "#403380",
          900: "#271A68",
        },
        // Accent Colors - Luminous Cyan/Teal
        accent: {
          DEFAULT: "#5ED4D9", // cyan glass
          50: "#F0FEFF",
          100: "#E0FDFF",
          200: "#B8F9FF",
          300: "#90F5FF",
          400: "#68EEFF",
          500: "#5ED4D9",
          600: "#4BB8BD",
          700: "#389CA1",
          800: "#258085",
          900: "#126469",
        },
        // Background Colors - Translucent
        background: "#FAFBFF", // very light blue-tinted white
        surface: {
          DEFAULT: "#FFFFFF", // white
          glass: "rgba(255, 255, 255, 0.7)", // glass surface
          100: "#F8FAFF",
          200: "#F0F4FF",
        },
        // Text Colors - Softer
        text: {
          primary: "#2D3748", // softer dark
          secondary: "#718096", // medium gray
          tertiary: "#A0AEC0", // light gray
          glass: "#4A5568", // glass text
        },
        // Status Colors
        success: {
          DEFAULT: "#48D597", // soft green
          50: "#F0FFF8",
          100: "#C6F6E0",
          500: "#48D597",
          600: "#38C785",
          700: "#28B873",
        },
        warning: {
          DEFAULT: "#FFB84D", // soft orange
          50: "#FFF8F0",
          100: "#FFECD6",
          500: "#FFB84D",
          600: "#F5A63B",
          700: "#EB9429",
        },
        error: {
          DEFAULT: "#FF6B7A", // soft red
          50: "#FFF5F7",
          100: "#FFE0E5",
          500: "#FF6B7A",
          600: "#F55968",
          700: "#EB4756",
        },
        // Border Colors
        border: {
          DEFAULT: "rgba(226, 232, 240, 0.5)", // translucent
          active: "rgba(94, 212, 217, 0.6)", // translucent cyan
          glass: "rgba(255, 255, 255, 0.3)", // glass border
        },
      },
      fontFamily: {
        headline: ['Playfair Display', 'serif'],
        body: ['Inter', 'sans-serif'],
        cta: ['Inter', 'sans-serif'],
        accent: ['Source Sans Pro', 'sans-serif'],
      },
      boxShadow: {
        'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.03)',
        'md': '0 4px 12px -2px rgba(74, 144, 226, 0.12), 0 2px 6px -1px rgba(139, 126, 200, 0.08)',
        'lg': '0 10px 25px -5px rgba(74, 144, 226, 0.15), 0 4px 10px -3px rgba(139, 126, 200, 0.1)',
        'xl': '0 20px 40px -10px rgba(74, 144, 226, 0.2), 0 10px 20px -5px rgba(139, 126, 200, 0.12)',
        'glass': '0 8px 32px 0 rgba(74, 144, 226, 0.15)',
        'glass-lg': '0 12px 48px 0 rgba(74, 144, 226, 0.2)',
        'inner-glass': 'inset 0 1px 2px 0 rgba(255, 255, 255, 0.5)',
        'glow': '0 0 20px rgba(94, 212, 217, 0.4)',
        'glow-lg': '0 0 40px rgba(94, 212, 217, 0.5)',
      },
      backdropBlur: {
        xs: '2px',
        glass: '12px',
        'glass-lg': '16px',
        'glass-xl': '24px',
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
        'fade-in': 'fadeIn 500ms ease-out',
        'slide-up': 'slideUp 500ms ease-out',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'shimmer': 'shimmer 3s linear infinite',
        'glow-pulse': 'glowPulse 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(30px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        glowPulse: {
          '0%, 100%': { opacity: '1', boxShadow: '0 0 20px rgba(94, 212, 217, 0.4)' },
          '50%': { opacity: '0.8', boxShadow: '0 0 40px rgba(94, 212, 217, 0.6)' },
        },
      },
      backgroundImage: {
        'glass-gradient': 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%)',
        'glass-gradient-primary': 'linear-gradient(135deg, rgba(74, 144, 226, 0.1) 0%, rgba(139, 126, 200, 0.1) 100%)',
        'glass-gradient-accent': 'linear-gradient(135deg, rgba(94, 212, 217, 0.15) 0%, rgba(74, 144, 226, 0.15) 100%)',
        'shimmer-gradient': 'linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.4) 50%, transparent 100%)',
      },
    },
  },
  plugins: [],
}
