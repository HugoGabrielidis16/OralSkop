import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { Inter } from 'next/font/google'
import { useEffect } from 'react'
import { useRouter } from 'next/router'
import { supabase } from '@/lib/supabase'

const inter = Inter({
  subsets: ['latin'],
  weight: ['400', '600', '700', '800'],
  variable: '--font-inter',
  display: 'swap',
})

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter()
  const isDentist = router.pathname === '/dentist' ||
    (router.pathname === '/profile' && router.query.view === 'dentist')

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.access_token) {
        localStorage.setItem('oralskop_token', session.access_token)
      } else {
        localStorage.removeItem('oralskop_token')
      }
    })
    return () => { subscription.unsubscribe() }
  }, [])

  // Dentist view: full desktop layout, no phone frame
  if (isDentist) {
    return (
      <div className={`${inter.variable} font-inter`}>
        <Component {...pageProps} />
      </div>
    )
  }

  // All other pages: mobile phone frame simulation
  return (
    <div className={`${inter.variable} font-inter phone-wrapper`}>
      <div className="phone-device">

        {/* Phone chrome — desktop only */}
        <div className="hidden md:block">
          {/* Dynamic island */}
          <div className="flex justify-center pt-3 pb-1">
            <div style={{ width: 120, height: 34, backgroundColor: '#000', borderRadius: 20 }} />
          </div>
          {/* Status bar */}
          <div
            className="flex items-center justify-between px-7 pb-2"
            style={{ color: 'rgba(255,255,255,0.9)', fontSize: 12.5, fontWeight: 600, letterSpacing: '-0.2px' }}
          >
            <span>9:41</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <svg width="17" height="11" viewBox="0 0 17 11" fill="white">
                <rect x="0" y="8" width="3" height="3" rx="0.7" />
                <rect x="4.5" y="5.5" width="3" height="5.5" rx="0.7" />
                <rect x="9" y="3" width="3" height="8" rx="0.7" />
                <rect x="13.5" y="0" width="3" height="11" rx="0.7" />
              </svg>
              <svg width="16" height="12" viewBox="0 0 16 12" fill="none">
                <path d="M8 9 L9.5 12H6.5Z" fill="white" />
                <path d="M4.5 6.5C5.6 5.4 6.7 4.8 8 4.8C9.3 4.8 10.4 5.4 11.5 6.5" stroke="white" strokeWidth="1.4" strokeLinecap="round" />
                <path d="M1.5 3.5C3.4 1.6 5.5 0.6 8 0.6C10.5 0.6 12.6 1.6 14.5 3.5" stroke="white" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{ width: 26, height: 12, border: '1.5px solid rgba(255,255,255,0.7)', borderRadius: 3.5, padding: 1.5, display: 'flex', alignItems: 'stretch' }}>
                  <div style={{ width: '80%', backgroundColor: 'white', borderRadius: 1.5 }} />
                </div>
                <div style={{ width: 2, height: 5, backgroundColor: 'rgba(255,255,255,0.45)', borderRadius: 1, marginLeft: 1 }} />
              </div>
            </div>
          </div>
        </div>

        {/* App screen */}
        <div className="phone-screen">
          <Component {...pageProps} />
        </div>

        {/* Home indicator — desktop only */}
        <div className="hidden md:flex justify-center py-3">
          <div style={{ width: 134, height: 5, backgroundColor: 'rgba(255,255,255,0.25)', borderRadius: 3 }} />
        </div>

      </div>
    </div>
  )
}
