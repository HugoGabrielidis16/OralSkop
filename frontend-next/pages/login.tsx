import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { motion } from 'framer-motion'
import { supabase } from '@/lib/supabase'
import { registerUser } from '@/lib/api'

function GoogleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" />
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
    </svg>
  )
}

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) router.push('/guide')
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session) router.push('/guide')
    })

    return () => subscription.unsubscribe()
  }, [router])

  const handleGoogleSignIn = async () => {
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: (typeof window !== 'undefined' ? window.location.origin : '') + '/guide',
      },
    })
  }

  const handleEmailSignIn = async () => {
    if (!email || !password) { setError('Enter email and password'); return }
    setLoading(true)
    setError(null)
    const { data, error: err } = await supabase.auth.signInWithPassword({ email, password })
    if (err) { setError(err.message); setLoading(false); return }
    if (data.session) {
      localStorage.setItem('oralskop_token', data.session.access_token)
      router.push('/guide')
    }
    setLoading(false)
  }

  const handleRegister = async () => {
    if (!email || !password) { setError('Enter email and password'); return }
    setLoading(true)
    setError(null)
    try {
      await registerUser(email, password)
      await handleEmailSignIn()
    } catch {
      setError('Registration failed')
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-cream flex flex-col items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="flex flex-col items-center gap-6 w-full max-w-[320px]"
      >
        {/* Logo */}
        <div className="flex flex-col items-center gap-3">
          <svg width="72" height="72" viewBox="0 0 36 36" fill="none">
            <circle cx="18" cy="18" r="18" fill="#0F6E56" />
            <circle cx="18" cy="16" r="9" stroke="white" strokeWidth="2.5" fill="none" />
            <path d="M12 19 Q18 25 24 19" stroke="white" strokeWidth="2" strokeLinecap="round" fill="none" />
          </svg>
          <span className="text-3xl font-extrabold text-charcoal tracking-tight">OralSkop</span>
          <p className="text-gray text-sm text-center">AI-powered oral health screening</p>
        </div>

        {/* Google sign-in */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleGoogleSignIn}
          className="w-full flex items-center justify-center gap-3 bg-white border border-line rounded-pill py-3.5 px-6 shadow-sm font-semibold text-charcoal text-sm transition-shadow hover:shadow-md"
        >
          <GoogleIcon />
          Continue with Google
        </motion.button>

        {/* Divider */}
        <div className="w-full flex items-center gap-3 text-gray text-xs">
          <div className="flex-1 h-px bg-line" />
          <span>or</span>
          <div className="flex-1 h-px bg-line" />
        </div>

        {/* Email / Password */}
        <div className="flex flex-col gap-2.5 w-full">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-4 py-3.5 border border-line rounded-xl text-sm font-inter outline-none focus:border-teal bg-white text-charcoal"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleEmailSignIn()}
            className="w-full px-4 py-3.5 border border-line rounded-xl text-sm font-inter outline-none focus:border-teal bg-white text-charcoal"
          />
        </div>

        {error && (
          <div className="w-full px-4 py-2.5 rounded-xl text-sm font-semibold" style={{ backgroundColor: '#FDEDE6', color: '#D85A30' }}>
            {error}
          </div>
        )}

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleEmailSignIn}
          disabled={loading}
          className="w-full py-4 rounded-pill text-white font-bold text-base disabled:opacity-60"
          style={{ backgroundColor: '#0F6E56' }}
        >
          Sign in
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleRegister}
          disabled={loading}
          className="w-full py-3.5 rounded-pill font-bold text-sm border-2 disabled:opacity-60"
          style={{ borderColor: '#0F6E56', color: '#0F6E56', backgroundColor: 'transparent' }}
        >
          Create account
        </motion.button>

        <p className="text-xs text-gray text-center">
          By signing in you agree to our terms of use and privacy policy.
        </p>
      </motion.div>
    </div>
  )
}
