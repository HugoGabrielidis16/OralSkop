import { useEffect, useState, useCallback, useRef } from 'react'
import { useRouter } from 'next/router'
import { motion, AnimatePresence } from 'framer-motion'
import Logo from '@/components/Logo'
import SegmentedToggle from '@/components/SegmentedToggle'
import BottomNav from '@/components/BottomNav'
import Toast from '@/components/Toast'
import { supabase } from '@/lib/supabase'
import { fetchHistory, registerUser, deleteScreening } from '@/lib/api'
import { HistoryScreening, UserProfile } from '@/lib/types'

const CONDITION_LABELS: Record<string, string> = {
  cavity: 'Cavity',
  caries: 'Caries',
  abrasion: 'Abrasion',
  gingivitis: 'Localised gingivitis',
  tartar: 'Tartar build-up',
  lesion_suspicious: 'Area to monitor',
  crown: 'Crown',
}

const CONDITION_COLORS: Record<string, string> = {
  abrasion: '#F97316',
  filling: '#8B5CF6',
  crown: '#3B82F6',
  caries: '#EF4444',
  cavity: '#EF4444',
  gingivitis: '#EC4899',
  tartar: '#EAB308',
  lesion_suspicious: '#991B1B',
}

// ── Google SVG ────────────────────────────────────────────────────────────────
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

// ── Login overlay ─────────────────────────────────────────────────────────────
interface LoginOverlayProps {
  onLoggedIn: () => void
}
function LoginOverlay({ onLoggedIn }: LoginOverlayProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleGoogle = async () => {
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: typeof window !== 'undefined' ? window.location.href : '' },
    })
  }

  const handleEmail = async () => {
    if (!email || !password) { setError('Enter email and password'); return }
    setLoading(true)
    setError(null)
    const { data, error: err } = await supabase.auth.signInWithPassword({ email, password })
    if (err) { setError(err.message); setLoading(false); return }
    if (data.session) {
      localStorage.setItem('oralskop_token', data.session.access_token)
      onLoggedIn()
    }
    setLoading(false)
  }

  const handleRegister = async () => {
    if (!email || !password) { setError('Enter email and password'); return }
    setLoading(true)
    setError(null)
    try {
      await registerUser(email, password)
      await handleEmail()
    } catch {
      setError('Registration failed')
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center gap-5 px-10" style={{ backgroundColor: '#FAF9F6' }}>
      <svg width="64" height="64" viewBox="0 0 36 36" fill="none">
        <circle cx="18" cy="18" r="18" fill="#0F6E56" />
        <circle cx="18" cy="16" r="9" stroke="white" strokeWidth="2.5" fill="none" />
        <path d="M12 19 Q18 25 24 19" stroke="white" strokeWidth="2" strokeLinecap="round" fill="none" />
      </svg>
      <p className="text-2xl font-extrabold text-charcoal">OralSkop</p>
      <p className="text-gray text-sm">Dentist workspace — sign in to continue</p>

      <button
        onClick={handleGoogle}
        className="flex items-center gap-2.5 px-7 py-3.5 bg-white border border-line rounded-pill font-semibold text-charcoal text-sm shadow-sm hover:shadow-md transition-shadow"
      >
        <GoogleIcon />
        Continue with Google
      </button>

      <div className="w-80 flex items-center gap-3 text-gray text-xs">
        <div className="flex-1 h-px bg-line" />
        <span>or</span>
        <div className="flex-1 h-px bg-line" />
      </div>

      <div className="flex flex-col gap-2.5 w-80">
        <input
          type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)}
          className="w-full px-4 py-3.5 border border-line rounded-xl text-sm outline-none focus:border-teal"
        />
        <input
          type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleEmail()}
          className="w-full px-4 py-3.5 border border-line rounded-xl text-sm outline-none focus:border-teal"
        />
        {error && <p className="text-xs font-semibold px-1" style={{ color: '#D85A30' }}>{error}</p>}
        <button
          onClick={handleEmail} disabled={loading}
          className="w-full py-3.5 rounded-pill text-white font-bold text-sm disabled:opacity-60"
          style={{ backgroundColor: '#0F6E56' }}
        >
          Sign in
        </button>
        <button
          onClick={handleRegister} disabled={loading}
          className="w-full py-3 rounded-pill font-bold text-sm border-2 disabled:opacity-60"
          style={{ borderColor: '#0F6E56', color: '#0F6E56' }}
        >
          Create account
        </button>
      </div>
    </div>
  )
}

// ── Timeline SVG ──────────────────────────────────────────────────────────────
function TimelineSVG({ screenings, activeIndex, onSelect }: {
  screenings: HistoryScreening[]
  activeIndex: number
  onSelect: (i: number) => void
}) {
  if (!screenings.length) {
    return <p className="text-xs text-gray">No sessions yet</p>
  }

  const allConditions: string[] = []
  screenings.forEach((s) => {
    ;(s.condition_summary || []).forEach((c) => {
      if (!allConditions.includes(c)) allConditions.push(c)
    })
  })
  if (allConditions.length === 0) allConditions.push('—')

  const LABEL_W = 110
  const COL_W = 60
  const ROW_H = 26
  const TOP_PAD = 28
  const DOT_R = 5
  const totalW = LABEL_W + screenings.length * COL_W + 20
  const totalH = TOP_PAD + allConditions.length * ROW_H + 16

  const parts: React.ReactNode[] = []

  screenings.forEach((s, i) => {
    const x = LABEL_W + i * COL_W + COL_W / 2
    const d = new Date(s.captured_at)
    const label = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
    if (i === activeIndex) {
      parts.push(
        <rect key={`hl-${i}`} x={x - COL_W / 2 + 2} y={0} width={COL_W - 4} height={totalH} rx={6} fill="#0F6E5611" />
      )
    }
    parts.push(
      <text key={`lbl-${i}`} x={x} y={12} textAnchor="middle" fontSize={10} fill="#6B7280" fontFamily="Inter, sans-serif">
        {label}
      </text>
    )
  })

  allConditions.forEach((cond, row) => {
    const y = TOP_PAD + row * ROW_H + ROW_H / 2
    const color = CONDITION_COLORS[cond] || '#6B7280'
    const label = CONDITION_LABELS[cond] || cond

    parts.push(
      <text key={`clbl-${row}`} x={LABEL_W - 8} y={y} textAnchor="end" fontSize={10} fill="#1F2937" fontFamily="Inter, sans-serif" dominantBaseline="middle">
        {label}
      </text>
    )
    parts.push(
      <line key={`grid-${row}`} x1={LABEL_W} y1={y} x2={totalW - 10} y2={y} stroke="#E5E7EB" strokeWidth={1} />
    )

    const sessionXs = screenings
      .map((s, i) => ((s.condition_summary || []).includes(cond) ? LABEL_W + i * COL_W + COL_W / 2 : null))
      .filter((x): x is number => x !== null)

    if (sessionXs.length > 1) {
      parts.push(
        <line key={`conn-${row}`} x1={sessionXs[0]} y1={y} x2={sessionXs[sessionXs.length - 1]} y2={y} stroke={color} strokeWidth={2} strokeOpacity={0.35} />
      )
    }

    screenings.forEach((s, i) => {
      const x = LABEL_W + i * COL_W + COL_W / 2
      const present = (s.condition_summary || []).includes(cond)
      if (present) {
        parts.push(
          <circle
            key={`dot-${row}-${i}`} cx={x} cy={y} r={DOT_R}
            fill={color} stroke="white" strokeWidth={2}
            style={{ cursor: 'pointer' }}
            onClick={() => onSelect(i)}
          />
        )
      } else {
        parts.push(<circle key={`empty-${row}-${i}`} cx={x} cy={y} r={2.5} fill="#E5E7EB" />)
      }
    })
  })

  return (
    <div style={{ overflowX: 'auto', paddingBottom: 4 }}>
      <svg width={totalW} height={totalH} style={{ display: 'block' }}>
        {parts}
      </svg>
    </div>
  )
}

// ── Case sidebar card ─────────────────────────────────────────────────────────
function CaseCard({ screening, index, selected, onClick }: {
  screening: HistoryScreening
  index: number
  selected: boolean
  onClick: () => void
}) {
  const hasDetections = screening.condition_summary && screening.condition_summary.length > 0
  const date = new Date(screening.captured_at).toLocaleDateString('en-GB', {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
  })
  const thumbUrl = screening.masked_image_url || screening.photo_url

  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 rounded-xl p-3 text-left transition-colors"
      style={{
        backgroundColor: '#1F2937',
        border: `2px solid ${selected ? '#0F6E56' : 'transparent'}`,
      }}
    >
      {/* Thumbnail */}
      <div className="w-[52px] h-10 rounded-md flex-shrink-0 overflow-hidden flex items-center justify-center" style={{ backgroundColor: '#374151' }}>
        {thumbUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={thumbUrl} alt={`Case ${index + 1}`} className="w-full h-full object-cover" loading="lazy" />
        ) : (
          <div className="flex items-center gap-0.5">
            {[0, 1, 2, 3].map((t) => (
              <div key={t} className="w-2 h-3.5 bg-white rounded-sm opacity-80" />
            ))}
          </div>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="text-white text-[13px] font-semibold">Case #{index + 1}</p>
        <p className="text-[11px] mt-0.5" style={{ color: '#6B7280' }}>{date}</p>
        <p className={`text-[11px] font-semibold mt-0.5 ${hasDetections ? '' : ''}`} style={{ color: hasDetections ? '#D85A30' : '#6B7280' }}>
          {hasDetections
            ? '● ' + screening.condition_summary.slice(0, 2).map((c) => CONDITION_LABELS[c] || c).join(', ')
            : 'clear'}
        </p>
      </div>
    </button>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DentistPage() {
  const router = useRouter()
  const [authed, setAuthed] = useState(false)
  const [showLogin, setShowLogin] = useState(false)
  const [screenings, setScreenings] = useState<HistoryScreening[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [imgToggle, setImgToggle] = useState(0)
  const [toast, setToast] = useState<string | null>(null)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [profile, setProfile] = useState<UserProfile>({})
  const [mobileDetail, setMobileDetail] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [initials, setInitials] = useState('Dr')
  const [userName, setUserName] = useState('—')
  const [userEmail, setUserEmail] = useState('—')
  const dropdownRef = useRef<HTMLDivElement>(null)

  const showToast = (msg: string) => setToast(msg)

  const handleLogout = async () => {
    await supabase.auth.signOut()
    localStorage.removeItem('oralskop_token')
    localStorage.removeItem('oralskop_last_result')
    localStorage.removeItem('oralskop_profile')
    localStorage.removeItem('oralskop_dentist_profile')
    router.push('/login')
  }

  const loadHistory = useCallback(async () => {
    const token = localStorage.getItem('oralskop_token') ?? ''
    if (!token) return
    setLoadingHistory(true)
    setHistoryError(null)
    try {
      const data = await fetchHistory(token, 5)
      setScreenings(data)
      if (data.length > 0) setSelectedIndex(0)
    } catch (e) {
      setHistoryError(e instanceof Error ? e.message : 'Failed to load cases')
    } finally {
      setLoadingHistory(false)
    }
  }, [])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        setShowLogin(true)
      } else {
        localStorage.setItem('oralskop_token', session.access_token)
        setAuthed(true)
        loadHistory()
        const meta = session.user.user_metadata || {}
        const name = meta.full_name || meta.name || session.user.email?.split('@')[0] || '?'
        const inits = name.split(' ').map((w: string) => w[0]).slice(0, 2).join('').toUpperCase() || 'Dr'
        setInitials(inits)
        setUserName(name)
        setUserEmail(session.user.email || '—')
      }
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session) {
        localStorage.setItem('oralskop_token', session.access_token)
        setAuthed(true)
        setShowLogin(false)
        loadHistory()
      }
    })

    // Load profile from localStorage
    const rawProfile = localStorage.getItem('oralskop_profile')
    if (rawProfile) {
      try { setProfile(JSON.parse(rawProfile)) } catch { /* ignore */ }
    }

    return () => subscription.unsubscribe()
  }, [loadHistory])

  const selected = screenings[selectedIndex] ?? null

  const profileMeta = (() => {
    const parts: string[] = []
    if (profile.age) parts.push(`${profile.age} yrs`)
    if (profile.sex) parts.push({ M: 'Male', F: 'Female', O: 'Other' }[profile.sex] ?? profile.sex)
    parts.push(profile.smoker ? 'smoker' : 'non-smoker')
    parts.push(profile.diabetic ? 'diabetic' : 'non-diabetic')
    return parts.length > 1 ? parts.join(' · ') : '—'
  })()

  const handleSelectCase = (i: number) => {
    setSelectedIndex(i)
    setMobileDetail(true)
  }

  const handleDeleteCase = async () => {
    if (!selected) return
    const token = localStorage.getItem('oralskop_token') ?? ''
    try {
      await deleteScreening(selected.screening_id, token)
      const updated = screenings.filter((_, i) => i !== selectedIndex)
      setScreenings(updated)
      setSelectedIndex(Math.max(0, selectedIndex - 1))
      setMobileDetail(false)
      showToast('Case deleted')
    } catch (e) {
      showToast(`Error: ${e instanceof Error ? e.message : 'Delete failed'}`)
    }
  }

  return (
    <div className="h-screen overflow-hidden flex flex-col" style={{ backgroundColor: '#F4F6F8' }}>

      {/* Login overlay */}
      {showLogin && <LoginOverlay onLoggedIn={() => { setShowLogin(false); setAuthed(true); loadHistory() }} />}

      {/* Top nav */}
      <header className="bg-white border-b border-line sticky top-0 z-40 flex items-center justify-between px-4 md:px-8 h-[60px]">
        <div className="flex items-center gap-2.5">
          <Logo size="sm" />
          <span className="text-gray text-sm font-normal hidden md:inline">· Dentist workspace</span>
        </div>
        <SegmentedToggle
          options={['Patient view', 'Dentist view']}
          active={1}
          onChange={(i) => i === 0 && router.push('/guide')}
        />
        {/* Hamburger menu */}
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setDropdownOpen((v) => !v)}
            className="w-[38px] h-[38px] rounded-xl flex flex-col items-center justify-center gap-[5px] flex-shrink-0 transition-colors hover:bg-cream"
            style={{ border: '1.5px solid #E5E7EB' }}
            aria-label="Menu"
          >
            <span className="block w-[16px] h-[1.5px] rounded-full" style={{ backgroundColor: dropdownOpen ? '#0F6E56' : '#1F2937' }} />
            <span className="block w-[16px] h-[1.5px] rounded-full" style={{ backgroundColor: dropdownOpen ? '#0F6E56' : '#1F2937' }} />
            <span className="block w-[16px] h-[1.5px] rounded-full" style={{ backgroundColor: dropdownOpen ? '#0F6E56' : '#1F2937' }} />
          </button>

          {dropdownOpen && (
            <div className="absolute top-12 right-0 bg-white border border-line rounded-xl shadow-xl z-[100] overflow-hidden" style={{ minWidth: 220 }}>
              {/* User identity */}
              <div className="px-4 py-3 flex items-center gap-3 border-b border-line bg-cream">
                <div className="w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0" style={{ backgroundColor: '#0F6E56' }}>
                  {initials}
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-bold text-charcoal truncate">{userName}</p>
                  <p className="text-xs text-gray truncate">{userEmail}</p>
                </div>
              </div>
              {/* Queue count */}
              <div className="px-4 py-2.5 border-b border-line flex items-center justify-between">
                <span className="text-xs font-semibold text-gray">Cases in queue</span>
                <span className="text-xs font-bold px-2 py-0.5 rounded-pill" style={{ backgroundColor: '#FDEDE6', color: '#D85A30' }}>
                  {screenings.length}
                </span>
              </div>
              {/* Profile link */}
              <button
                onClick={() => { setDropdownOpen(false); router.push('/profile?view=dentist') }}
                className="flex items-center justify-between w-full px-4 py-3 text-sm font-semibold text-charcoal hover:bg-cream transition-colors border-b border-line"
              >
                <span>My profile</span>
                <span className="text-gray">→</span>
              </button>
              {/* Logout */}
              <button
                onClick={handleLogout}
                className="flex items-center gap-2.5 w-full px-4 py-3 text-sm hover:bg-cream transition-colors"
                style={{ color: '#D85A30' }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
                  <polyline points="16 17 21 12 16 7" />
                  <line x1="21" y1="12" x2="9" y2="12" />
                </svg>
                Log out
              </button>
            </div>
          )}
        </div>
      </header>

      {/* 3-column workspace */}
      <div
        className="flex flex-1 min-h-0"
        style={{ overflow: 'hidden' }}
      >

        {/* LEFT — Patient photos */}
        <aside
          id="left-col"
          className={`flex-shrink-0 bg-white border-r border-line flex-col overflow-y-auto ${mobileDetail ? 'hidden md:flex' : 'flex w-full md:w-[220px]'}`}
        >
          <div className="px-4 py-3 border-b border-line flex-shrink-0">
            <p className="text-xs font-bold text-charcoal uppercase tracking-wider">Patient photos</p>
            <p className="text-[11px] font-semibold mt-0.5" style={{ color: '#0F6E56' }}>
              {loadingHistory
                ? 'Loading…'
                : screenings.length === 0
                ? 'No cases yet'
                : `${screenings.length} screening${screenings.length !== 1 ? 's' : ''} · latest first`}
            </p>
          </div>

          <div className="flex flex-col gap-1.5 p-2 flex-1">
            {loadingHistory ? (
              <div className="flex items-center justify-center flex-1 min-h-[120px]">
                <div className="spinner" style={{ width: 28, height: 28, borderWidth: 3 }} />
              </div>
            ) : historyError ? (
              <p className="text-xs text-gray text-center p-4">
                Could not load cases.<br />{historyError}
              </p>
            ) : screenings.length === 0 ? (
              <p className="text-xs text-gray text-center p-4">
                No screenings found.<br />Take a photo from the patient app.
              </p>
            ) : (
              screenings.map((s, i) => (
                <CaseCard
                  key={s.screening_id}
                  screening={s}
                  index={i}
                  selected={i === selectedIndex}
                  onClick={() => handleSelectCase(i)}
                />
              ))
            )}
          </div>
        </aside>

        {/* MOBILE DETAIL — full-width scrollable view shown when a case is selected */}
        {mobileDetail && (
          <div className="md:hidden flex-1 flex flex-col overflow-hidden bg-cream">
            {/* Sticky back header */}
            <div className="flex-shrink-0 bg-white border-b border-line">
              <button
                className="flex items-center gap-2 w-full px-4 py-3 text-sm font-semibold"
                style={{ color: '#0F6E56' }}
                onClick={() => setMobileDetail(false)}
              >
                ← All cases
              </button>
            </div>

            {/* Scrollable content */}
            <div className="flex-1 overflow-y-auto">
              <div className="p-4 pb-8 flex flex-col gap-4">

                {/* Case header */}
                <div className="flex items-center justify-between">
                  <span
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-pill text-xs font-bold"
                    style={{
                      backgroundColor: selected?.escalation_triggered ? '#FDEDE6' : '#EAF4F1',
                      color: selected?.escalation_triggered ? '#D85A30' : '#0F6E56',
                    }}
                  >
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'currentColor' }} />
                    {selected?.escalation_triggered ? 'High priority' : 'Standard'}
                  </span>
                  <p className="text-sm font-extrabold text-charcoal">Case #{selectedIndex + 1}</p>
                </div>

                <p className="text-xs text-gray -mt-2">{profileMeta}</p>

                {/* Analysis image */}
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: '#0F6E56' }}>
                      Analysis
                    </p>
                    <SegmentedToggle
                      options={['Segmentation', 'Heatmap']}
                      active={imgToggle}
                      onChange={setImgToggle}
                    />
                  </div>
                  <div className="rounded-card overflow-hidden" style={{ backgroundColor: '#0A4D3C' }}>
                    {selected?.masked_image_url || selected?.photo_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={selected.masked_image_url || selected.photo_url}
                        alt="Analysis"
                        className="w-full block"
                        loading="lazy"
                      />
                    ) : (
                      <div className="h-48 flex items-center justify-center">
                        <p className="text-xs text-center p-4" style={{ color: '#5EC9A8' }}>No image available</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Original photo */}
                <div className="flex flex-col gap-2">
                  <p className="text-xs font-semibold text-gray uppercase tracking-wide">Original photo</p>
                  <div className="rounded-card overflow-hidden" style={{ backgroundColor: '#1F2937' }}>
                    {selected?.photo_url || selected?.masked_image_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={selected.photo_url || selected.masked_image_url}
                        alt="Original"
                        className="w-full block"
                        loading="lazy"
                      />
                    ) : (
                      <div className="h-32 flex items-center justify-center">
                        <p className="text-gray text-xs text-center p-4">Image unavailable</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Detections */}
                <div className="bg-white rounded-card border border-line p-4">
                  <p className="text-[10px] font-bold text-gray uppercase tracking-[0.08em] mb-3">Detections</p>
                  {!selected?.condition_summary?.length ? (
                    <p className="text-sm text-gray">No detections</p>
                  ) : (
                    <div className="flex flex-col gap-2.5">
                      {selected.condition_summary.map((c, i) => {
                        const isAlert = c.includes('lesion') || c.includes('suspicious')
                        return (
                          <div key={i} className="flex items-center gap-2 text-sm font-semibold">
                            <span
                              className="w-2 h-2 rounded-full flex-shrink-0"
                              style={{ backgroundColor: isAlert ? '#D85A30' : '#1F2937' }}
                            />
                            {CONDITION_LABELS[c] || c}
                          </div>
                        )
                      })}
                    </div>
                  )}
                  <p className="text-[11px] text-gray mt-3 leading-relaxed">Grad-CAM interpretability · you have the final say.</p>
                </div>

                {/* Actions */}
                <div className="flex flex-col gap-2">
                  <motion.button
                    whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.96 }}
                    onClick={() => showToast('✓ Response sent to patient')}
                    className="w-full flex items-center justify-center gap-2 py-3.5 rounded-pill text-white font-bold text-sm"
                    style={{ backgroundColor: '#0F6E56' }}
                  >
                    Validate &amp; reply to patient
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.96 }}
                    onClick={() => showToast('Request sent')}
                    className="w-full flex items-center justify-center gap-2 py-3.5 rounded-pill font-bold text-sm border border-line bg-white"
                    style={{ color: '#0F6E56' }}
                  >
                    Request more photos
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.96 }}
                    onClick={handleDeleteCase}
                    className="w-full flex items-center justify-center gap-2 py-3.5 rounded-pill font-bold text-sm border"
                    style={{ color: '#D85A30', borderColor: '#FECACA' }}
                  >
                    Delete case
                  </motion.button>
                </div>

                {/* Timeline */}
                <div className="bg-white border border-line rounded-xl p-4">
                  <p className="text-[10px] font-bold text-gray uppercase tracking-[0.06em] mb-3.5">
                    Condition history · all sessions
                  </p>
                  <TimelineSVG
                    screenings={screenings}
                    activeIndex={selectedIndex}
                    onSelect={handleSelectCase}
                  />
                </div>

              </div>
            </div>
          </div>
        )}

        {/* CENTER — Image comparison (desktop only) */}
        <main
          id="center-col"
          className="flex-1 flex-col overflow-y-auto p-6 gap-5 min-w-0 hidden md:flex"
        >
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-extrabold text-charcoal">
              {selected ? `Case #${selectedIndex + 1} — comparison` : 'Select a case'}
            </h2>
            <SegmentedToggle
              options={['Segmentation', 'Heatmap']}
              active={imgToggle}
              onChange={setImgToggle}
            />
          </div>

          {/* Image panels */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div className="flex flex-col gap-2">
              <p className="text-xs font-semibold text-gray uppercase tracking-wide">Original</p>
              <div
                className="rounded-card overflow-hidden flex items-center justify-center"
                style={{ backgroundColor: '#1F2937', aspectRatio: '4/3' }}
              >
                {selected?.photo_url || selected?.masked_image_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={selected.photo_url || selected.masked_image_url}
                    alt="Original"
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                ) : (
                  <p className="text-gray text-xs text-center p-4">
                    {selected ? 'Image unavailable' : 'Select a case from the left panel'}
                  </p>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: '#0F6E56' }}>
                Analysis — {imgToggle === 0 ? 'segmentation' : 'heatmap'}
              </p>
              <div
                className="rounded-card overflow-hidden flex items-center justify-center"
                style={{ backgroundColor: '#0A4D3C', aspectRatio: '4/3' }}
              >
                {selected?.masked_image_url || selected?.photo_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={selected.masked_image_url || selected.photo_url}
                    alt="Analysis"
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                ) : (
                  <p className="text-xs text-center p-4" style={{ color: '#5EC9A8' }}>—</p>
                )}
              </div>
            </div>
          </div>

          <p className="text-xs text-gray leading-relaxed">
            Highlighted zones = model attention (Grad-CAM). The full photo remains visible: you have the final say.
          </p>

          {/* Timeline */}
          <div className="bg-white border border-line rounded-xl p-4 mt-2">
            <p className="text-[10px] font-bold text-gray uppercase tracking-[0.06em] mb-3.5">
              Condition history · all sessions
            </p>
            <TimelineSVG
              screenings={screenings}
              activeIndex={selectedIndex}
              onSelect={handleSelectCase}
            />
          </div>
        </main>

        {/* RIGHT — Case details (desktop only) */}
        <aside
          id="right-col"
          className="flex-shrink-0 bg-white border-l border-line flex-col overflow-y-auto hidden md:flex"
          style={{ width: 260 }}
        >
          <div className="p-4 flex flex-col gap-4 flex-1">

            {/* Priority badge */}
            <div>
              <span
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-pill text-xs font-bold"
                style={{
                  backgroundColor: selected?.escalation_triggered ? '#FDEDE6' : '#EAF4F1',
                  color: selected?.escalation_triggered ? '#D85A30' : '#0F6E56',
                }}
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'currentColor' }} />
                {selected?.escalation_triggered ? '● High priority' : '● Standard'}
              </span>
            </div>

            {/* Profile */}
            <div>
              <p className="text-[10px] font-bold text-gray uppercase tracking-[0.08em] mb-1.5">Profile</p>
              <p className="text-[15px] font-bold text-charcoal">
                Case #{selected ? selectedIndex + 1 : '—'} · —
              </p>
              <p className="text-[13px] text-gray mt-0.5">{profileMeta}</p>
            </div>

            {/* Detections */}
            <div className="flex-1">
              <p className="text-[10px] font-bold text-gray uppercase tracking-[0.08em] mb-2">Detections</p>
              {!selected ? (
                <p className="text-sm text-gray">No case selected</p>
              ) : !selected.condition_summary || selected.condition_summary.length === 0 ? (
                <p className="text-sm text-gray">No detections</p>
              ) : (
                <div className="flex flex-col gap-2">
                  {selected.condition_summary.map((c, i) => {
                    const isAlert = c.includes('lesion') || c.includes('suspicious')
                    return (
                      <div key={i} className="flex items-center justify-between text-[13px]">
                        <div className="flex items-center gap-2 font-semibold">
                          <span
                            className="w-2 h-2 rounded-full flex-shrink-0"
                            style={{ backgroundColor: isAlert ? '#D85A30' : '#1F2937' }}
                          />
                          {CONDITION_LABELS[c] || c}
                        </div>
                        <span className="text-xs text-gray">—</span>
                      </div>
                    )
                  })}
                </div>
              )}
              <p className="text-[11px] text-gray mt-3 leading-relaxed">Calibrated scores · Grad-CAM interpretability.</p>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-2 pt-3 border-t border-line">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.96 }}
                onClick={() => showToast('✓ Response sent to patient')}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-pill text-white font-bold text-sm"
                style={{ backgroundColor: '#0F6E56' }}
              >
                Validate &amp; reply to patient
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.96 }}
                onClick={() => showToast('Request sent')}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-pill font-bold text-sm border border-line"
                style={{ color: '#0F6E56' }}
              >
                Request more photos
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.96 }}
                onClick={handleDeleteCase}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-pill font-bold text-sm border border-line"
                style={{ color: '#D85A30', borderColor: '#FECACA' }}
              >
                Delete case
              </motion.button>
            </div>

          </div>
        </aside>
      </div>

      {/* Footer bar — desktop only */}
      <div className="md:hidden"><BottomNav /></div>

      {/* Toast */}
      <AnimatePresence>
        {toast && <Toast key={toast} message={toast} onDone={() => setToast(null)} />}
      </AnimatePresence>
    </div>
  )
}
