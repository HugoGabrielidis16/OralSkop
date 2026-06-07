import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/router'
import Logo from './Logo'
import SegmentedToggle from './SegmentedToggle'
import { supabase } from '@/lib/supabase'

interface HeaderProps {
  activeRole: 'patient' | 'dentist'
  onRoleChange: (role: 'patient' | 'dentist') => void
}

export default function Header({ activeRole, onRoleChange }: HeaderProps) {
  const router = useRouter()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [initials, setInitials] = useState('?')
  const [userName, setUserName] = useState('—')
  const [userEmail, setUserEmail] = useState('—')
  const [lang, setLang] = useState<'en' | 'fr'>('en')
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        const meta = session.user.user_metadata || {}
        const name = meta.full_name || meta.name || session.user.email?.split('@')[0] || '?'
        const inits = name.split(' ').map((w: string) => w[0]).slice(0, 2).join('').toUpperCase() || '?'
        setInitials(inits)
        setUserName(name)
        setUserEmail(session.user.email || '—')
      }
    })

    const saved = localStorage.getItem('oralskop_lang')
    if (saved === 'en' || saved === 'fr') setLang(saved)
  }, [])

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const toggleLang = () => {
    const next = lang === 'en' ? 'fr' : 'en'
    setLang(next)
    localStorage.setItem('oralskop_lang', next)
    window.dispatchEvent(new CustomEvent('oralskop:langChange', { detail: next }))
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    localStorage.removeItem('oralskop_token')
    localStorage.removeItem('oralskop_last_result')
    localStorage.removeItem('oralskop_profile')
    localStorage.removeItem('oralskop_dentist_profile')
    router.push('/login')
  }

  const handleRoleChange = (i: number) => {
    const role = i === 0 ? 'patient' : 'dentist'
    onRoleChange(role)
    if (role === 'patient') router.push('/guide')
    else router.push('/dentist')
  }

  return (
    <header className="bg-white border-b border-line sticky top-0 z-40">
      <div className="max-w-[390px] mx-auto px-4 h-14 flex items-center justify-between gap-2">
        <Logo size="sm" />

        <div className="hidden">
          <SegmentedToggle
            options={['Patient', 'Dentist']}
            active={activeRole === 'patient' ? 0 : 1}
            onChange={handleRoleChange}
          />
        </div>

        {/* Right side: lang toggle + menu */}
        <div className="flex items-center gap-2">
          {/* EN / FR toggle */}
          <button
            onClick={toggleLang}
            className="flex items-center gap-1 px-2.5 py-1 rounded-pill border border-line bg-cream text-xs font-bold"
          >
            <span style={{ color: lang === 'en' ? '#0F6E56' : '#9CA3AF' }}>EN</span>
            <span className="text-gray">/</span>
            <span style={{ color: lang === 'fr' ? '#0F6E56' : '#9CA3AF' }}>FR</span>
          </button>

          {/* Hamburger + dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setDropdownOpen((v) => !v)}
              className="w-[38px] h-[38px] rounded-xl flex flex-col items-center justify-center gap-[5px] flex-shrink-0 transition-colors hover:bg-cream"
              style={{ border: '1.5px solid #E5E7EB' }}
              aria-label="Menu"
            >
              <span className="block w-[16px] h-[1.5px] rounded-full transition-all" style={{ backgroundColor: dropdownOpen ? '#0F6E56' : '#1F2937' }} />
              <span className="block w-[16px] h-[1.5px] rounded-full transition-all" style={{ backgroundColor: dropdownOpen ? '#0F6E56' : '#1F2937' }} />
              <span className="block w-[16px] h-[1.5px] rounded-full transition-all" style={{ backgroundColor: dropdownOpen ? '#0F6E56' : '#1F2937' }} />
            </button>

            {dropdownOpen && (
              <div
                className="absolute top-12 right-0 bg-white border border-line rounded-xl shadow-xl z-[100] overflow-hidden"
                style={{ minWidth: 210 }}
              >
                {/* User identity */}
                <div className="px-4 py-3 flex items-center gap-3 border-b border-line bg-cream">
                  <div
                    className="w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0"
                    style={{ backgroundColor: '#0F6E56' }}
                  >
                    {initials}
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-bold text-charcoal truncate">{userName}</p>
                    <p className="text-xs text-gray truncate">{userEmail}</p>
                  </div>
                </div>

                {/* Profile link */}
                <button
                  onClick={() => { setDropdownOpen(false); router.push('/profile') }}
                  className="flex items-center justify-between w-full px-4 py-3 text-sm font-semibold text-charcoal hover:bg-cream transition-colors border-b border-line"
                >
                  <span>My profile</span>
                  <span className="text-gray text-base">→</span>
                </button>

                {/* Logout */}
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-2.5 w-full px-4 py-3 text-sm text-left hover:bg-cream transition-colors"
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
        </div>
      </div>
    </header>
  )
}
