import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { motion, AnimatePresence } from 'framer-motion'
import Logo from '@/components/Logo'
import SegmentedToggle from '@/components/SegmentedToggle'
import BottomNav from '@/components/BottomNav'
import { supabase } from '@/lib/supabase'
import { UserProfile, DentistProfile } from '@/lib/types'

type ProfileRole = 'patient' | 'dentist'

const SPECIALTY_LABELS: Record<string, string> = {
  general: 'General Dentistry',
  ortho: 'Orthodontics',
  perio: 'Periodontology',
  endo: 'Endodontics',
  oral_surgery: 'Oral Surgery',
  pedodontics: 'Pedodontics',
  prostho: 'Prosthodontics',
}

function Toggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="relative inline-block w-11 h-6 flex-shrink-0"
    >
      <span
        className="block w-full h-full rounded-pill transition-colors"
        style={{ backgroundColor: on ? '#0F6E56' : '#E5E7EB' }}
      />
      <span
        className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow transition-transform"
        style={{ transform: on ? 'translateX(20px)' : 'translateX(0)' }}
      />
    </button>
  )
}

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-3">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-charcoal">{label}</p>
        {hint && <p className="text-xs text-gray mt-0.5">{hint}</p>}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  )
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-card border border-line overflow-hidden">
      <div className="px-4 py-3 border-b border-line">
        <p className="text-xs font-bold text-charcoal uppercase tracking-wider">{title}</p>
      </div>
      <div className="divide-y divide-line">{children}</div>
    </div>
  )
}

export default function ProfilePage() {
  const router = useRouter()
  const [authed, setAuthed] = useState(false)
  const [role, setRole] = useState<ProfileRole>('patient')

  // Sync role with URL query param so _app.tsx can detect dentist view
  useEffect(() => {
    const v = router.query.view
    if (v === 'dentist') setRole('dentist')
    else if (v === 'patient') setRole('patient')
  }, [router.query.view])

  const handleRoleChange = (i: number) => {
    const next: ProfileRole = i === 0 ? 'patient' : 'dentist'
    setRole(next)
    router.push(next === 'dentist' ? '/profile?view=dentist' : '/profile', undefined, { shallow: true })
  }

  // Auth info
  const [userName, setUserName] = useState('—')
  const [userEmail, setUserEmail] = useState('—')
  const [initials, setInitials] = useState('?')

  // Patient profile
  const [patient, setPatient] = useState<UserProfile>({})

  // Dentist profile
  const [dentist, setDentist] = useState<DentistProfile>({})

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) { router.push('/login'); return }
      setAuthed(true)
      const meta = session.user.user_metadata || {}
      const name = meta.full_name || meta.name || session.user.email?.split('@')[0] || '?'
      const inits = name.split(' ').map((w: string) => w[0]).slice(0, 2).join('').toUpperCase() || '?'
      setInitials(inits)
      setUserName(name)
      setUserEmail(session.user.email || '—')
    })

    const raw = localStorage.getItem('oralskop_profile')
    if (raw) try { setPatient(JSON.parse(raw)) } catch { /* ignore */ }

    const rawD = localStorage.getItem('oralskop_dentist_profile')
    if (rawD) try { setDentist(JSON.parse(rawD)) } catch { /* ignore */ }

  }, [router])

  const savePatient = (updated: UserProfile) => {
    setPatient(updated)
    localStorage.setItem('oralskop_profile', JSON.stringify(updated))
  }

  const saveDentist = (updated: DentistProfile) => {
    setDentist(updated)
    localStorage.setItem('oralskop_dentist_profile', JSON.stringify(updated))
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    localStorage.removeItem('oralskop_token')
    localStorage.removeItem('oralskop_last_result')
    localStorage.removeItem('oralskop_profile')
    localStorage.removeItem('oralskop_dentist_profile')
    router.push('/login')
  }

  const inputCls = "text-sm text-right border border-line rounded-xl px-3 py-1.5 outline-none focus:border-teal bg-cream text-charcoal"

  if (!authed) return null

  const riskItems = [
    patient.smoker && 'Smoking raises gum disease and oral cancer risk.',
    patient.diabetic && 'Diabetes is linked to higher rates of periodontitis.',
  ].filter(Boolean) as string[]

  const isDentistDesktop = role === 'dentist'
  const maxW = isDentistDesktop ? 'max-w-2xl' : 'max-w-[390px]'

  return (
    <div className="min-h-screen bg-cream flex flex-col">

      {/* Header */}
      <header className="bg-white border-b border-line sticky top-0 z-40">
        <div className={`${maxW} mx-auto px-4 h-14 flex items-center justify-between`}>
          <Logo size="sm" />
          <SegmentedToggle
            options={['Patient', 'Dentist']}
            active={role === 'patient' ? 0 : 1}
            onChange={handleRoleChange}
          />
          <div style={{ width: 38 }} />
        </div>
      </header>

      <main className={`flex-1 ${maxW} mx-auto w-full px-4 pt-5 pb-28`}>
        <AnimatePresence mode="wait">
          <motion.div
            key={role}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.25 }}
            className="flex flex-col gap-4"
          >

            {/* Identity card */}
            <div className="bg-white rounded-card border border-line p-4 flex items-center gap-4">
              <div
                className="w-14 h-14 rounded-full flex items-center justify-center text-white text-xl font-extrabold flex-shrink-0"
                style={{ backgroundColor: '#0F6E56' }}
              >
                {initials}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-extrabold text-charcoal text-base truncate">{userName}</p>
                <p className="text-xs text-gray truncate mt-0.5">{userEmail}</p>
                <span
                  className="inline-block text-[11px] font-bold px-2.5 py-0.5 rounded-pill mt-1.5"
                  style={
                    role === 'dentist'
                      ? { backgroundColor: '#E6F4EF', color: '#0F6E56' }
                      : { backgroundColor: '#EEF2FF', color: '#4F46E5' }
                  }
                >
                  {role === 'dentist' ? 'Dental Professional' : 'Patient'}
                </span>
              </div>
            </div>

            {role === 'patient' ? (
              <>
                {/* Health info */}
                <Card title="Health info">
                  <Row label="Age">
                    <input
                      type="number" min={1} max={120}
                      value={patient.age ?? ''}
                      onChange={(e) => savePatient({ ...patient, age: e.target.value })}
                      placeholder="—"
                      className={inputCls}
                      style={{ width: 80 }}
                    />
                  </Row>
                  <Row label="Sex">
                    <select
                      value={patient.sex ?? ''}
                      onChange={(e) => savePatient({ ...patient, sex: e.target.value })}
                      className={inputCls}
                      style={{ width: 110 }}
                    >
                      <option value="">—</option>
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                      <option value="O">Other</option>
                    </select>
                  </Row>
                  <Row label="Smoker" hint="Affects gum health risk">
                    <Toggle on={!!patient.smoker} onToggle={() => savePatient({ ...patient, smoker: !patient.smoker })} />
                  </Row>
                  <Row label="Diabetic" hint="Raises periodontal risk">
                    <Toggle on={!!patient.diabetic} onToggle={() => savePatient({ ...patient, diabetic: !patient.diabetic })} />
                  </Row>
                </Card>

                {/* Risk summary */}
                <div
                  className="rounded-card p-4 border"
                  style={{ backgroundColor: riskItems.length > 0 ? '#FDEDE6' : '#F0FBF7', borderColor: riskItems.length > 0 ? '#FECACA' : '#A7F3D0' }}
                >
                  <p
                    className="text-xs font-bold uppercase tracking-wide mb-1.5"
                    style={{ color: riskItems.length > 0 ? '#D85A30' : '#0F6E56' }}
                  >
                    {riskItems.length > 0 ? 'Elevated risk factors' : 'No elevated risk factors'}
                  </p>
                  <p className="text-sm text-charcoal leading-relaxed">
                    {riskItems.length > 0
                      ? riskItems.join(' ')
                      : 'Your profile shows no known risk factors. Keep up with regular check-ups.'}
                  </p>
                </div>

                {/* Screening history shortcut */}
                <button
                  onClick={() => router.push('/patient')}
                  className="bg-white rounded-card border border-line w-full flex items-center justify-between px-4 py-3.5 hover:bg-cream transition-colors"
                >
                  <div className="text-left">
                    <p className="text-sm font-bold text-charcoal">Screening history</p>
                    <p className="text-xs text-gray mt-0.5">View your past analyses</p>
                  </div>
                  <span className="text-gray text-lg">→</span>
                </button>
              </>
            ) : (
              <>
                {/* Professional info */}
                <Card title="Professional info">
                  <Row label="Specialty">
                    <select
                      value={dentist.specialty ?? ''}
                      onChange={(e) => saveDentist({ ...dentist, specialty: e.target.value })}
                      className={inputCls}
                      style={{ width: 170 }}
                    >
                      <option value="">—</option>
                      {Object.entries(SPECIALTY_LABELS).map(([val, label]) => (
                        <option key={val} value={val}>{label}</option>
                      ))}
                    </select>
                  </Row>
                  <Row label="Clinic">
                    <input
                      type="text"
                      value={dentist.clinic ?? ''}
                      onChange={(e) => saveDentist({ ...dentist, clinic: e.target.value })}
                      placeholder="Clinic name"
                      className={inputCls}
                      style={{ width: 160 }}
                    />
                  </Row>
                  <Row label="License #">
                    <input
                      type="text"
                      value={dentist.license ?? ''}
                      onChange={(e) => saveDentist({ ...dentist, license: e.target.value })}
                      placeholder="—"
                      className={inputCls}
                      style={{ width: 120 }}
                    />
                  </Row>
                  <Row label="Experience">
                    <select
                      value={dentist.experience ?? ''}
                      onChange={(e) => saveDentist({ ...dentist, experience: e.target.value })}
                      className={inputCls}
                      style={{ width: 130 }}
                    >
                      <option value="">—</option>
                      <option value="0-2">0–2 years</option>
                      <option value="3-5">3–5 years</option>
                      <option value="6-10">6–10 years</option>
                      <option value="10+">10+ years</option>
                    </select>
                  </Row>
                  <Row label="Accept new cases" hint="Patients can send you photos">
                    <Toggle
                      on={!!dentist.available}
                      onToggle={() => saveDentist({ ...dentist, available: !dentist.available })}
                    />
                  </Row>
                </Card>

                {/* Profile completion hint */}
                {(!dentist.specialty || !dentist.clinic) && (
                  <div className="rounded-card p-4 border" style={{ backgroundColor: '#FFFBEB', borderColor: '#FDE68A' }}>
                    <p className="text-xs font-bold uppercase tracking-wide mb-1" style={{ color: '#92400E' }}>Complete your profile</p>
                    <p className="text-sm" style={{ color: '#78350F' }}>
                      Add your specialty and clinic so patients can identify their dentist.
                    </p>
                  </div>
                )}

                {/* Workspace shortcut */}
                <button
                  onClick={() => router.push('/dentist')}
                  className="bg-white rounded-card border border-line w-full flex items-center justify-between px-4 py-3.5 hover:bg-cream transition-colors"
                >
                  <div className="text-left">
                    <p className="text-sm font-bold text-charcoal">Dentist workspace</p>
                    <p className="text-xs text-gray mt-0.5">Review and manage patient cases</p>
                  </div>
                  <span className="text-gray text-lg">→</span>
                </button>
              </>
            )}

            {/* Logout */}
            <button
              onClick={handleLogout}
              className="w-full py-3.5 rounded-card border flex items-center justify-center gap-2 text-sm font-bold"
              style={{ color: '#D85A30', borderColor: '#FECACA', backgroundColor: '#FFF5F2' }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
              Log out
            </button>

          </motion.div>
        </AnimatePresence>
      </main>

      <BottomNav />
    </div>
  )
}
