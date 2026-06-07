import { useRouter } from 'next/router'
import { Camera, Tooth, Stethoscope, UserCircle } from '@phosphor-icons/react'

export default function BottomNav() {
  const router = useRouter()
  const isGuide = router.pathname === '/guide'
  const isPatient = router.pathname === '/patient'
  const isDentist = router.pathname === '/dentist'
  const isProfile = router.pathname === '/profile'

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-line z-40">
      <div className="max-w-[390px] mx-auto flex">
        <button
          onClick={() => router.push('/guide')}
          className="flex-1 flex flex-col items-center justify-center py-3 gap-1 focus:outline-none"
        >
          <Camera size={22} weight={isGuide ? 'fill' : 'light'} color={isGuide ? '#0F6E56' : '#6B7280'} />
          <span className="text-[10px] font-semibold" style={{ color: isGuide ? '#0F6E56' : '#6B7280' }}>Guide</span>
        </button>

        <button
          onClick={() => router.push('/patient')}
          className="flex-1 flex flex-col items-center justify-center py-3 gap-1 focus:outline-none"
        >
          <Tooth size={22} weight={isPatient ? 'fill' : 'light'} color={isPatient ? '#0F6E56' : '#6B7280'} />
          <span className="text-[10px] font-semibold" style={{ color: isPatient ? '#0F6E56' : '#6B7280' }}>Patient</span>
        </button>

        <button
          onClick={() => router.push('/dentist')}
          className="flex-1 flex flex-col items-center justify-center py-3 gap-1 focus:outline-none"
        >
          <Stethoscope size={22} weight={isDentist ? 'fill' : 'light'} color={isDentist ? '#0F6E56' : '#6B7280'} />
          <span className="text-[10px] font-semibold" style={{ color: isDentist ? '#0F6E56' : '#6B7280' }}>Dentist</span>
        </button>

        <button
          onClick={() => router.push('/profile')}
          className="flex-1 flex flex-col items-center justify-center py-3 gap-1 focus:outline-none"
        >
          <UserCircle size={22} weight={isProfile ? 'fill' : 'light'} color={isProfile ? '#0F6E56' : '#6B7280'} />
          <span className="text-[10px] font-semibold" style={{ color: isProfile ? '#0F6E56' : '#6B7280' }}>Profile</span>
        </button>
      </div>
    </nav>
  )
}
