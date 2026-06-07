import { useRef, useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, ArrowRight, VideoCamera } from '@phosphor-icons/react'
import Header from '@/components/Header'
import ViewfinderCard from '@/components/ViewfinderCard'
import BottomNav from '@/components/BottomNav'
import Toast from '@/components/Toast'
import { supabase } from '@/lib/supabase'
import { analyzePhoto, mergeResults } from '@/lib/api'
import { ScreeningResult } from '@/lib/types'

type Lang = 'en' | 'fr'

const T = {
  en: {
    title: 'Frame your photo correctly',
    subtitle: 'For a reliable analysis, follow these markers.',
    tips: [
      { n: 1, title: 'Good lighting', text: 'Face a source of light' },
      { n: 2, title: 'Open mouth', text: 'Show teeth and gums clearly' },
      { n: 3, title: 'Stay sharp', text: 'Hold the phone steady, no blur' },
    ],
    takePhoto: 'Upload the photo',
    useCamera: 'Use your camera',
    align: 'Align your mouth in the frame',
    capture: 'Capture',
    cancel: 'Cancel',
    analysing: 'Analysing your photo…',
    analysingN: (i: number, n: number) => `Analysing photo ${i} of ${n}…`,
    seconds: 'This takes a few seconds',
    notLoggedIn: 'Not logged in',
  },
  fr: {
    title: 'Cadrez votre photo correctement',
    subtitle: 'Pour une analyse fiable, suivez ces repères.',
    tips: [
      { n: 1, title: 'Bon éclairage', text: 'Faites face à une source de lumière' },
      { n: 2, title: 'Bouche ouverte', text: 'Montrez clairement les dents et les gencives' },
      { n: 3, title: 'Restez net', text: 'Tenez le téléphone stable, sans flou' },
    ],
    takePhoto: 'Charger une photo',
    useCamera: 'Utiliser la caméra',
    align: 'Alignez votre bouche dans le cadre',
    capture: 'Capturer',
    cancel: 'Annuler',
    analysing: 'Analyse de votre photo…',
    analysingN: (i: number, n: number) => `Analyse de la photo ${i} sur ${n}…`,
    seconds: 'Cela prend quelques secondes',
    notLoggedIn: 'Non connecté',
  },
}

export default function GuidePage() {
  const router = useRouter()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [loading, setLoading] = useState(false)
  const [loadingText, setLoadingText] = useState('')
  const [toast, setToast] = useState<string | null>(null)
  const [authed, setAuthed] = useState(false)
  const [lang, setLang] = useState<Lang>('en')
  const [cameraOpen, setCameraOpen] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)

  const t = T[lang]

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) router.push('/login')
      else setAuthed(true)
    })
    const saved = localStorage.getItem('oralskop_lang') as Lang | null
    if (saved === 'en' || saved === 'fr') setLang(saved)

    const onLangChange = (e: Event) => {
      const next = (e as CustomEvent<Lang>).detail
      if (next === 'en' || next === 'fr') setLang(next)
    }
    window.addEventListener('oralskop:langChange', onLangChange)
    return () => window.removeEventListener('oralskop:langChange', onLangChange)
  }, [router])

  useEffect(() => {
    if (cameraOpen && videoRef.current && stream) {
      videoRef.current.srcObject = stream
      videoRef.current.play().catch(() => {})
    }
  }, [cameraOpen, stream])

  // Stop stream on unmount
  useEffect(() => {
    return () => { stream?.getTracks().forEach((tr) => tr.stop()) }
  }, [stream])

  const showToast = (msg: string) => setToast(msg)

  const processFiles = async (files: File[]) => {
    const token = localStorage.getItem('oralskop_token') ?? ''
    if (!token) { showToast(t.notLoggedIn); return }

    setLoading(true)
    const allResults: ScreeningResult[] = []

    try {
      for (let i = 0; i < files.length; i++) {
        setLoadingText(files.length > 1 ? t.analysingN(i + 1, files.length) : t.analysing)
        const result = await analyzePhoto(files[i], token)
        allResults.push(result)
      }
      const merged = mergeResults(allResults)
      localStorage.setItem('oralskop_last_result', JSON.stringify(merged))
      router.push('/patient')
    } catch (err) {
      console.error(err)
      showToast(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
      if (cameraInputRef.current) cameraInputRef.current.value = ''
    }
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    if (!files.length) return
    await processFiles(files)
  }

  const openCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      setStream(s)
      setCameraOpen(true)
    } catch {
      // Fallback: trigger file input with capture attribute
      cameraInputRef.current?.click()
    }
  }

  const closeCamera = () => {
    stream?.getTracks().forEach((tr) => tr.stop())
    setStream(null)
    setCameraOpen(false)
  }

  const capturePhoto = () => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')?.drawImage(video, 0, 0)
    canvas.toBlob(async (blob) => {
      if (!blob) return
      closeCamera()
      await processFiles([new File([blob], 'capture.jpg', { type: 'image/jpeg' })])
    }, 'image/jpeg', 0.92)
  }

  if (!authed) return null

  return (
    <div className="min-h-screen bg-cream flex flex-col">
      <Header activeRole="patient" onRoleChange={(role) => role === 'dentist' && router.push('/dentist')} />

      {/* Loading overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-6"
            style={{ backgroundColor: '#FAF9F6' }}
          >
            <svg width="56" height="56" viewBox="0 0 36 36" fill="none">
              <circle cx="18" cy="18" r="18" fill="#0F6E56" />
              <circle cx="18" cy="16" r="9" stroke="white" strokeWidth="2.5" fill="none" />
              <path d="M12 19 Q18 25 24 19" stroke="white" strokeWidth="2" strokeLinecap="round" fill="none" />
            </svg>
            <div className="spinner" />
            <div className="text-center">
              <p className="text-charcoal font-bold text-lg">{loadingText}</p>
              <p className="text-gray text-sm mt-1">{t.seconds}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="flex-1 max-w-[390px] mx-auto w-full px-5 pt-6 pb-28">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="flex flex-col gap-6"
        >
          {/* Title — hidden when camera is live */}
          {!cameraOpen && (
            <div>
              <h1 className="text-[22px] font-extrabold text-charcoal">{t.title}</h1>
              <p className="text-gray text-sm mt-1">{t.subtitle}</p>
            </div>
          )}

          {/* Viewfinder — shows live feed when camera is open */}
          <ViewfinderCard videoRef={videoRef} stream={stream} />

          {/* Tips — hidden when camera is live */}
          {!cameraOpen && (
            <div className="flex flex-col gap-4">
              {t.tips.map((tip) => (
                <div key={tip.n} className="flex items-start gap-3.5">
                  <span
                    className="w-7 h-7 min-w-[28px] rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0"
                    style={{ backgroundColor: '#0F6E56' }}
                  >
                    {tip.n}
                  </span>
                  <div className="pt-0.5">
                    <strong className="block text-sm font-bold text-charcoal">{tip.title}</strong>
                    <span className="text-xs text-gray">{tip.text}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* CTA buttons */}
          <AnimatePresence mode="wait">
            {cameraOpen ? (
              <motion.div
                key="camera-active"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 4 }}
                transition={{ duration: 0.2 }}
                className="flex flex-col gap-3"
              >
                {/* Capture button */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={capturePhoto}
                  disabled={loading}
                  className="w-full flex items-center justify-center gap-3 py-4 rounded-pill text-white font-bold text-base disabled:opacity-60"
                  style={{ backgroundColor: '#0F6E56' }}
                >
                  {/* Shutter icon */}
                  <span
                    className="w-6 h-6 rounded-full border-2 border-white flex items-center justify-center flex-shrink-0"
                  >
                    <span className="w-3 h-3 rounded-full bg-white" />
                  </span>
                  {t.capture}
                </motion.button>

                {/* Cancel */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={closeCamera}
                  className="w-full flex items-center justify-center py-3 rounded-pill font-semibold text-sm border border-line bg-white"
                  style={{ color: '#6B7280' }}
                >
                  {t.cancel}
                </motion.button>
              </motion.div>
            ) : (
              <motion.div
                key="camera-idle"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 4 }}
                transition={{ duration: 0.2 }}
                className="flex flex-col gap-3"
              >
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={openCamera}
                  disabled={loading}
                  className="w-full flex items-center justify-center gap-2 py-4 rounded-pill text-white font-bold text-base disabled:opacity-60"
                  style={{ backgroundColor: '#0F6E56' }}
                >
                  <VideoCamera size={22} weight="light" />
                  {t.useCamera}
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  className="w-full flex items-center justify-center gap-2 py-4 rounded-pill font-bold text-base disabled:opacity-60 border border-line bg-white"
                  style={{ color: '#1F2937' }}
                >
                  <Camera size={22} weight="light" />
                  {t.takePhoto}
                  <ArrowRight size={18} weight="light" />
                </motion.button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Hidden file inputs */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleFileChange}
          />
        </motion.div>
      </main>

      <BottomNav />

      <AnimatePresence>
        {toast && <Toast key={toast} message={toast} onDone={() => setToast(null)} />}
      </AnimatePresence>
    </div>
  )
}
