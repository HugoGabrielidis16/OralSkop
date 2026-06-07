import { useEffect, useState, useRef } from 'react'
import { useRouter } from 'next/router'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ChatCircle } from '@phosphor-icons/react'
import { marked } from 'marked'
import Header from '@/components/Header'
import BottomNav from '@/components/BottomNav'
import DentistModal from '@/components/DentistModal'
import Toast from '@/components/Toast'
import { supabase } from '@/lib/supabase'
import { sendChatMessage } from '@/lib/api'
import { ScreeningResult, Detection } from '@/lib/types'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const CONDITION_LABELS: Record<string, string> = {
  cavity: 'Cavity',
  caries: 'Caries',
  abrasion: 'Abrasion',
  gingivitis: 'Localised gingivitis',
  tartar: 'Tartar build-up',
  lesion_suspicious: 'Area to monitor',
  crown: 'Crown detected',
}

const SEVERITY_LABELS: Record<string, string> = {
  low: 'Mild',
  moderate: 'Moderate severity',
  high: 'High severity',
}

// Group detections by condition+severity
function groupDetections(detections: Detection[]) {
  const grouped: Record<string, Detection & { count: number }> = {}
  detections.forEach((d) => {
    const key = `${d.condition}|${d.severity}`
    if (!grouped[key]) grouped[key] = { ...d, count: 0 }
    grouped[key].count++
  })
  return Object.values(grouped)
}

// ── Toast ────────────────────────────────────────────────────────────────────

// ── Chatbot Panel ────────────────────────────────────────────────────────────
interface ChatMsg { role: 'user' | 'assistant'; content: string }

interface ChatPanelProps {
  open: boolean
  onClose: () => void
  result: ScreeningResult
  token: string
}

function ChatPanel({ open, onClose, result, token }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [typing, setTyping] = useState(false)
  const [initialised, setInitialised] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const segmentation = {
    detections: (result.detections || []).map((d) => ({
      class_name: d.condition,
      confidence: d.confidence,
      severity: d.severity,
      bbox: d.box_coordinates || [],
    })),
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    if (open && !initialised) {
      setInitialised(true)
      autoSummarise()
    }
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    scrollToBottom()
  }, [messages, typing])

  const autoSummarise = async () => {
    const AUTO_MSG = 'Summarise what you found in my scan in 2-3 sentences.'
    const userMsg: ChatMsg = { role: 'user', content: AUTO_MSG }
    const history = [userMsg]
    setTyping(true)
    try {
      const reply = await sendChatMessage(history, segmentation, token)
      const assistantMsg: ChatMsg = { role: 'assistant', content: reply }
      setMessages([assistantMsg])
    } catch (e) {
      setMessages([{ role: 'assistant', content: `Couldn't load scan summary. (${e instanceof Error ? e.message : 'error'})` }])
    } finally {
      setTyping(false)
    }
  }

  const sendMsg = async () => {
    const text = input.trim()
    if (!text || typing) return
    setInput('')
    const userMsg: ChatMsg = { role: 'user', content: text }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setTyping(true)
    try {
      const apiHistory = [
        { role: 'user', content: 'Summarise what you found in my scan in 2-3 sentences.' },
        ...newMessages.map((m) => ({ role: m.role, content: m.content })),
      ]
      const reply = await sendChatMessage(apiHistory, segmentation, token)
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }])
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Sorry, I couldn't reach the AI assistant. (${e instanceof Error ? e.message : 'error'})` }])
    } finally {
      setTyping(false)
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="chat-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-[390]"
            style={{ background: 'rgba(0,0,0,0.3)' }}
          />
          <motion.div
            key="chat-panel"
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', stiffness: 350, damping: 40 }}
            className="fixed bottom-0 left-0 right-0 z-[400] bg-white flex flex-col max-w-[390px] mx-auto"
            style={{ height: '70vh', borderRadius: '20px 20px 0 0', boxShadow: '0 -4px 24px rgba(0,0,0,0.12)' }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-line flex-shrink-0">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-white font-extrabold text-sm" style={{ backgroundColor: '#0F6E56' }}>O</div>
                <div>
                  <p className="font-bold text-[15px] text-charcoal">OralSkop Assistant</p>
                  <p className="text-[11px] font-semibold" style={{ color: '#0F6E56' }}>● Online · Ask about your scan</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-full flex items-center justify-center"
                style={{ backgroundColor: '#E5E7EB' }}
              >
                <X size={16} color="#1F2937" />
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-2.5">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`max-w-[85%] px-3.5 py-2.5 rounded-2xl text-[13px] leading-relaxed ${
                    msg.role === 'user'
                      ? 'self-end text-white rounded-br-[4px]'
                      : 'self-start text-charcoal rounded-bl-[4px] max-w-[90%]'
                  }`}
                  style={{
                    backgroundColor: msg.role === 'user' ? '#0F6E56' : '#EAF4F1',
                  }}
                >
                  {msg.role === 'assistant' ? (
                    <div
                      className="prose-chat"
                      dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) as string }}
                    />
                  ) : (
                    msg.content
                  )}
                </div>
              ))}
              {typing && (
                <div
                  className="self-start px-4 py-2.5 rounded-2xl text-[13px] text-gray"
                  style={{ backgroundColor: '#E5E7EB' }}
                >
                  Thinking…
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="flex gap-2 px-4 py-3 border-t border-line flex-shrink-0">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMsg()}
                placeholder="Ask about your scan…"
                className="flex-1 px-3.5 py-2.5 border border-line rounded-pill text-sm outline-none"
                style={{ fontFamily: 'Inter, sans-serif' }}
              />
              <button
                onClick={sendMsg}
                disabled={!input.trim() || typing}
                className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 disabled:opacity-50"
                style={{ backgroundColor: '#0F6E56' }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" fill="white" stroke="white" strokeWidth="1.5" />
                </svg>
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// ── Findings Drawer ───────────────────────────────────────────────────────────
interface FindingsDrawerProps {
  open: boolean
  onOpen: () => void
  onClose: () => void
  detections: Detection[]
}

function FindingsDrawer({ open, onOpen, onClose, detections }: FindingsDrawerProps) {
  const grouped = groupDetections(detections)

  return (
    <>
      {/* Handle */}
      {detections.length > 0 && (
        <button
          onClick={onOpen}
          className="fixed right-0 z-40 flex flex-col items-center text-white rounded-l-[10px] px-2 py-3 gap-1.5 cursor-pointer"
          style={{
            top: '50%',
            transform: 'translateY(-50%)',
            backgroundColor: '#0F6E56',
            boxShadow: '-2px 0 12px rgba(0,0,0,0.15)',
            writingMode: 'vertical-rl',
          }}
        >
          <span
            className="w-5 h-5 rounded-full flex items-center justify-center text-white font-bold text-[11px]"
            style={{ backgroundColor: '#D85A30', writingMode: 'horizontal-tb' }}
          >
            {grouped.length}
          </span>
          <span
            className="text-xs font-bold tracking-wide"
            style={{ transform: 'rotate(180deg)', writingMode: 'vertical-rl' }}
          >
            Findings
          </span>
        </button>
      )}

      {/* Overlay */}
      <AnimatePresence>
        {open && (
          <>
            <motion.div
              key="findings-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
              className="fixed inset-0 z-[45]"
              style={{ background: 'rgba(0,0,0,0.35)' }}
            />
            <motion.div
              key="findings-drawer"
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', stiffness: 350, damping: 40 }}
              className="fixed top-0 right-0 h-full z-[46] flex flex-col"
              style={{
                width: '85%',
                maxWidth: 340,
                backgroundColor: '#FAF9F6',
                boxShadow: '-4px 0 24px rgba(0,0,0,0.15)',
              }}
            >
              {/* Drawer header */}
              <div className="flex items-center justify-between px-5 py-5 border-b border-line bg-white flex-shrink-0">
                <span className="text-base font-bold text-charcoal">Findings</span>
                <button
                  onClick={onClose}
                  className="w-8 h-8 rounded-full flex items-center justify-center text-charcoal font-bold text-lg"
                  style={{ backgroundColor: '#E5E7EB' }}
                >
                  ×
                </button>
              </div>

              {/* Drawer body */}
              <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
                {grouped.length === 0 ? (
                  <p className="text-sm text-gray text-center mt-4">No detections found.</p>
                ) : (
                  grouped.map((d, i) => {
                    const isSuspicious = d.condition === 'lesion_suspicious'
                    const label = CONDITION_LABELS[d.condition] || d.condition
                    const sevLabel = SEVERITY_LABELS[d.severity] || d.severity
                    return (
                      <div
                        key={i}
                        className="rounded-xl p-4"
                        style={{
                          backgroundColor: isSuspicious ? '#FDEDE6' : '#ffffff',
                          borderLeft: '4px solid #D85A30',
                          boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                        }}
                      >
                        <div className="flex items-center gap-1.5">
                          <span className="font-bold text-sm text-charcoal">{label}</span>
                          {d.count > 1 && (
                            <span className="text-[11px] font-bold px-1.5 py-0.5 rounded-pill" style={{ backgroundColor: '#E5E7EB', color: '#1F2937' }}>
                              ×{d.count}
                            </span>
                          )}
                        </div>
                        <div className="mt-1.5 flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: '#D85A30' }} />
                          <span className="text-xs font-semibold" style={{ color: '#D85A30' }}>{sevLabel}</span>
                          <span className="text-xs text-gray ml-1">({Math.round(d.confidence * 100)}%)</span>
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function PatientPage() {
  const router = useRouter()
  const [result, setResult] = useState<ScreeningResult | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [chatNotifDismissed, setChatNotifDismissed] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [token, setToken] = useState('')
  const [authed, setAuthed] = useState(false)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        router.push('/login')
        return
      }
      setAuthed(true)
      const t = session.access_token
      setToken(t)
      localStorage.setItem('oralskop_token', t)
    })
  }, [router])

  useEffect(() => {
    const raw = localStorage.getItem('oralskop_last_result')
    if (raw) {
      try {
        const parsed: ScreeningResult = JSON.parse(raw)
        setResult(parsed)
      } catch { /* ignore */ }
    }
  }, [])

  const showToast = (msg: string) => setToast(msg)

  const subtitleText = result
    ? result.detections.length === 0
      ? 'All clear — keep monitoring'
      : `${result.detections.length} area${result.detections.length > 1 ? 's' : ''} to monitor detected`
    : ''

  const subtitleColor = result?.detections.length === 0 ? '#0F6E56' : '#D85A30'

  if (!authed) return null

  return (
    <div className="min-h-screen bg-cream flex flex-col">
      <Header activeRole="patient" onRoleChange={(role) => role === 'dentist' && router.push('/dentist')} />

      <main className="flex-1 max-w-[390px] mx-auto w-full px-5 pt-6 pb-32">
        {!result ? (
          <div className="flex flex-col gap-4">
            {[48, 32, 160, 80, 80].map((h, i) => (
              <div key={i} className="rounded-card bg-line animate-pulse" style={{ height: h }} />
            ))}
          </div>
        ) : result.detections.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="flex flex-col items-center justify-center gap-5 pt-16 text-center"
          >
            <div className="w-16 h-16 rounded-full flex items-center justify-center" style={{ backgroundColor: '#E6F4EF' }}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#0F6E56" strokeWidth="2" strokeLinecap="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
            </div>
            <div>
              <p className="text-lg font-extrabold text-charcoal">No findings detected</p>
              <p className="text-sm text-gray mt-1 max-w-[240px]">
                The photo may be unclear. Try taking another picture to analyze.
              </p>
            </div>
            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={() => router.push('/guide')}
              className="flex items-center gap-2 px-6 py-3.5 rounded-pill text-white font-bold text-sm"
              style={{ backgroundColor: '#0F6E56' }}
            >
              Take another photo
            </motion.button>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="flex flex-col gap-5"
          >
            {/* Title */}
            <div>
              <h1 className="text-[22px] font-extrabold text-charcoal">Your dental arch</h1>
              <p className="text-sm font-semibold mt-1" style={{ color: subtitleColor }}>{subtitleText}</p>
            </div>

            {/* Masked image(s) */}
            {result.masked_image_urls && result.masked_image_urls.length > 1 ? (
              <div
                className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide rounded-card"
                style={{ scrollSnapType: 'x mandatory' }}
              >
                {result.masked_image_urls.map((url, i) => (
                  <div
                    key={i}
                    className="flex-shrink-0 relative rounded-[10px] overflow-hidden"
                    style={{ flex: '0 0 calc(85vw - 40px)', scrollSnapAlign: 'start', backgroundColor: '#1F2937' }}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={url} alt={`Photo ${i + 1} overlay`} className="w-full block" loading="lazy" />
                    <div className="absolute bottom-2 right-2 text-white text-[11px] px-2 py-0.5 rounded-full" style={{ background: 'rgba(0,0,0,0.55)' }}>
                      Photo {i + 1}
                    </div>
                  </div>
                ))}
              </div>
            ) : result.masked_image_url ? (
              <div className="rounded-card overflow-hidden" style={{ backgroundColor: '#1F2937' }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.masked_image_url} alt="Analysis overlay" className="w-full block" />
              </div>
            ) : null}

            {/* Escalation banner */}
            {result.escalation_triggered && (
              <div className="flex items-start gap-3 rounded-card px-4 py-3" style={{ backgroundColor: '#FDEDE6' }}>
                <p className="text-sm font-semibold" style={{ color: '#D85A30' }}>
                  One or more findings require prompt professional attention. Please contact a dentist soon.
                </p>
              </div>
            )}

            {/* Send to dentist */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => setModalOpen(true)}
              className="w-full py-4 rounded-pill text-white font-bold text-base"
              style={{ backgroundColor: '#0F6E56' }}
            >
              Send to a dentist
            </motion.button>

            {/* Footer */}
            <p className="text-center text-xs text-gray pb-2 italic">
              Saved for follow-up · comparable over time
            </p>
          </motion.div>
        )}
      </main>

      <BottomNav />

      {/* Findings drawer */}
      {result && (
        <FindingsDrawer
          open={drawerOpen}
          onOpen={() => setDrawerOpen(true)}
          onClose={() => setDrawerOpen(false)}
          detections={result.detections}
        />
      )}

      {/* Chatbot badge — shown after results load */}
      {result && (
        <div className="fixed z-[300] flex flex-col items-end gap-2" style={{ bottom: 88, right: 16 }}>
          <div className="bg-charcoal text-white text-xs font-semibold px-3 py-1.5 rounded-pill whitespace-nowrap">
            OralSkop AI · Ask about your scan
          </div>
          <button
            onClick={() => {
              setChatNotifDismissed(true)
              setChatOpen(true)
            }}
            className="w-[52px] h-[52px] rounded-full flex items-center justify-center relative"
            style={{ backgroundColor: '#0F6E56', boxShadow: '0 4px 12px rgba(15,110,86,0.4)' }}
          >
            <ChatCircle size={24} color="white" weight="light" />
            {!chatNotifDismissed && (
              <span
                className="absolute top-0.5 right-0.5 w-3 h-3 rounded-full border-2 border-white"
                style={{ backgroundColor: '#D85A30' }}
              />
            )}
          </button>
        </div>
      )}

      {/* Chatbot panel */}
      {result && token && (
        <ChatPanel
          open={chatOpen}
          onClose={() => setChatOpen(false)}
          result={result}
          token={token}
        />
      )}

      {/* Dentist modal */}
      <AnimatePresence>
        {modalOpen && (
          <DentistModal open={modalOpen} onClose={() => setModalOpen(false)} onToast={showToast} />
        )}
      </AnimatePresence>

      {/* Toast */}
      <AnimatePresence>
        {toast && <Toast key={toast} message={toast} onDone={() => setToast(null)} />}
      </AnimatePresence>
    </div>
  )
}
