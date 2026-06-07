import { motion, AnimatePresence } from 'framer-motion'
import { X } from '@phosphor-icons/react'

interface DentistModalProps {
  open: boolean
  onClose: () => void
  onToast: (msg: string) => void
}

const DENTISTS = [
  { name: 'Dr. Martin',  distance: '0.3 km', status: 'Accepting new patients', available: true },
  { name: 'Dr. Lefevre', distance: '1.1 km', status: 'Accepting new patients', available: true },
  { name: 'Dr. Benali',  distance: '2.4 km', status: 'Limited availability',   available: false },
]

export default function DentistModal({ open, onClose, onToast }: DentistModalProps) {
  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            key="modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-50"
            style={{ background: 'rgba(0,0,0,0.4)' }}
          />

          {/* Bottom sheet */}
          <motion.div
            key="modal-sheet"
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', stiffness: 350, damping: 40 }}
            className="fixed bottom-0 left-0 right-0 z-50 bg-white max-w-[390px] mx-auto px-5 pt-5 pb-10"
            style={{ borderRadius: '20px 20px 0 0', boxShadow: '0 -4px 24px rgba(0,0,0,0.12)' }}
          >
            {/* Handle */}
            <div className="w-10 h-1 bg-line rounded-full mx-auto mb-4" />

            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-charcoal font-extrabold text-lg">Find a dentist near you</h2>
                <p className="text-gray text-sm mt-0.5">Based on your location</p>
              </div>
              <button onClick={onClose} className="p-1 rounded-full hover:bg-line transition-colors">
                <X size={20} color="#6B7280" />
              </button>
            </div>

            {/* Dentist list */}
            <div className="flex flex-col">
              {DENTISTS.map((d) => (
                <div key={d.name} className="flex items-center justify-between py-3 border-b border-line last:border-0">
                  <div className="flex-1 min-w-0">
                    <p className="font-bold text-charcoal text-sm">{d.name}</p>
                    <p className="text-gray text-xs mt-0.5">{d.distance}</p>
                    <p
                      className="text-[11px] font-semibold mt-0.5"
                      style={{ color: d.available ? '#0F6E56' : '#D85A30' }}
                    >
                      {d.status}
                    </p>
                  </div>
                  <motion.button
                    whileTap={{ scale: 0.95 }}
                    onClick={() => onToast('Coming soon')}
                    className="ml-3 px-4 py-1.5 rounded-pill text-white text-sm font-bold flex-shrink-0"
                    style={{ backgroundColor: '#0F6E56' }}
                  >
                    Book
                  </motion.button>
                </div>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
