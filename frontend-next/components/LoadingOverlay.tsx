import { motion } from 'framer-motion'
import Logo from './Logo'

interface LoadingOverlayProps {
  text?: string
}

export default function LoadingOverlay({ text = 'Analysing your photo…' }: LoadingOverlayProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="fixed inset-0 z-50 bg-cream flex flex-col items-center justify-center gap-6"
    >
      <Logo size="md" />
      <div className="spinner" />
      <div className="text-center">
        <p className="text-charcoal font-bold text-lg">{text}</p>
        <p className="text-gray text-sm mt-1">This takes a few seconds</p>
      </div>
    </motion.div>
  )
}
