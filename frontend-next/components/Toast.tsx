import { useEffect } from 'react'
import { motion } from 'framer-motion'

interface ToastProps {
  message: string
  onDone: () => void
  duration?: number
}

export default function Toast({ message, onDone, duration = 2500 }: ToastProps) {
  useEffect(() => {
    const t = setTimeout(onDone, duration)
    return () => clearTimeout(t)
  }, [onDone, duration])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="fixed bottom-24 left-1/2 -translate-x-1/2 z-[200] bg-charcoal text-white text-sm font-semibold px-5 py-3 rounded-pill shadow-xl whitespace-nowrap"
    >
      {message}
    </motion.div>
  )
}
