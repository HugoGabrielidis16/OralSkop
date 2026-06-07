import { useId } from 'react'
import { motion } from 'framer-motion'

interface SegmentedToggleProps {
  options: string[]
  active: number
  onChange: (i: number) => void
}

export default function SegmentedToggle({ options, active, onChange }: SegmentedToggleProps) {
  const uid = useId()

  return (
    <div className="relative flex items-center bg-line rounded-pill p-1 gap-1">
      {options.map((option, i) => (
        <button
          key={option}
          onClick={() => onChange(i)}
          className="relative z-10 px-4 py-1.5 text-sm font-semibold rounded-pill transition-colors duration-200 focus:outline-none"
          style={{ color: active === i ? '#ffffff' : '#6B7280' }}
        >
          {active === i && (
            <motion.span
              layoutId={`segment-bg-${uid}`}
              className="absolute inset-0 rounded-pill bg-teal"
              style={{ zIndex: -1 }}
              transition={{ type: 'spring', stiffness: 400, damping: 35 }}
            />
          )}
          {option}
        </button>
      ))}
    </div>
  )
}
