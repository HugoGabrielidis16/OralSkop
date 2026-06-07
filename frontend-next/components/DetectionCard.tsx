import { Detection } from '../lib/types'
import SeverityBadge from './SeverityBadge'

interface DetectionCardProps {
  detection: Detection
}

const conditionLabels: Record<Detection['condition'], string> = {
  cavity: 'Cavity',
  gingivitis: 'Localised gingivitis',
  tartar: 'Tartar build-up',
  lesion_suspicious: 'Area to monitor',
}

const toothSectors: Record<number, string> = {
  1: 'Upper right',
  2: 'Upper right',
  3: 'Upper right',
  4: 'Upper right',
  5: 'Upper right',
  6: 'Upper right',
  7: 'Upper right',
  8: 'Upper front',
  9: 'Upper front',
  10: 'Upper left',
  11: 'Upper left',
  12: 'Upper left',
  13: 'Upper left',
  14: 'Upper left',
  15: 'Upper left',
  16: 'Upper left',
  17: 'Lower left',
  18: 'Lower left',
  19: 'Lower left',
  20: 'Lower left',
  21: 'Lower left',
  22: 'Lower left',
  23: 'Lower left',
  24: 'Lower front',
  25: 'Lower front',
  26: 'Lower right',
  27: 'Lower right',
  28: 'Lower right',
  29: 'Lower right',
  30: 'Lower right',
  31: 'Lower right',
  32: 'Lower right',
}

export default function DetectionCard({ detection }: DetectionCardProps) {
  const isSuspicious = detection.condition === 'lesion_suspicious'
  const label = conditionLabels[detection.condition]
  const sector = detection.tooth_number ? toothSectors[detection.tooth_number] ?? `Tooth #${detection.tooth_number}` : 'General'

  return (
    <div
      className="rounded-xl p-4 border-l-4"
      style={{
        backgroundColor: isSuspicious ? '#FDEDE6' : '#ffffff',
        borderLeftColor: '#D85A30',
        borderWidth: '0 0 0 4px',
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="font-bold text-charcoal text-sm leading-tight">{label}</span>
        <span className="text-xs text-gray flex-shrink-0 mt-0.5">{sector}</span>
      </div>
      <div className="mt-2">
        <SeverityBadge severity={detection.severity} />
      </div>
    </div>
  )
}
