interface SeverityBadgeProps {
  severity: 'low' | 'moderate' | 'high'
}

const labels: Record<SeverityBadgeProps['severity'], string> = {
  low: 'Mild',
  moderate: 'Moderate severity',
  high: 'High severity',
}

export default function SeverityBadge({ severity }: SeverityBadgeProps) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-pill text-xs font-semibold"
      style={{ backgroundColor: '#FDEDE6', color: '#D85A30' }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
        style={{ backgroundColor: '#D85A30' }}
      />
      {labels[severity]}
    </span>
  )
}
