interface DentalArchProps {
  highlightedTooth: number | null
  onToothClick?: (tooth: number) => void
}

// 14 upper teeth positions along a curved arch (top-down SVG view)
// Tooth numbers 3-16 (upper arch, FDI/ADA simplified to 1-14 index)
const UPPER_TEETH = [
  { id: 1, cx: 100, cy: 60, rx: 10, ry: 12, label: 'UR8' },
  { id: 2, cx: 118, cy: 52, rx: 9, ry: 11, label: 'UR7' },
  { id: 3, cx: 135, cy: 46, rx: 9, ry: 11, label: 'UR6' },
  { id: 4, cx: 152, cy: 42, rx: 8, ry: 10, label: 'UR5' },
  { id: 5, cx: 168, cy: 40, rx: 7, ry: 9, label: 'UR4' },
  { id: 6, cx: 182, cy: 39, rx: 6, ry: 8, label: 'UR3' },
  { id: 7, cx: 195, cy: 40, rx: 5, ry: 7, label: 'UR2' },
  { id: 8, cx: 205, cy: 43, rx: 5, ry: 6, label: 'UR1' },
  { id: 9, cx: 215, cy: 43, rx: 5, ry: 6, label: 'UL1' },
  { id: 10, cx: 225, cy: 40, rx: 5, ry: 7, label: 'UL2' },
  { id: 11, cx: 238, cy: 39, rx: 6, ry: 8, label: 'UL3' },
  { id: 12, cx: 252, cy: 40, rx: 7, ry: 9, label: 'UL4' },
  { id: 13, cx: 268, cy: 42, rx: 8, ry: 10, label: 'UL5' },
  { id: 14, cx: 285, cy: 46, rx: 9, ry: 11, label: 'UL6' },
]

export default function DentalArch({ highlightedTooth, onToothClick }: DentalArchProps) {
  return (
    <div className="flex flex-col items-center">
      <svg
        viewBox="60 20 300 130"
        width="100%"
        style={{ maxWidth: 340 }}
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Arch gum shape */}
        <path
          d="M95 110 Q100 30 210 25 Q320 30 325 110 Q280 140 210 145 Q140 140 95 110 Z"
          fill="#F9E8E8"
          stroke="#E5C5C5"
          strokeWidth="1.5"
        />

        {/* Arch outline */}
        <path
          d="M100 108 Q106 35 210 28 Q314 35 320 108"
          fill="none"
          stroke="#D0A0A0"
          strokeWidth="1"
          strokeDasharray="3 2"
        />

        {/* Teeth */}
        {UPPER_TEETH.map((tooth) => {
          const isHighlighted = highlightedTooth === tooth.id
          return (
            <g
              key={tooth.id}
              onClick={() => onToothClick?.(tooth.id)}
              style={{ cursor: onToothClick ? 'pointer' : 'default' }}
            >
              <ellipse
                cx={tooth.cx}
                cy={tooth.cy}
                rx={tooth.rx}
                ry={tooth.ry}
                fill={isHighlighted ? '#D85A30' : '#FFFFFF'}
                stroke={isHighlighted ? '#B84020' : '#C8C8C8'}
                strokeWidth="1.5"
              />
              {/* Tooth number tooltip on highlighted */}
              {isHighlighted && (
                <text
                  x={tooth.cx}
                  y={tooth.cy + tooth.ry + 12}
                  textAnchor="middle"
                  fontSize="7"
                  fill="#D85A30"
                  fontWeight="700"
                  fontFamily="Inter, sans-serif"
                >
                  #{tooth.id}
                </text>
              )}
            </g>
          )
        })}

        {/* Center line */}
        <line
          x1="210"
          y1="26"
          x2="210"
          y2="50"
          stroke="#E5E7EB"
          strokeWidth="1"
          strokeDasharray="2 2"
        />
      </svg>

      <p className="text-xs text-gray mt-1">Tap a tooth to highlight</p>
    </div>
  )
}
