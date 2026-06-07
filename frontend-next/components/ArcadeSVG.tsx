// Exact rect positions from the original HTML arcade SVG
// Upper arch: teeth 1-11, Lower arch: teeth 12-22

interface ToothRect {
  id: number
  x: number
  y: number
  w: number
  h: number
}

const UPPER_TEETH: ToothRect[] = [
  { id: 1,  x: 20,  y: 20, w: 28, h: 36 },
  { id: 2,  x: 52,  y: 14, w: 26, h: 38 },
  { id: 3,  x: 82,  y: 10, w: 24, h: 38 },
  { id: 4,  x: 110, y: 8,  w: 22, h: 36 },
  { id: 5,  x: 136, y: 8,  w: 22, h: 36 },
  { id: 6,  x: 162, y: 8,  w: 22, h: 36 },
  { id: 7,  x: 188, y: 8,  w: 22, h: 36 },
  { id: 8,  x: 214, y: 8,  w: 22, h: 36 },
  { id: 9,  x: 240, y: 10, w: 24, h: 38 },
  { id: 10, x: 268, y: 14, w: 26, h: 38 },
  { id: 11, x: 298, y: 20, w: 28, h: 36 },
]

const LOWER_TEETH: ToothRect[] = [
  { id: 12, x: 20,  y: 88, w: 28, h: 34 },
  { id: 13, x: 52,  y: 84, w: 26, h: 36 },
  { id: 14, x: 82,  y: 82, w: 24, h: 36 },
  { id: 15, x: 110, y: 82, w: 22, h: 34 },
  { id: 16, x: 136, y: 82, w: 22, h: 34 },
  { id: 17, x: 162, y: 82, w: 22, h: 34 },
  { id: 18, x: 188, y: 82, w: 22, h: 34 },
  { id: 19, x: 214, y: 82, w: 22, h: 34 },
  { id: 20, x: 240, y: 82, w: 24, h: 36 },
  { id: 21, x: 268, y: 84, w: 26, h: 36 },
  { id: 22, x: 298, y: 88, w: 28, h: 34 },
]

const ALL_TEETH = [...UPPER_TEETH, ...LOWER_TEETH]

interface ArcadeSVGProps {
  highlightedTeeth?: number[]
  selectedTooth?: number | null
  onToothClick?: (tooth: number) => void
}

export default function ArcadeSVG({ highlightedTeeth = [], selectedTooth, onToothClick }: ArcadeSVGProps) {
  return (
    <svg viewBox="0 0 346 140" width="100%" xmlns="http://www.w3.org/2000/svg">
      {ALL_TEETH.map((tooth) => {
        const isHighlighted = highlightedTeeth.includes(tooth.id)
        const isSelected = selectedTooth === tooth.id
        return (
          <rect
            key={tooth.id}
            id={`tooth-${tooth.id}`}
            x={tooth.x}
            y={tooth.y}
            width={tooth.w}
            height={tooth.h}
            rx={6}
            fill={isSelected ? '#B84020' : isHighlighted ? '#D85A30' : 'white'}
            stroke={isHighlighted ? '#D85A30' : '#E5E7EB'}
            strokeWidth={1.5}
            style={{ cursor: onToothClick ? 'pointer' : 'default', transition: 'fill 0.15s' }}
            onClick={() => onToothClick?.(tooth.id)}
          />
        )
      })}
    </svg>
  )
}
