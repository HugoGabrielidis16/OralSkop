import Link from 'next/link'

interface LogoProps {
  size?: 'sm' | 'md'
}

export default function Logo({ size = 'md' }: LogoProps) {
  const isSmall = size === 'sm'
  const circleSize = isSmall ? 28 : 36
  const fontSize = isSmall ? 'text-base' : 'text-xl'

  return (
    <Link href="/guide" className="flex items-center gap-2 no-underline">
      <svg
        width={circleSize}
        height={circleSize}
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Teal circle background */}
        <circle cx="18" cy="18" r="18" fill="#0F6E56" />
        {/* White outer ring of O */}
        <circle cx="18" cy="16" r="9" stroke="white" strokeWidth="2.5" fill="none" />
        {/* Smile arc inside the O */}
        <path
          d="M12 19 Q18 25 24 19"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
      </svg>
      <span
        className={`font-bold text-charcoal tracking-tight ${fontSize}`}
        style={{ fontFamily: 'Inter, sans-serif' }}
      >
        OralSkop
      </span>
    </Link>
  )
}
