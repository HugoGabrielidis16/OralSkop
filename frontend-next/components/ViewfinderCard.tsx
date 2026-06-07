import { useEffect, RefObject } from 'react'

interface ViewfinderCardProps {
  videoRef?: RefObject<HTMLVideoElement>
  stream?: MediaStream | null
}

export default function ViewfinderCard({ videoRef, stream }: ViewfinderCardProps) {
  // Start video playback when stream is attached
  useEffect(() => {
    if (videoRef?.current && stream) {
      videoRef.current.srcObject = stream
      videoRef.current.play().catch(() => {})
    }
  }, [stream, videoRef])

  return (
    <div
      className="w-full rounded-card flex flex-col items-center justify-center px-4 relative overflow-hidden transition-all"
      style={{ backgroundColor: '#1F2937', minHeight: stream ? 300 : 260, paddingTop: stream ? 0 : 32, paddingBottom: stream ? 0 : 32 }}
    >
      {/* Live camera feed */}
      {stream && (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
        />
      )}

      {/* Corner brackets — always on top */}
      <svg className="absolute top-4 left-4 z-10" width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M2 14 L2 2 L14 2" stroke="#5EC9A8" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      <svg className="absolute top-4 right-4 z-10" width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 2 L26 2 L26 14" stroke="#5EC9A8" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      <svg className="absolute bottom-10 left-4 z-10" width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M2 14 L2 26 L14 26" stroke="#5EC9A8" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      <svg className="absolute bottom-10 right-4 z-10" width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 26 L26 26 L26 14" stroke="#5EC9A8" strokeWidth="2.5" strokeLinecap="round" />
      </svg>

      {/* Static illustration — shown when no stream */}
      {!stream && (
        <svg width="200" height="120" viewBox="0 0 200 120" fill="none" className="relative z-10">
          <ellipse cx="100" cy="60" rx="90" ry="52" stroke="#5EC9A8" strokeWidth="2" strokeDasharray="8 5" />
          <rect x="52" y="52" width="96" height="22" rx="8" fill="#C97070" opacity="0.7" />
          <rect x="56" y="50" width="18" height="20" rx="5" fill="white" />
          <rect x="78" y="48" width="20" height="22" rx="5" fill="white" />
          <rect x="102" y="48" width="20" height="22" rx="5" fill="white" />
          <rect x="126" y="50" width="18" height="20" rx="5" fill="white" />
        </svg>
      )}

      {/* Dashed oval guide — shown over live feed */}
      {stream && (
        <svg width="220" height="130" viewBox="0 0 200 120" fill="none" className="relative z-10">
          <ellipse cx="100" cy="60" rx="90" ry="52" stroke="#5EC9A8" strokeWidth="2" strokeDasharray="8 5" />
        </svg>
      )}

      {/* Caption */}
      <p
        className="mt-4 text-sm italic font-light text-center relative z-10"
        style={{ color: '#5EC9A8' }}
      >
        Align your mouth in the frame
      </p>
    </div>
  )
}
