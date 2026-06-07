export interface Detection {
  condition: string
  confidence: number
  severity: 'low' | 'moderate' | 'high'
  tooth_number: number | null
  recommendations: string[]
  box_coordinates?: number[]
}

export interface ScreeningResult {
  screening_id: string
  photo_url: string
  masked_image_url: string
  masked_image_urls?: string[]
  photo_count?: number
  escalation_triggered: boolean
  detections: Detection[]
}

export interface HistoryScreening {
  screening_id: string
  captured_at: string
  photo_url: string
  masked_image_url: string
  escalation_triggered: boolean
  condition_summary: string[]
}

export interface UserProfile {
  age?: string
  sex?: string
  smoker?: boolean
  diabetic?: boolean
}

export interface DentistProfile {
  specialty?: string
  clinic?: string
  license?: string
  experience?: string
  available?: boolean
}
