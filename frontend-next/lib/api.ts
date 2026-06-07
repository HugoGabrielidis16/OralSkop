import { ScreeningResult, HistoryScreening } from './types'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const HEADERS = {
  'ngrok-skip-browser-warning': 'true',
}

export async function analyzePhoto(file: File, token: string): Promise<ScreeningResult> {
  const formData = new FormData()
  formData.append('photo', file)

  const response = await fetch(`${BASE_URL}/api/screenings`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      ...HEADERS,
    },
    body: formData,
  })

  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || `HTTP ${response.status}`)
  }

  return response.json() as Promise<ScreeningResult>
}

export async function fetchHistory(token: string, limit = 5): Promise<HistoryScreening[]> {
  const response = await fetch(`${BASE_URL}/api/history?limit=${limit}`, {
    headers: {
      Authorization: `Bearer ${token}`,
      ...HEADERS,
    },
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }

  return response.json() as Promise<HistoryScreening[]>
}

export async function sendChatMessage(
  messages: { role: string; content: string }[],
  segmentation: object,
  token: string
): Promise<string> {
  const response = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...HEADERS,
    },
    body: JSON.stringify({ messages, segmentation }),
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }

  const data = await response.json() as { reply?: string; message?: string }
  return data.reply || data.message || JSON.stringify(data)
}

export async function deleteScreening(screeningId: string, token: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/api/screenings/${screeningId}`, {
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${token}`,
      ...HEADERS,
    },
  })

  if (!response.ok && response.status !== 204) {
    const err = await response.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || `HTTP ${response.status}`)
  }
}

export async function registerUser(email: string, password: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...HEADERS },
    body: JSON.stringify({ email, password }),
  })

  if (!response.ok) {
    throw new Error('Registration failed')
  }
}

export function mergeResults(results: ScreeningResult[]): ScreeningResult {
  if (results.length === 1) return results[0]
  return {
    screening_id: results[results.length - 1].screening_id,
    photo_url: results[results.length - 1].photo_url,
    masked_image_url: results[results.length - 1].masked_image_url,
    masked_image_urls: results.map((r) => r.masked_image_url),
    photo_count: results.length,
    escalation_triggered: results.some((r) => r.escalation_triggered),
    detections: results.flatMap((r) => r.detections || []),
  }
}
