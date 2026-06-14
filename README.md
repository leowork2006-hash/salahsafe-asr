# SalahSafe Tilāwah ASR worker

RunPod serverless worker for realtime Quran recitation tracking.

- Model: [`tarteel-ai/whisper-base-ar-quran`](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) (public).
- Phone streams short rolling audio windows → worker returns transcribed Arabic.
- The app aligns the text against the known mushaf ayah (green/red, makharij).
  **No scripture is generated server-side** — only what was heard, for matching.

Image is built by GitHub Actions and published to
`ghcr.io/<owner>/salahsafe-asr:latest`, then deployed on RunPod serverless
(scale-to-zero). See `TILAWAH_VOICE_PLAN.md` in the app repo.
