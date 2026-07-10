# Raspberry Roulette Interface Fix Notes

This package focuses on the core workflow:

1. The backend camera reads the roulette number.
2. OCR and stability validation confirm a stable value.
3. The frontend displays the same backend camera stream used by OCR.
4. The frontend receives validated readings through WebSocket.
5. When the validated number matches the selected target, the frontend shows the match popup and the backend triggers the Pico/actuator.

## Main fixes

- Added frontend-compatible `/api/v1/...` backend endpoints.
- Added session start/stop support.
- Added live WebSocket endpoint at `/api/v1/live`.
- Added readings history stored in memory.
- Added test-reading endpoint for frontend test buttons.
- Changed the frontend camera preview to use the backend MJPEG stream instead of the browser camera.
- Changed the default surveillance client from `mock` to `backend`.
- Kept the original legacy backend endpoints such as `/api/status`, `/api/start`, `/api/video`.

## Important files changed

Backend:

- `backend/app/main.py`
- `backend/app/controller.py`

Frontend:

- `frontend/src/hooks/useCameraPreview.ts`
- `frontend/src/pages/SurveillancePage.tsx`
- `frontend/src/components/surveillance/GamePanel.tsx`
- `frontend/src/components/surveillance/CameraPreviewCard.tsx`
- `frontend/src/api/createSurveillanceClient.ts`
- `frontend/src/styles/main.css`
- `frontend/.env.example`

## Run backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open the frontend and press `Démarrer`. The video panel should show the backend camera stream from `/api/video`, and validated readings should arrive through `/api/v1/live`.

## Validation performed

- Python syntax check: `python3 -m py_compile backend/app/*.py`
- TypeScript check: `node node_modules/typescript/bin/tsc -b`

A full Vite build should be executed after reinstalling dependencies on the target machine because the old ZIP contained a platform-specific `node_modules` directory.
