#!/usr/bin/env bash

URL="http://127.0.0.1:8080/?v=kiosk"

# صبر تا backend بالا بیاید
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8080/api/settings >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# جلوگیری از خاموش شدن صفحه
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

CHROME="$(command -v chromium-browser || command -v chromium || true)"

if [ -z "$CHROME" ]; then
  echo "Chromium not found"
  exit 1
fi

exec "$CHROME" \
  --kiosk "$URL" \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --disable-restore-session-state \
  --check-for-update-interval=31536000 \
  --autoplay-policy=no-user-gesture-required \
  --start-fullscreen
