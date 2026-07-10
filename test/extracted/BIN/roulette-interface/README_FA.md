# Roulette Interface + Backend برای Raspberry Pi 5 و Pico 2 W

این نسخه یک فولدر `backend` به ساختار پروژه اضافه می‌کند و فایل `pico/main.py` را هم برای Pico آماده کرده است.

## کاری که backend انجام می‌دهد

1. تصویر دوربین را از Raspberry Pi 5 می‌گیرد.
2. عدد روی صفحه را با OCR می‌خواند.
3. اگر عدد تشخیص‌داده‌شده در چند فریم پشت سر هم ثابت بود، مثلا 3 فریم، آن را معتبر حساب می‌کند.
4. اگر عدد معتبر با عدد هدف کاربر برابر بود، از طریق USB Serial به Pico دستور `TRIGGER` می‌فرستد.
5. Pico همزمان actuator مدل Actuonix L12-30-50-6-R و buzzer KY-012 را فعال می‌کند.

## ساختار

```text
roulette-interface-with-backend/
├── frontend/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── controller.py
│   │   ├── camera.py
│   │   ├── ocr.py
│   │   ├── pico_serial.py
│   │   ├── settings.py
│   │   └── stability.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── install_pi_dependencies.sh
│   └── run_backend.sh
└── pico/
    └── main.py
```

## سیم‌کشی Pico

### Actuonix L12-30-50-6-R

```text
Actuator white wire  -> Pico GP15
Actuator red wire    -> external 6V +
Actuator black wire  -> external 6V GND
Pico GND             -> external 6V GND
```

### KY-012 buzzer

```text
KY-012 Signal / S -> Pico GP14
KY-012 +          -> Pico 3V3
KY-012 -          -> Pico GND
```

## نصب روی Raspberry Pi 5

داخل فولدر پروژه:

```bash
cd backend
./install_pi_dependencies.sh
./run_backend.sh
```

بعد در مرورگر باز کن:

```text
http://localhost:8000/docs
```

یا از یک کامپیوتر دیگر در همان شبکه:

```text
http://RASPBERRY_PI_IP:8000/docs
```

## تنظیمات مهم

فایل زیر را بساز یا از نمونه کپی کن:

```bash
cd backend
cp .env.example .env
```

مقادیر مهم:

```text
TARGET_NUMBER=17
STABLE_FRAMES=3
PICO_PORT=auto
CAMERA_SOURCE=0
ROI_X=0
ROI_Y=0
ROI_W=0
ROI_H=0
```

اگر `ROI_W` و `ROI_H` برابر 0 باشند، کل تصویر دوربین برای OCR استفاده می‌شود. برای تشخیص بهتر باید ROI را روی همان قسمت صفحه که عدد نمایش داده می‌شود تنظیم کنی.

## API های مهم

وضعیت سیستم:

```text
GET /api/status
```

شروع پردازش دوربین:

```text
POST /api/start
```

توقف پردازش:

```text
POST /api/stop
```

تغییر عدد هدف:

```text
POST /api/target
Body:
{
  "target_number": 17
}
```

تغییر ناحیه OCR:

```text
POST /api/roi
Body:
{
  "x": 100,
  "y": 100,
  "w": 300,
  "h": 120
}
```

تست دستی actuator و buzzer:

```text
POST /api/trigger
```

نمایش ویدیوی دوربین همراه با debug overlay:

```text
GET /api/video
```

## نکته مهم برای دقت OCR

برای اینکه تشخیص عدد خیلی بهتر شود، بهتر است یک تصویر نمونه از دوربین بفرستی که همان صفحه و عدد داخل آن دیده شود. مخصوصا اگر عدد داخل roulette UI کوچک، رنگی، یا روی پس‌زمینه شلوغ است.

با آن تصویر می‌توانم ROI دقیق، threshold مناسب، و preprocessing مخصوص صفحه تو را تنظیم کنم.
