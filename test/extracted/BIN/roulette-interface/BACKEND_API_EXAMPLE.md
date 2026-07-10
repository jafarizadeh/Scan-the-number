# نمونه اتصال frontend به backend

Backend روی این آدرس اجرا می‌شود:

```text
http://localhost:8000
```

برای تغییر عدد هدف از React:

```ts
await fetch("http://localhost:8000/api/target", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ target_number: 17 }),
});
```

برای گرفتن status:

```ts
const res = await fetch("http://localhost:8000/api/status");
const status = await res.json();
console.log(status);
```

برای نمایش ویدیو:

```tsx
<img src="http://localhost:8000/api/video" />
```
