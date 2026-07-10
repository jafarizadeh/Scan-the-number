from __future__ import annotations

import argparse
import glob
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "data" / "config.json"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Tapper Calibration</title>
<style>
*{box-sizing:border-box}
body{
  margin:0;
  background:#eef6fb;
  color:#19324a;
  font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
.page{
  width:100vw;
  min-height:100vh;
  padding:14px;
  display:flex;
  flex-direction:column;
  gap:12px;
}
.top{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:10px;
}
h1{
  margin:0;
  font-size:24px;
  font-weight:950;
}
.sub{
  font-size:12px;
  color:#60788f;
  font-weight:800;
}
.back{
  height:42px;
  border:0;
  border-radius:14px;
  padding:0 16px;
  background:#315b7c;
  color:white;
  font-weight:950;
}
.grid{
  flex:1;
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:12px;
}
.card{
  background:white;
  border:1px solid #d4e2ee;
  border-radius:22px;
  box-shadow:0 10px 28px rgba(31,64,98,.10);
  padding:14px;
}
.cardTitle{
  font-size:13px;
  font-weight:950;
  color:#375977;
  text-transform:uppercase;
  letter-spacing:.7px;
  margin-bottom:12px;
}
.field{
  margin-bottom:14px;
}
label{
  display:flex;
  justify-content:space-between;
  align-items:center;
  font-size:13px;
  font-weight:900;
  margin-bottom:6px;
}
.value{
  color:#2f80ed;
  font-variant-numeric:tabular-nums;
}
.row{
  display:grid;
  grid-template-columns:48px 1fr 48px;
  gap:8px;
  align-items:center;
}
button{
  touch-action:manipulation;
}
.small{
  height:38px;
  border:0;
  border-radius:12px;
  font-size:20px;
  font-weight:950;
  color:#24455f;
  background:#e7f1f8;
}
input[type=number]{
  width:100%;
  height:38px;
  border:1px solid #d4e2ee;
  border-radius:12px;
  text-align:center;
  font-size:17px;
  font-weight:950;
  color:#19324a;
  background:#fbfdff;
}
input[type=range]{
  width:100%;
  margin-top:8px;
}
.actions{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:10px;
  margin-top:12px;
}
.action{
  height:50px;
  border:0;
  border-radius:16px;
  font-size:15px;
  font-weight:950;
  color:white;
  background:#2f80ed;
  box-shadow:0 8px 18px rgba(47,128,237,.22);
}
.secondary{background:#315b7c}
.safe{background:#1d9b68}
.danger{background:#d64545}
.log{
  height:calc(100vh - 150px);
  overflow:auto;
  background:#0f1720;
  color:#d7f7ff;
  border-radius:16px;
  padding:10px;
  font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:12px;
  line-height:1.45;
  white-space:pre-wrap;
}
.status{
  margin-bottom:10px;
  padding:10px;
  border-radius:16px;
  background:#f7fbfe;
  border:1px solid #d4e2ee;
  font-size:12px;
  font-weight:850;
  color:#4f6b82;
}
.hint{
  margin-top:12px;
  border-radius:16px;
  background:#f2f8fc;
  border:1px solid #d9e8f2;
  padding:10px;
  font-size:12px;
  line-height:1.45;
  font-weight:750;
  color:#567085;
}
@media(max-width:760px){
  .grid{grid-template-columns:1fr}
  .log{height:260px}
}
</style>
</head>
<body>
<div class="page">
  <div class="top">
    <div>
      <h1>Tapper Calibration</h1>
      <div class="sub">GP2 Servo · GP0 Buzzer · Raspberry Pi Pico</div>
    </div>
    <button class="back" onclick="location.href='http://'+location.hostname+':8080/?v=back-from-calibration'">Back</button>
  </div>

  <div class="grid">
    <section class="card">
      <div class="cardTitle">Servo Calibration</div>

      <div class="field">
        <label><span>Up position</span><span class="value"><span id="upVal">1000</span> µs</span></label>
        <div class="row">
          <button class="small" data-step="upUs:-25">−</button>
          <input id="upUs" type="number" min="800" max="1400" step="25" value="1000">
          <button class="small" data-step="upUs:25">+</button>
        </div>
        <input id="upRange" type="range" min="800" max="1400" step="25" value="1000">
      </div>

      <div class="field">
        <label><span>Tap position</span><span class="value"><span id="tapVal">1200</span> µs</span></label>
        <div class="row">
          <button class="small" data-step="tapUs:-25">−</button>
          <input id="tapUs" type="number" min="900" max="1600" step="25" value="1200">
          <button class="small" data-step="tapUs:25">+</button>
        </div>
        <input id="tapRange" type="range" min="900" max="1600" step="25" value="1200">
      </div>

      <div class="field">
        <label><span>Hold time</span><span class="value"><span id="holdVal">200</span> ms</span></label>
        <div class="row">
          <button class="small" data-step="holdMs:-50">−</button>
          <input id="holdMs" type="number" min="50" max="700" step="50" value="200">
          <button class="small" data-step="holdMs:50">+</button>
        </div>
        <input id="holdRange" type="range" min="50" max="700" step="50" value="200">
      </div>

      <div class="actions">
        <button class="action secondary" id="moveUp">Move Up</button>
        <button class="action secondary" id="moveTap">Move To Tap</button>
        <button class="action" id="testTap">Test Tap</button>
        <button class="action safe" id="save">Save</button>
        <button class="action secondary" id="buzz">Test Buzzer</button>
        <button class="action danger" id="off">Off / Safe</button>
      </div>

      <div class="hint">
        برای شروع Tap position را روی 1100 تا 1200 بگذار. اگر به صفحه نرسید، 25 تا 25 تا زیاد کن.
        اگر فشار زیاد بود، سریع مقدار را کم کن و Off / Safe بزن.
      </div>
    </section>

    <section class="card">
      <div class="cardTitle">Status & Logs</div>
      <div id="status" class="status">Loading...</div>
      <pre id="log" class="log"></pre>
    </section>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

const ids = ["upUs","tapUs","holdMs"];
const ranges = {upUs:$("upRange"), tapUs:$("tapRange"), holdMs:$("holdRange")};

function clamp(n,min,max){return Math.max(min,Math.min(max,Number(n)||0));}

function values(){
  return {
    upUs: clamp($("upUs").value,800,1400),
    tapUs: clamp($("tapUs").value,900,1600),
    holdMs: clamp($("holdMs").value,50,700)
  };
}

function sync(){
  const v = values();
  $("upUs").value = v.upUs;
  $("tapUs").value = v.tapUs;
  $("holdMs").value = v.holdMs;
  $("upRange").value = v.upUs;
  $("tapRange").value = v.tapUs;
  $("holdRange").value = v.holdMs;
  $("upVal").textContent = v.upUs;
  $("tapVal").textContent = v.tapUs;
  $("holdVal").textContent = v.holdMs;
}

function log(msg,data){
  const t = new Date().toLocaleTimeString();
  let out = "["+t+"] "+msg;
  if(data !== undefined) out += "\\n" + JSON.stringify(data,null,2);
  $("log").textContent = out + "\\n\\n" + $("log").textContent;
}

async function post(path, body){
  const res = await fetch(path,{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body||{})
  });
  const text = await res.text();
  let data;
  try{ data = JSON.parse(text); }
  catch(e){ data = {ok:false, raw:text}; }
  if(!res.ok || data.ok === false) throw data;
  return data;
}

async function getStatus(){
  try{
    const res = await fetch("/api/status");
    const data = await res.json();
    if(data.config){
      $("upUs").value = data.config.upUs;
      $("tapUs").value = data.config.tapUs;
      $("holdMs").value = data.config.holdMs;
      sync();
    }
    $("status").textContent = data.pico && data.pico.ok ? "Pico connected. Ready." : "Page loaded. Pico not ready.";
    log("STATUS", data);
  }catch(e){
    $("status").textContent = "Status error";
    log("STATUS ERROR", e);
  }
}

async function send(command, extra){
  sync();
  const body = Object.assign({command}, values(), extra||{});
  log("SEND "+command, body);
  const data = await post("/api/servo", body);
  log("RESULT", data);
}

$("moveUp").onclick = () => send("up").catch(e=>log("ERROR",e));
$("moveTap").onclick = () => send("move", {positionUs:values().tapUs}).catch(e=>log("ERROR",e));
$("testTap").onclick = () => send("tap").catch(e=>log("ERROR",e));
$("buzz").onclick = () => send("buzz", {buzzMs:300}).catch(e=>log("ERROR",e));
$("off").onclick = () => send("off").catch(e=>log("ERROR",e));
$("save").onclick = async () => {
  try{
    sync();
    const data = await post("/api/save", values());
    $("status").textContent = "Saved. Trigger will use these values.";
    log("SAVE RESULT", data);
  }catch(e){ log("SAVE ERROR", e); }
};

document.querySelectorAll("[data-step]").forEach(btn=>{
  btn.onclick = () => {
    const [id,delta] = btn.dataset.step.split(":");
    $(id).value = Number($(id).value) + Number(delta);
    sync();
  };
});

ids.forEach(id => $(id).addEventListener("input", sync));
$("upRange").oninput = () => {$("upUs").value=$("upRange").value;sync();};
$("tapRange").oninput = () => {$("tapUs").value=$("tapRange").value;sync();};
$("holdRange").oninput = () => {$("holdMs").value=$("holdRange").value;sync();};

sync();
getStatus();
</script>
</body>
</html>
"""

def read_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def write_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def clamp(value, low, high):
    try:
        value = int(value)
    except Exception:
        value = low
    return max(low, min(high, value))

def find_port(cfg):
    hw = cfg.get("hardware", {})
    configured = hw.get("picoPort") or hw.get("serialPort")
    if configured and configured != "auto" and Path(configured).exists():
        return configured
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        raise RuntimeError("No Pico serial port found")
    return ports[0]

def send_pico(command, wait=1.0):
    import serial
    cfg = read_config()
    hw = cfg.get("hardware", {})
    port = find_port(cfg)
    baud = int(hw.get("picoBaud", 115200))
    lines = []
    with serial.Serial(port, baud, timeout=0.25, write_timeout=1) as ser:
        ser.write((command.strip() + "\n").encode())
        ser.flush()
        end = time.time() + wait
        while time.time() < end:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                lines.append(line)
    return {"ok": True, "port": port, "command": command, "lines": lines}

class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        raw = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode(errors="replace")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/calibrate", "/calibrate.html"):
            raw = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/status":
            cfg = read_config()
            hw = cfg.setdefault("hardware", {})
            config = {
                "upUs": int(hw.get("servoUpUs", 1000)),
                "tapUs": int(hw.get("servoTapUs", 1200)),
                "holdMs": int(hw.get("servoHoldMs", 200)),
            }
            try:
                pico = send_pico("GETCAL", 1.0)
            except Exception as exc:
                pico = {"ok": False, "error": str(exc)}
            self.send_json({"ok": True, "config": config, "pico": pico})
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        path = urlparse(self.path).path
        data = self.read_json()

        if path == "/api/servo":
            command = str(data.get("command", "")).lower().strip()

            up_us = clamp(data.get("upUs", 1000), 800, 1400)
            tap_us = clamp(data.get("tapUs", data.get("positionUs", 1200)), 900, 1600)
            position_us = clamp(data.get("positionUs", tap_us), 800, 1600)
            hold_ms = clamp(data.get("holdMs", 200), 50, 700)
            buzz_ms = clamp(data.get("buzzMs", 300), 50, 1500)

            if command == "up":
                pico_cmd, wait = f"SERVO {up_us}", 0.8
            elif command == "move":
                pico_cmd, wait = f"SERVO {position_us}", 0.8
            elif command == "tap":
                pico_cmd, wait = f"TAP {tap_us} {hold_ms}", 1.5
            elif command == "off":
                pico_cmd, wait = "OFF", 0.8
            elif command == "buzz":
                pico_cmd, wait = f"BUZZ {buzz_ms}", 0.8
            else:
                self.send_json({"ok": False, "error": f"Unknown command: {command}"}, 400)
                return

            try:
                self.send_json(send_pico(pico_cmd, wait))
            except Exception as exc:
                self.send_json({"ok": False, "command": pico_cmd, "error": str(exc)}, 500)
            return

        if path == "/api/save":
            up_us = clamp(data.get("upUs", 1000), 800, 1400)
            tap_us = clamp(data.get("tapUs", 1200), 900, 1600)
            hold_ms = clamp(data.get("holdMs", 200), 50, 700)

            cfg = read_config()
            hw = cfg.setdefault("hardware", {})
            hw["dryRun"] = False
            hw["hardwareMode"] = "pico_serial"
            hw["picoPort"] = hw.get("picoPort", "auto")
            hw["picoBaud"] = int(hw.get("picoBaud", 115200))
            hw["buzzerEnabled"] = True
            hw["buzzerPin"] = 0
            hw["actuatorEnabled"] = True
            hw["actuatorPin"] = 2
            hw["servoUpUs"] = up_us
            hw["servoTapUs"] = tap_us
            hw["servoHoldMs"] = hold_ms
            write_config(cfg)

            try:
                pico = send_pico(f"SETCAL {up_us} {tap_us} {hold_ms}", 1.2)
            except Exception as exc:
                pico = {"ok": False, "error": str(exc)}

            self.send_json({"ok": True, "saved": {"upUs": up_us, "tapUs": tap_us, "holdMs": hold_ms}, "pico": pico})
            return

        self.send_error(404, "Not Found")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Calibration server running on http://{args.host}:{args.port}")
    server.serve_forever()

if __name__ == "__main__":
    main()
