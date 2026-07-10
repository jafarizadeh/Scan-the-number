import { state } from "../state.js";

export function cameraHTML(){
  return `
    <section class="card camera-panel">
      <div class="panel-title">Live Camera</div>

      <div class="camera-frame" id="cameraFrame">
        <img id="cameraImg" src="/stream.mjpg">
        <div class="live-badge">LIVE</div>
      </div>

      <div class="camera-buttons">
        <button class="btn btn-start" id="startScanBtn">▶ START</button>
        <button class="btn btn-stop" id="stopScanBtn">■ STOP</button>
      </div>
    </section>
  `;
}

export function bindCamera(actions){
  document.getElementById("startScanBtn").onclick = actions.startScan;
  document.getElementById("stopScanBtn").onclick = actions.stopScan;
}

export function renderCamera(){
  const frame = document.getElementById("cameraFrame");
  if(frame){
    frame.classList.toggle("off", !state.cameraOn);
  }
}
