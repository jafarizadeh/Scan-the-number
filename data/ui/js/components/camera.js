import { state } from "../state.js";

export function cameraHTML(){
  return `
    <section class="card cameraPanel">
      <div class="cardTitle">Live Camera</div>

      <div class="cameraFrame" id="cameraFrame">
        <img src="/focus.mjpg">
        <div class="liveBadge">LIVE</div>
      </div>

      <div class="cameraButtons">
        <button class="scanBtn startBtn" id="startScanBtn">▶ START</button>
        <button class="scanBtn stopBtn" id="stopScanBtn">■ STOP</button>
      </div>
    </section>
  `;
}

export function bindCamera(actions){
  document.getElementById("startScanBtn").onclick = actions.startScan;
  document.getElementById("stopScanBtn").onclick = actions.stopScan;
}

export function renderCamera(){
  document.getElementById("cameraFrame").classList.toggle("off", !state.cameraOn);
}
