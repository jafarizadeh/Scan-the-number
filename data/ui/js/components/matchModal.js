import { state } from "../state.js";
import { numberClass } from "../utils.js";

export function matchModalHTML(){
  return `
    <div id="matchModal" class="modal">
      <div class="modalBox popupBox">
        <div class="popupTitle">TARGET FOUND</div>
        <div id="popupNumber" class="popupNumber">--</div>
        <div id="popupText">Number found in selected list</div>
        <br>
        <button id="closeMatchModalBtn" class="closeBtn">OK</button>
      </div>
    </div>
  `;
}

export function bindMatchModal(){
  document.getElementById("closeMatchModalBtn").onclick = closeMatchModal;
}

export function showMatchIfNeeded(){
  const match = state.liveStatus?.latestMatch;
  if(!match) return;

  if(match.id === state.lastMatchId) return;
  state.lastMatchId = match.id;

  const numberBox = document.getElementById("popupNumber");
  numberBox.textContent = match.number;
  numberBox.className = `popupNumber ${numberClass(match.number)}`;

  document.getElementById("popupText").textContent = `Number ${match.number} is in selected list`;
  document.getElementById("matchModal").classList.add("show");

  setTimeout(closeMatchModal, 2200);
}

export function closeMatchModal(){
  document.getElementById("matchModal").classList.remove("show");
}
