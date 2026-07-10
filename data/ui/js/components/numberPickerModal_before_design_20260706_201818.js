import { state, editingList } from "../state.js";
import { numberClass } from "../utils.js";

export function numberPickerModalHTML(){
  return `
    <div id="numberPickerModal" class="modal">
      <div class="modalBox numberPickerBox">
        <div class="modalHead">
          <h2 id="numberPickerTitle">Fill Numbers</h2>
          <button id="closeNumberPickerBtn" class="closeBtn">Close</button>
        </div>

        <div class="modalBody">
          <div id="numberPickerGrid" class="numberPickerGrid"></div>

          <div class="numberPickerFooter">
            <div id="numberPickerCount" class="numberPickerCount">0 selected</div>
            <div class="numberPickerActions">
              <button id="clearPickerBtn" class="settingsBtnSmall">Clear</button>
              <button id="savePickerBtn" class="settingsBtnSmall settingsBtnPrimary">Done</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

export function bindNumberPicker(actions){
  document.getElementById("closeNumberPickerBtn").onclick = actions.closeNumberPicker;
  document.getElementById("clearPickerBtn").onclick = actions.clearEditingNumbers;
  document.getElementById("savePickerBtn").onclick = actions.closeNumberPicker;
}

export function openNumberPicker(){
  document.getElementById("numberPickerModal").classList.add("show");
}

export function closeNumberPicker(){
  document.getElementById("numberPickerModal").classList.remove("show");
}

export function renderNumberPicker(actions){
  const list = editingList();
  const grid = document.getElementById("numberPickerGrid");

  if(!list || !grid) return;

  document.getElementById("numberPickerTitle").textContent = `Fill Numbers - ${list.name}`;
  document.getElementById("numberPickerCount").textContent = `${list.numbers.length} selected`;

  grid.innerHTML = Array.from({length:37}, (_, n) => {
    const selected = list.numbers.includes(n) ? "selected" : "";
    return `
      <button class="numberPickBtn ${numberClass(n)} ${selected}" data-number="${n}">
        ${n}
      </button>
    `;
  }).join("");

  grid.querySelectorAll(".numberPickBtn").forEach(btn => {
    btn.onclick = () => actions.toggleNumber(Number(btn.dataset.number));
  });
}
