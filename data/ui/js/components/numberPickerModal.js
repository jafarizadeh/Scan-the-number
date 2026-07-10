import { editingList } from "../state.js";
import { numberClass } from "../utils.js";

export function numberPickerModalHTML(){
  return `
    <div id="numberPickerModal" class="modal">
      <div class="modalBox numberPickerBox">
        <div class="numberPickerHeader">
          <div>
            <div id="numberPickerTitle" class="numberPickerTitle">Fill Numbers</div>
            <div id="numberPickerSubtitle" class="numberPickerSubtitle">Choose numbers for this list</div>
          </div>

          <button id="closeNumberPickerBtn" class="numberPickerClose">Close</button>
        </div>

        <div class="numberPickerBody">
          <div class="numberPickerTop">
            <div id="numberPickerCount" class="numberPickerCount">0 selected</div>
            <div class="numberPickerHint">Done applies, Close cancels</div>
          </div>

          <div id="numberPickerGrid" class="numberPickerGrid"></div>

          <div class="numberPickerFooter">
            <button id="clearPickerBtn" class="numberPickerBtn secondary">Clear</button>
            <button id="savePickerBtn" class="numberPickerBtn primary">Done</button>
          </div>
        </div>
      </div>
    </div>
  `;
}

export function bindNumberPicker(actions){
  document.getElementById("closeNumberPickerBtn").onclick = actions.cancelNumberPicker;
  document.getElementById("clearPickerBtn").onclick = actions.clearEditingNumbers;
  document.getElementById("savePickerBtn").onclick = actions.confirmNumberPicker;
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

  const selected = new Set((list.numbers || []).map(Number));

  document.getElementById("numberPickerTitle").textContent = "Fill Numbers";
  document.getElementById("numberPickerSubtitle").textContent = list.name;
  document.getElementById("numberPickerCount").textContent = `${selected.size} selected`;

  grid.innerHTML = Array.from({length:37}, (_, n) => {
    const isSelected = selected.has(n) ? "selected" : "";
    return `
      <button class="numberPickBtn ${numberClass(n)} ${isSelected}" data-number="${n}">
        <span>${n}</span>
      </button>
    `;
  }).join("");

  grid.querySelectorAll(".numberPickBtn").forEach(btn => {
    btn.onclick = () => actions.toggleNumber(Number(btn.dataset.number));
  });
}
