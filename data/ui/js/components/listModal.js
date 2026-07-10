import { state } from "../state.js";
import { chip } from "../utils.js";

export function listModalHTML(){
  return `
    <div id="listModal" class="modal">
      <div class="modalBox">
        <div class="modalHead">
          <h2>Select Active List</h2>
          <button id="closeListModalBtn" class="closeBtn">Close</button>
        </div>
        <div id="listChoices" class="modalBody"></div>
      </div>
    </div>
  `;
}

export function bindListModal(actions){
  document.getElementById("closeListModalBtn").onclick = actions.closeListModal;
}

export function openListModal(){
  document.getElementById("listModal").classList.add("show");
}

export function closeListModal(){
  document.getElementById("listModal").classList.remove("show");
}

export function renderListModal(actions){
  const box = document.getElementById("listChoices");
  if(!state.appSettings || !box) return;

  box.innerHTML = state.appSettings.lists.map(list => {
    const active = list.id === state.appSettings.selectedListId ? "active" : "";
    return `
      <button class="listChoice ${active}" data-list-id="${list.id}">
        <div>
          <div class="listChoiceName">${list.name}</div>
          <div class="listChoiceMeta">${list.numbers.length} numbers</div>
        </div>
        <div class="numberRow">${list.numbers.slice(0,6).map(n => chip(n)).join("")}</div>
      </button>
    `;
  }).join("");

  box.querySelectorAll(".listChoice").forEach(btn => {
    btn.onclick = () => actions.selectList(btn.dataset.listId);
  });
}
