import { currentList } from "../state.js";
import { chip } from "../utils.js";

export function activeListHTML(){
  return `
    <section class="card listPanel">
      <div class="activeListHeader">
        <div class="cardTitle activeListTitle">Active List</div>
      </div>

      <button id="activeListButton" class="activeListCard">
        <div id="listName" class="activeListName">List 1</div>
        <div id="listMeta" class="activeListCountBadge">0 numbers</div>
      </button>

      <div class="activeNumbersArea">
        <div id="listNumbers" class="activeNumbers"></div>
      </div>
    </section>
  `;
}

export function bindActiveList(actions){
  document.getElementById("activeListButton").onclick = actions.openListModal;
}

export function renderActiveList(){
  const list = currentList();
  if(!list) return;

  document.getElementById("listName").textContent = list.name;
  document.getElementById("listMeta").textContent = `${list.numbers.length} numbers`;
  document.getElementById("listNumbers").innerHTML = list.numbers.map(n => chip(n)).join("");
}
