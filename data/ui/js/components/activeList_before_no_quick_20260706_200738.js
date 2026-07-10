import { currentList } from "../state.js";
import { chip } from "../utils.js";

export function activeListHTML(){
  return `
    <section class="card listPanel">
      <div class="activeListHeader">
        <div class="cardTitle activeListTitle">Active List</div>
        <div id="activeListCount" class="activeListCount">0</div>
      </div>

      <button id="activeListButton" class="activeListButton">
        <div>
          <div id="listName" class="activeListName">List 1</div>
          <div id="listMeta" class="activeListMeta">0 numbers</div>
        </div>
        <div class="activeListChevron">⌄</div>
      </button>

      <div class="activeNumbersWrap">
        <div id="listNumbers" class="numberRow activeNumbers"></div>
      </div>

      <div class="divider"></div>

      <div class="quickHeader">
        <div class="quickTitle">Quick Preview</div>
        <div class="quickHint">0–11</div>
      </div>
      <div id="quickPreview" class="numberRow quickPreview"></div>
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
  document.getElementById("activeListCount").textContent = list.numbers.length;
  document.getElementById("listNumbers").innerHTML = list.numbers.map(n => chip(n)).join("");

  const quick = [0,1,2,3,4,5,6,7,8,9,10,11];
  document.getElementById("quickPreview").innerHTML = quick.map(n => chip(n)).join("");
}
