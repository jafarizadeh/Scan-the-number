import { currentList } from "../state.js";
import { chip } from "../utils.js";

export function activeListHTML(){
  return `
    <section class="card list-panel">
      <div class="panel-title">Active List</div>

      <div class="list-top">
        <div class="list-select"><span id="listName">List 1</span><span class="chev">⌄</span></div>
        <button id="selectListBtn" class="list-btn">Select</button>
      </div>

      <div id="listMeta" class="subtle">0 numbers</div>
      <div id="listNumbers" class="number-row"></div>

      <div class="divider"></div>

      <div class="quickTitle">Quick Preview</div>
      <div id="quickPreview" class="number-row"></div>
    </section>
  `;
}

export function bindActiveList(actions){
  document.getElementById("selectListBtn").onclick = actions.openListModal;
}

export function renderActiveList(){
  const list = currentList();
  if(!list) return;

  document.getElementById("listName").textContent = list.name;
  document.getElementById("listMeta").textContent = `${list.numbers.length} numbers`;
  document.getElementById("listNumbers").innerHTML = list.numbers.map(n => chip(n)).join("");

  const quick = [0,1,2,3,4,5,6,7,8,9,10,11];
  document.getElementById("quickPreview").innerHTML = quick.map(n => chip(n)).join("");
}
