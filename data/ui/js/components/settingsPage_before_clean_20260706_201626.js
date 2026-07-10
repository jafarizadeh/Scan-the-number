import { state, editingList } from "../state.js";
import { chip } from "../utils.js";

export function settingsPageHTML(){
  return `
    <section id="settingsView" class="settingsView hidden">
      <div class="settingsTop">
        <div class="settingsTitle">List Management</div>
        <div class="settingsActions">
          <button id="settingsSaveBtn" class="settingsBtnSmall settingsBtnPrimary">Save</button>
          <button id="settingsBackBtn" class="settingsBtnSmall">Back</button>
        </div>
      </div>

      <div class="settingsGrid">
        <section class="card settingsListPanel">
          <div class="cardTitle" style="padding:0 4px">Lists</div>
          <div id="settingsListScroll" class="settingsListScroll"></div>

          <div class="settingsListFooter">
            <button id="addListBtn" class="settingsBtnSmall settingsBtnPrimary">Add</button>
            <button id="deleteListBtn" class="settingsBtnSmall">Delete</button>
          </div>
        </section>

        <section class="card settingsDetailPanel">
          <div class="cardTitle" style="padding:0 4px">Selected List</div>

          <div class="detailBox">
            <div class="detailMain">
              <div id="detailName" class="detailName">List</div>
              <div id="detailMeta" class="detailMeta">0 numbers selected</div>
              <div class="detailNote">
                Manage the selected list here. Use Fill Numbers to select roulette numbers from a touch popup.
              </div>
              <div id="detailNumbers" class="detailNumbers"></div>
            </div>

            <div class="detailSide">
              <button id="fillNumbersBtn" class="primary">Fill Numbers</button>
              <button id="useOnMainBtn" class="green">Use on Main</button>
              <button id="clearNumbersBtn">Clear Numbers</button>
              <button id="deleteListSideBtn" class="red">Delete List</button>
            </div>
          </div>

          <div class="settingsBottom">
            <div id="settingsMessage" class="settingsMessage">Ready</div>
            <button id="settingsSaveBottomBtn" class="settingsBtnSmall settingsBtnPrimary">Save Lists</button>
          </div>
        </section>
      </div>
    </section>
  `;
}

export function bindSettingsPage(actions){
  document.getElementById("settingsBackBtn").onclick = actions.showMain;
  document.getElementById("settingsSaveBtn").onclick = actions.saveLists;
  document.getElementById("settingsSaveBottomBtn").onclick = actions.saveLists;
  document.getElementById("addListBtn").onclick = actions.addList;
  document.getElementById("deleteListBtn").onclick = actions.deleteEditingList;
  document.getElementById("deleteListSideBtn").onclick = actions.deleteEditingList;
  document.getElementById("fillNumbersBtn").onclick = actions.openNumberPicker;
  document.getElementById("useOnMainBtn").onclick = actions.useEditingListOnMain;
  document.getElementById("clearNumbersBtn").onclick = actions.clearEditingNumbers;
}

export function renderSettingsPage(actions){
  const listBox = document.getElementById("settingsListScroll");
  const list = editingList();

  if(!state.appSettings || !listBox || !list) return;

  listBox.innerHTML = state.appSettings.lists.map(item => {
    const active = item.id === state.editingListId ? "active" : "";
    return `
      <button class="settingsListCard ${active}" data-list-id="${item.id}">
        <div class="settingsListName">${item.name}</div>
        <div class="settingsListMeta">${item.numbers.length} numbers</div>
        <div class="settingsListMini">${item.numbers.slice(0,8).map(n => chip(n)).join("")}</div>
      </button>
    `;
  }).join("");

  listBox.querySelectorAll(".settingsListCard").forEach(btn => {
    btn.onclick = () => actions.editList(btn.dataset.listId);
  });

  document.getElementById("detailName").textContent = list.name;
  document.getElementById("detailMeta").textContent = `${list.numbers.length} numbers selected`;
  document.getElementById("detailNumbers").innerHTML = list.numbers.map(n => chip(n)).join("");
}

export function showSettingsView(){
  document.getElementById("mainView").classList.add("hidden");
  document.getElementById("settingsView").classList.remove("hidden");
}

export function showMainView(){
  document.getElementById("settingsView").classList.add("hidden");
  document.getElementById("mainView").classList.remove("hidden");
}

export function setSettingsMessage(text){
  const el = document.getElementById("settingsMessage");
  if(el) el.textContent = text;
}
