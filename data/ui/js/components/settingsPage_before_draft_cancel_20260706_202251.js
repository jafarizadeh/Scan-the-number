import { state, editingList } from "../state.js";
import { chip } from "../utils.js";

export function settingsPageHTML(){
  return `
    <section id="settingsView" class="settingsView hidden">
      <div class="settingsTop">
        <div>
          <div class="settingsTitle">List Management</div>
          <div class="settingsSubtitle">Create lists, choose numbers, then save changes</div>
        </div>

        <div class="settingsActions">
          <button id="settingsSaveBtn" class="settingsBtn settingsPrimary">Save</button>
          <button id="settingsBackBtn" class="settingsBtn">Back</button>
        </div>
      </div>

      <div class="settingsGrid">
        <section class="card settingsListPanel">
          <div class="settingsPanelHeader">
            <div class="settingsPanelTitle">Lists</div>
            <div id="settingsListCount" class="settingsCounter">0</div>
          </div>

          <div id="settingsListScroll" class="settingsListScroll"></div>

          <div class="settingsListFooter">
            <button id="addListBtn" class="settingsActionBtn add">Add List</button>
            <button id="deleteListBtn" class="settingsActionBtn delete">Delete</button>
          </div>
        </section>

        <section class="card settingsDetailPanel">
          <div class="settingsPanelHeader">
            <div class="settingsPanelTitle">Selected List</div>
            <button id="fillNumbersBtn" class="settingsActionBtn fill">Fill Numbers</button>
          </div>

          <div class="selectedListCard">
            <div class="selectedListTop">
              <div>
                <div id="detailName" class="detailName">List</div>
                <div id="detailMeta" class="detailMeta">0 numbers selected</div>
              </div>

              <div id="detailBadge" class="detailBadge">0</div>
            </div>

            <div class="selectedListDivider"></div>

            <div id="detailNumbers" class="detailNumbers"></div>

            <div id="emptyListNote" class="emptyListNote">
              No numbers selected. Tap <b>Fill Numbers</b> to choose roulette numbers.
            </div>
          </div>

          <div class="settingsMessageRow">
            <div id="settingsMessage" class="settingsMessage">Ready</div>
          </div>
        </section>
      </div>
    </section>
  `;
}

export function bindSettingsPage(actions){
  document.getElementById("settingsBackBtn").onclick = actions.showMain;
  document.getElementById("settingsSaveBtn").onclick = actions.saveLists;
  document.getElementById("addListBtn").onclick = actions.addList;
  document.getElementById("deleteListBtn").onclick = actions.deleteEditingList;
  document.getElementById("fillNumbersBtn").onclick = actions.openNumberPicker;
}

export function renderSettingsPage(actions){
  const listBox = document.getElementById("settingsListScroll");
  const list = editingList();

  if(!state.appSettings || !listBox || !list) return;

  document.getElementById("settingsListCount").textContent = state.appSettings.lists.length;

  listBox.innerHTML = state.appSettings.lists.map(item => {
    const active = item.id === state.editingListId ? "active" : "";
    const preview = item.numbers.slice(0, 8).map(n => chip(n)).join("");

    return `
      <button class="settingsListCard ${active}" data-list-id="${item.id}">
        <div class="settingsListMain">
          <div class="settingsListName">${item.name}</div>
          <div class="settingsListMeta">${item.numbers.length} numbers</div>
        </div>
        <div class="settingsListMini">${preview}</div>
      </button>
    `;
  }).join("");

  listBox.querySelectorAll(".settingsListCard").forEach(btn => {
    btn.onclick = () => actions.editList(btn.dataset.listId);
  });

  document.getElementById("detailName").textContent = list.name;
  document.getElementById("detailMeta").textContent = `${list.numbers.length} numbers selected`;
  document.getElementById("detailBadge").textContent = list.numbers.length;

  const numbers = document.getElementById("detailNumbers");
  numbers.innerHTML = list.numbers.map(n => chip(n)).join("");

  document.getElementById("emptyListNote").classList.toggle("hidden", list.numbers.length > 0);
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
