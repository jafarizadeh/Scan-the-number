import { getJSON, postJSON } from "./api.js";
import { state, editingList } from "./state.js";

import { headerHTML, bindHeader, renderHeader } from "./components/header.js";
import { controlsHTML, bindControls, renderControls } from "./components/controls.js";
import { cameraHTML, bindCamera, renderCamera } from "./components/camera.js";
import { infoPanelHTML, renderInfoPanel } from "./components/infoPanel.js";
import { activeListHTML, bindActiveList, renderActiveList } from "./components/activeList.js";
import { historyHTML, renderHistory, enableHistoryDrag } from "./components/history.js";
import { listModalHTML, bindListModal, renderListModal, openListModal, closeListModal } from "./components/listModal.js";
import { matchModalHTML, bindMatchModal, showMatchIfNeeded } from "./components/matchModal.js";
import { settingsPageHTML, bindSettingsPage, renderSettingsPage, showSettingsView, showMainView, setSettingsMessage } from "./components/settingsPage.js";
import { numberPickerModalHTML, bindNumberPicker, renderNumberPicker, openNumberPicker, closeNumberPicker } from "./components/numberPickerModal.js";

const actions = {
  async toggleSound(){
    if(!state.appSettings) return;
    state.appSettings.hardware.buzzerEnabled = !state.appSettings.hardware.buzzerEnabled;
    state.appSettings.hardware = await postJSON("/api/hardware", state.appSettings.hardware);
    renderAll();
  },

  async toggleArm(){
    if(!state.appSettings) return;
    state.appSettings.hardware.actuatorEnabled = !state.appSettings.hardware.actuatorEnabled;
    state.appSettings.hardware = await postJSON("/api/hardware", state.appSettings.hardware);
    renderAll();
  },

  toggleCamera(){
    state.cameraOn = !state.cameraOn;
    renderAll();
  },

  async startScan(){
    if(!state.appSettings) return;
    state.liveStatus = await postJSON("/api/start", {
      selectedListId: state.appSettings.selectedListId
    });
    renderAll();
  },

  async stopScan(){
    state.liveStatus = await postJSON("/api/stop", {});
    renderAll();
  },

  openSettings(){
    state.view = "settings";
    if(!state.editingListId){
      state.editingListId = state.appSettings?.selectedListId;
    }
    showSettingsView();
    renderAll();
  },

  showMain(){
    state.view = "main";
    showMainView();
    renderAll();
  },

  openListModal(){
    renderListModal(actions);
    openListModal();
  },

  closeListModal(){
    closeListModal();
  },

  async selectList(id){
    state.appSettings.selectedListId = id;
    state.editingListId = id;
    state.appSettings = await postJSON("/api/lists", state.appSettings);
    renderAll();
    closeListModal();
  },

  editList(id){
    state.editingListId = id;
    renderAll();
  },

  addList(){
    const count = state.appSettings.lists.length + 1;
    const item = {
      id: "list-" + Math.random().toString(16).slice(2, 10),
      name: "List " + count,
      numbers: []
    };

    state.appSettings.lists.push(item);
    state.editingListId = item.id;
    setSettingsMessage("New list added. Press Save to store it.");
    renderAll();
  },

  deleteEditingList(){
    if(!state.appSettings || state.appSettings.lists.length <= 1){
      setSettingsMessage("At least one list is required.");
      return;
    }

    const id = state.editingListId;
    state.appSettings.lists = state.appSettings.lists.filter(x => x.id !== id);

    if(state.appSettings.selectedListId === id){
      state.appSettings.selectedListId = state.appSettings.lists[0].id;
    }

    state.editingListId = state.appSettings.selectedListId;
    setSettingsMessage("List deleted. Press Save to store changes.");
    renderAll();
  },

  useEditingListOnMain(){
    const list = editingList();
    if(!list) return;

    state.appSettings.selectedListId = list.id;
    setSettingsMessage(`${list.name} selected for main page. Press Save.`);
    renderAll();
  },

  clearEditingNumbers(){
    const list = editingList();
    if(!list) return;

    list.numbers = [];
    setSettingsMessage("Numbers cleared. Press Save.");
    renderAll();
  },

  openNumberPicker(){
    renderNumberPicker(actions);
    openNumberPicker();
  },

  closeNumberPicker(){
    closeNumberPicker();
    renderAll();
  },

  toggleNumber(n){
    const list = editingList();
    if(!list) return;

    const idx = list.numbers.indexOf(n);

    if(idx >= 0){
      list.numbers.splice(idx, 1);
    }else{
      list.numbers.push(n);
    }

    list.numbers.sort((a,b) => a-b);
    renderAll();
  },

  async saveLists(){
    state.appSettings = await postJSON("/api/lists", {
      selectedListId: state.appSettings.selectedListId,
      lists: state.appSettings.lists
    });

    setSettingsMessage("Lists saved.");
    renderAll();
  }
};

function appHTML(){
  return `
    <div class="app">
      ${headerHTML()}

      <div id="mainView" class="mainView">
        <main class="mainGrid">
          ${controlsHTML()}
          ${cameraHTML()}
          ${infoPanelHTML()}
          ${activeListHTML()}
        </main>

        ${historyHTML()}
      </div>

      ${settingsPageHTML()}
      ${listModalHTML()}
      ${numberPickerModalHTML()}
      ${matchModalHTML()}
    </div>
  `;
}

function bindAll(){
  bindHeader(actions);
  bindControls(actions);
  bindCamera(actions);
  bindActiveList(actions);
  bindListModal(actions);
  bindNumberPicker(actions);
  bindSettingsPage(actions);
  bindMatchModal();
  enableHistoryDrag();
}

function renderAll(){
  renderHeader();
  renderControls();
  renderCamera();
  renderInfoPanel();
  renderActiveList();
  renderHistory();
  renderListModal(actions);
  renderSettingsPage(actions);
  renderNumberPicker(actions);
}

async function loadSettings(){
  state.appSettings = await getJSON("/api/settings");

  if(!state.editingListId){
    state.editingListId = state.appSettings.selectedListId;
  }
}

async function poll(){
  try{
    state.liveStatus = await getJSON("/api/status");

    if(
      state.liveStatus.selectedList &&
      state.appSettings &&
      state.liveStatus.selectedList.id !== state.appSettings.selectedListId
    ){
      state.appSettings.selectedListId = state.liveStatus.selectedList.id;
    }

    renderAll();
    showMatchIfNeeded();
  }catch(e){
    console.error(e);
  }
}

async function boot(){
  document.getElementById("app").innerHTML = appHTML();

  await loadSettings();
  await poll();

  bindAll();
  renderAll();

  setInterval(poll, 350);
}

boot();
