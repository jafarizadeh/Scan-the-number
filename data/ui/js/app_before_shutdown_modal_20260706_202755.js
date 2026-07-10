import { getJSON, postJSON } from "./api.js";
import { state, editingList, cloneSettings, settingsSource } from "./state.js";

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

    // Create a draft. All Settings changes happen here until Save.
    state.settingsDraft = cloneSettings(state.appSettings);
    state.editingListId = state.settingsDraft.selectedListId || state.settingsDraft.lists[0]?.id || null;

    showSettingsView();
    setSettingsMessage("Unsaved changes will be discarded by Back.");
    renderAll();
  },

  showMain(){
    // Back means cancel unsaved Settings changes.
    state.view = "main";
    state.settingsDraft = null;
    state.numberPickerOriginalNumbers = null;
    state.editingListId = state.appSettings?.selectedListId || null;

    closeNumberPicker();
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
    // Main page list selection is intentional and saved immediately.
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
    const source = settingsSource();
    if(!source) return;

    const count = source.lists.length + 1;
    const item = {
      id: "list-" + Math.random().toString(16).slice(2, 10),
      name: "List " + count,
      numbers: []
    };

    source.lists.push(item);
    state.editingListId = item.id;
    setSettingsMessage("New list added. Press Save to store it.");
    renderAll();
  },

  deleteEditingList(){
    const source = settingsSource();
    if(!source || source.lists.length <= 1){
      setSettingsMessage("At least one list is required.");
      return;
    }

    const id = state.editingListId;
    source.lists = source.lists.filter(x => x.id !== id);

    if(source.selectedListId === id){
      source.selectedListId = source.lists[0].id;
    }

    state.editingListId = source.selectedListId;
    setSettingsMessage("List deleted. Press Save to store changes.");
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
    const list = editingList();
    if(!list) return;

    // Snapshot for popup cancel.
    state.numberPickerOriginalNumbers = [...list.numbers];

    renderNumberPicker(actions);
    openNumberPicker();
  },

  cancelNumberPicker(){
    const list = editingList();

    if(list && state.numberPickerOriginalNumbers){
      list.numbers = [...state.numberPickerOriginalNumbers];
      list.numbers.sort((a,b) => a-b);
    }

    state.numberPickerOriginalNumbers = null;
    closeNumberPicker();
    setSettingsMessage("Number changes cancelled.");
    renderAll();
  },

  confirmNumberPicker(){
    state.numberPickerOriginalNumbers = null;
    closeNumberPicker();
    setSettingsMessage("Numbers applied. Press Save to store changes.");
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
    const source = settingsSource();
    if(!source) return;

    const payload = {
      ...state.appSettings,
      selectedListId: source.selectedListId,
      lists: source.lists
    };

    state.appSettings = await postJSON("/api/lists", payload);

    // After save, refresh the draft to the saved state.
    state.settingsDraft = cloneSettings(state.appSettings);
    state.editingListId = state.settingsDraft.selectedListId || state.settingsDraft.lists[0]?.id || null;

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
  state.editingListId = state.appSettings.selectedListId;
}

async function poll(){
  try{
    state.liveStatus = await getJSON("/api/status");

    if(
      state.liveStatus.selectedList &&
      state.appSettings &&
      !state.settingsDraft &&
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
