import { getJSON, postJSON } from "./api.js";
import { state } from "./state.js";

import { headerHTML, bindHeader, renderHeader } from "./components/header.js";
import { controlsHTML, bindControls, renderControls } from "./components/controls.js";
import { cameraHTML, bindCamera, renderCamera } from "./components/camera.js";
import { infoPanelHTML, renderInfoPanel } from "./components/infoPanel.js";
import { activeListHTML, bindActiveList, renderActiveList } from "./components/activeList.js";
import { historyHTML, renderHistory, enableHistoryDrag } from "./components/history.js";
import { listModalHTML, bindListModal, renderListModal, openListModal, closeListModal } from "./components/listModal.js";
import { matchModalHTML, bindMatchModal, showMatchIfNeeded } from "./components/matchModal.js";

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
    alert("Settings page will be modularized next.");
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
    state.appSettings = await postJSON("/api/lists", state.appSettings);
    renderAll();
    closeListModal();
  }
};

function appHTML(){
  return `
    <div class="app">
      ${headerHTML()}

      <main class="mainGrid">
        ${controlsHTML()}
        ${cameraHTML()}
        ${infoPanelHTML()}
        ${activeListHTML()}
      </main>

      ${historyHTML()}
      ${listModalHTML()}
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
}

async function loadSettings(){
  state.appSettings = await getJSON("/api/settings");
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
