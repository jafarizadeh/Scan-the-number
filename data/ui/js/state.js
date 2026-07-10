export const state = {
  appSettings: null,
  liveStatus: null,
  cameraOn: true,
  lastMatchId: null,

  view: "main",

  // Draft used only inside Settings page.
  // Back discards it. Save writes it to server.
  settingsDraft: null,
  editingListId: null,

  // Used to cancel changes inside Fill Numbers popup.
  numberPickerOriginalNumbers: null
};

export const RED_NUMBERS = new Set([
  1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36
]);

export function cloneSettings(settings){
  return JSON.parse(JSON.stringify(settings));
}

export function settingsSource(){
  return state.settingsDraft || state.appSettings;
}

export function currentList(){
  if(!state.appSettings || !state.appSettings.lists || !state.appSettings.lists.length){
    return null;
  }
  return state.appSettings.lists.find(x => x.id === state.appSettings.selectedListId) || state.appSettings.lists[0];
}

export function editingList(){
  const source = settingsSource();

  if(!source || !source.lists || !source.lists.length){
    return null;
  }

  if(!state.editingListId){
    state.editingListId = source.selectedListId || source.lists[0].id;
  }

  return source.lists.find(x => x.id === state.editingListId) || source.lists[0];
}
