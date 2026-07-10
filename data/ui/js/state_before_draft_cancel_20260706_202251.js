export const state = {
  appSettings: null,
  liveStatus: null,
  cameraOn: true,
  lastMatchId: null,

  // UI navigation
  view: "main",

  // Settings/list editing
  editingListId: null
};

export const RED_NUMBERS = new Set([
  1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36
]);

export function currentList(){
  if(!state.appSettings || !state.appSettings.lists || !state.appSettings.lists.length){
    return null;
  }
  return state.appSettings.lists.find(x => x.id === state.appSettings.selectedListId) || state.appSettings.lists[0];
}

export function editingList(){
  if(!state.appSettings || !state.appSettings.lists || !state.appSettings.lists.length){
    return null;
  }

  if(!state.editingListId){
    state.editingListId = state.appSettings.selectedListId || state.appSettings.lists[0].id;
  }

  return state.appSettings.lists.find(x => x.id === state.editingListId) || currentList();
}
