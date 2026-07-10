import { state, currentList } from "../state.js";
import { chip } from "../utils.js";

function getHistoryItems(){
  const status = state.liveStatus || {};

  const raw =
    status.history ||
    status.results ||
    status.detections ||
    status.recent ||
    [];

  return Array.isArray(raw) ? raw : [];
}

function getNumber(item){
  if(typeof item === "number") return item;

  if(item && typeof item === "object"){
    const value =
      item.number ??
      item.result ??
      item.value ??
      item.detectedNumber ??
      item.prediction;

    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  const n = Number(item);
  return Number.isFinite(n) ? n : null;
}

export function historyHTML(){
  return `
    <section class="historyPanel">
      <div class="historyHeader">
        <div class="historyTitle">History</div>
        <div class="historyHint">Swipe left/right to scroll</div>
      </div>

      <div id="historyScroller" class="historyScroller">
        <div id="historyList" class="historyList"></div>
      </div>
    </section>
  `;
}

export function renderHistory(){
  const box = document.getElementById("historyList");
  if(!box) return;

  const active = currentList();
  const activeNumbers = new Set((active?.numbers || []).map(Number));

  const items = getHistoryItems()
    .map(item => {
      const number = getNumber(item);
      if(number === null) return null;

      const matched =
        item?.matched === true ||
        item?.match === true ||
        item?.isMatch === true ||
        activeNumbers.has(number);

      return { number, matched };
    })
    .filter(Boolean);

  if(!items.length){
    box.innerHTML = `<div class="historyEmpty">No results yet</div>`;
    return;
  }

  box.innerHTML = items.map(item => {
    const base = chip(item.number);
    const cls = item.matched ? " historyHit" : "";
    return `<div class="historyChipWrap${cls}">${base}</div>`;
  }).join("");
}

export function enableHistoryDrag(){
  const scroller = document.getElementById("historyScroller");
  if(!scroller) return;

  let isDown = false;
  let startX = 0;
  let startScrollLeft = 0;

  scroller.addEventListener("pointerdown", event => {
    isDown = true;
    startX = event.clientX;
    startScrollLeft = scroller.scrollLeft;
    scroller.classList.add("dragging");
    scroller.setPointerCapture(event.pointerId);
  });

  scroller.addEventListener("pointermove", event => {
    if(!isDown) return;
    const dx = event.clientX - startX;
    scroller.scrollLeft = startScrollLeft - dx;
  });

  function stopDrag(){
    isDown = false;
    scroller.classList.remove("dragging");
  }

  scroller.addEventListener("pointerup", stopDrag);
  scroller.addEventListener("pointercancel", stopDrag);
  scroller.addEventListener("pointerleave", stopDrag);
}
