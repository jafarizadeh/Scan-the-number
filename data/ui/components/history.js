import { state } from "../state.js";
import { chip, shortTime } from "../utils.js";

function historyItemHTML(item){
  return `
    <div class="histItem">
      ${chip(item.number, item.isMatch ? "numMatch" : "")}
      <div class="timeLabel">${shortTime(item.timestamp)}</div>
    </div>
  `;
}

export function historyHTML(){
  return `
    <section class="card history-panel">
      <div class="history-top">
        <div class="history-title">History</div>
        <div class="history-note">Swipe left/right to scroll</div>
      </div>

      <div id="historyGrid" class="history-grid"></div>
    </section>
  `;
}

export function renderHistory(){
  const items = state.liveStatus?.history || [];
  document.getElementById("historyGrid").innerHTML = items.slice(0, 120).map(historyItemHTML).join("");
}

export function enableHistoryDrag(){
  const el = document.getElementById("historyGrid");
  if(!el) return;

  let down = false;
  let startX = 0;
  let scrollLeft = 0;

  el.addEventListener("mousedown", e => {
    down = true;
    startX = e.pageX - el.offsetLeft;
    scrollLeft = el.scrollLeft;
  });

  el.addEventListener("mouseup", () => down = false);
  el.addEventListener("mouseleave", () => down = false);

  el.addEventListener("mousemove", e => {
    if(!down) return;
    e.preventDefault();
    const x = e.pageX - el.offsetLeft;
    el.scrollLeft = scrollLeft - (x - startX) * 1.2;
  });

  let touchX = 0;
  let touchScroll = 0;

  el.addEventListener("touchstart", e => {
    touchX = e.touches[0].clientX;
    touchScroll = el.scrollLeft;
  }, {passive:true});

  el.addEventListener("touchmove", e => {
    el.scrollLeft = touchScroll - (e.touches[0].clientX - touchX) * 1.2;
  }, {passive:true});
}
