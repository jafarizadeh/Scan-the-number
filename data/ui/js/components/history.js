import { state, currentList } from "../state.js";
import { chip } from "../utils.js";

let clearConfirmTimer = null;

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

function getHistoryStats(){
  const frequency = new Map();

  for(const item of getHistoryItems()){
    const number = getNumber(item);
    if(number === null) continue;

    frequency.set(
      number,
      (frequency.get(number) || 0) + 1
    );
  }

  const uniqueCounts = [
    ...new Set(frequency.values())
  ].sort((a, b) => b - a);

  const ranks = new Map();

  for(const [number, count] of frequency.entries()){
    ranks.set(
      number,
      uniqueCounts.indexOf(count) + 1
    );
  }

  return { frequency, ranks };
}

function hideHistoryStats(){
  const popup = document.getElementById("historyStatsPopup");
  if(popup) popup.remove();
}

function bindHistoryStatsDismiss(){
  document.addEventListener("pointerdown", event => {
    const popup = document.getElementById("historyStatsPopup");
    if(!popup) return;

    const clickedPopup = event.target.closest("#historyStatsPopup");
    const clickedChip = event.target.closest(
      ".historyChipWrap[data-history-number]"
    );

    if(clickedPopup || clickedChip) return;

    hideHistoryStats();
  });
}

function showHistoryStats(number, anchor){
  hideHistoryStats();

  const { frequency, ranks } = getHistoryStats();

  const count = frequency.get(number) || 0;
  const rank = ranks.get(number) || 0;

  const popup = document.createElement("div");
  popup.id = "historyStatsPopup";
  popup.className = "historyStatsPopup";

  popup.innerHTML = `
    <div class="historyStatsNumber ${number === 0 ? "green" : ([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36].includes(number) ? "red" : "black")}">${number}</div>

    <div class="historyStatsData">
      <div class="historyStatsItem">
        <span>Count</span>
        <strong>${count}</strong>
      </div>

      <div class="historyStatsDivider"></div>

      <div class="historyStatsItem">
        <span>Rank</span>
        <strong>#${rank}</strong>
      </div>
    </div>
  `;

  document.body.appendChild(popup);

  const rect = anchor.getBoundingClientRect();
  const popupRect = popup.getBoundingClientRect();

  let left =
    rect.left +
    rect.width / 2 -
    popupRect.width / 2;

  let top =
    rect.top -
    popupRect.height -
    8;

  left = Math.max(
    8,
    Math.min(
      left,
      window.innerWidth - popupRect.width - 8
    )
  );

  if(top < 8){
    top = rect.bottom + 8;
  }

  popup.style.left = `${left}px`;
  popup.style.top = `${top}px`;
}

function resetClearButton(){
  const button = document.getElementById("historyClearButton");
  if(!button) return;

  button.classList.remove("confirm");
  button.innerHTML = `
    <svg class="historyClearSvg" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7h16"></path>
      <path d="M9 7V4h6v3"></path>
      <path d="M7 7l1 13h8l1-13"></path>
      <path d="M10 11v5"></path>
      <path d="M14 11v5"></path>
    </svg>
    <span>Clear</span>
  `;
}

export function historyHTML(){
  return `
    <section class="historyPanel">
      <div class="historyHeader">
        <div class="historyHeaderLeft">
          <div class="historyTitle">History</div>
          <div id="historyCount" class="historyCount">0</div>
        </div>

        <div class="historyHeaderRight">
          <div class="historyHint">Swipe to scroll</div>

          <button
            id="historyClearButton"
            class="historyClearButton"
            type="button"
            aria-label="Clear history"
          >
            <svg class="historyClearSvg" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 7h16"></path>
              <path d="M9 7V4h6v3"></path>
              <path d="M7 7l1 13h8l1-13"></path>
              <path d="M10 11v5"></path>
              <path d="M14 11v5"></path>
            </svg>
            <span>Clear</span>
          </button>
        </div>
      </div>

      <div class="historyScrollArea">
        <div id="historyScroller" class="historyScroller">
          <div id="historyList" class="historyList"></div>
        </div>

        <div id="historyScrollRail" class="historyScrollRail">
          <div id="historyScrollThumb" class="historyScrollThumb"></div>
        </div>
      </div>
    </section>
  `;
}

export function bindHistory(actions){
  bindHistoryStatsDismiss();

  const button = document.getElementById("historyClearButton");
  const scroller = document.getElementById("historyScroller");

  if(scroller){
    let tapStartX = 0;
    let tapStartY = 0;
    let tapChip = null;

    scroller.addEventListener("pointerdown", event => {
      tapChip = event.target.closest(
        ".historyChipWrap[data-history-number]"
      );

      tapStartX = event.clientX;
      tapStartY = event.clientY;
    });

    scroller.addEventListener("pointerup", event => {
      if(!tapChip) return;

      const dx = Math.abs(event.clientX - tapStartX);
      const dy = Math.abs(event.clientY - tapStartY);

      const originalChip = tapChip;
      tapChip = null;

      if(dx > 8 || dy > 8) return;

      const currentElement = document.elementFromPoint(
        event.clientX,
        event.clientY
      );

      const chip =
        currentElement?.closest(
          ".historyChipWrap[data-history-number]"
        ) ||
        (originalChip?.isConnected ? originalChip : null);

      if(!chip) return;

      const number = Number(chip.dataset.historyNumber);
      if(!Number.isFinite(number)) return;

      event.preventDefault();
      event.stopPropagation();

      showHistoryStats(number, chip);
    });

    scroller.addEventListener("pointercancel", () => {
      tapChip = null;
    });
  }

  if(!button) return;

  button.addEventListener("pointerup", async event => {
    event.preventDefault();
    event.stopPropagation();

    const items = getHistoryItems();
    if(!items.length) return;

    if(!button.classList.contains("confirm")){
      button.classList.add("confirm");
      button.innerHTML = `
        <span class="historyClearIcon">!</span>
        <span>Confirm</span>
      `;

      clearTimeout(clearConfirmTimer);
      clearConfirmTimer = setTimeout(resetClearButton, 3000);
      return;
    }

    clearTimeout(clearConfirmTimer);
    button.disabled = true;
    button.innerHTML = `
      <span class="historyClearSpinner"></span>
      <span>Clearing</span>
    `;

    try{
      await actions.clearHistory();
    }finally{
      button.disabled = false;
      resetClearButton();
    }
  });
}

export function renderHistory(){
  const box = document.getElementById("historyList");
  const count = document.getElementById("historyCount");

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

  if(count){
    count.textContent = String(items.length);
  }

  if(!items.length){
    box.innerHTML = `
      <div class="historyEmpty">
        <div class="historyEmptyIcon">—</div>
        <div>
          <div class="historyEmptyTitle">No history yet</div>
          <div class="historyEmptyText">Detected numbers will appear here</div>
        </div>
      </div>
    `;

    resetClearButton();
    requestAnimationFrame(updateHistoryScrollbar);
    return;
  }

  const { ranks } = getHistoryStats();

  box.innerHTML = items.map(item => {
    const base = chip(item.number);
    const cls = item.matched ? " historyHit" : "";

    const topRank =
      ranks.get(item.number) === 1
        ? '<span class="historyRankOneDot" aria-label="Most frequent"></span>'
        : "";

    return `
      <div
        class="historyChipWrap${cls}"
        data-history-number="${item.number}"
      >
        ${base}
        ${topRank}
      </div>
    `;
  }).join("");

  requestAnimationFrame(updateHistoryScrollbar);
}

function updateHistoryScrollbar(){
  const scroller = document.getElementById("historyScroller");
  const rail = document.getElementById("historyScrollRail");
  const thumb = document.getElementById("historyScrollThumb");

  if(!scroller || !rail || !thumb) return;

  const viewport = scroller.clientHeight;
  const content = scroller.scrollHeight;
  const railHeight = rail.clientHeight;

  if(content <= viewport){
    rail.classList.add("inactive");
    thumb.style.height = "100%";
    thumb.style.transform = "translateY(0)";
    return;
  }

  rail.classList.remove("inactive");

  const minThumbHeight = 42;
  const thumbHeight = Math.max(
    minThumbHeight,
    Math.round((viewport / content) * railHeight)
  );

  const maxThumbTravel = railHeight - thumbHeight;
  const maxScroll = content - viewport;

  const progress = maxScroll > 0
    ? scroller.scrollTop / maxScroll
    : 0;

  const thumbTop = Math.round(progress * maxThumbTravel);

  thumb.style.height = `${thumbHeight}px`;
  thumb.style.transform = `translateY(${thumbTop}px)`;
}

export function enableHistoryDrag(){
  const scroller = document.getElementById("historyScroller");
  const rail = document.getElementById("historyScrollRail");
  const thumb = document.getElementById("historyScrollThumb");

  if(!scroller) return;

  scroller.addEventListener("scroll", updateHistoryScrollbar);
  requestAnimationFrame(updateHistoryScrollbar);

  if(rail && thumb){
    let thumbDragging = false;
    let thumbStartY = 0;
    let scrollStartTop = 0;

    thumb.addEventListener("pointerdown", event => {
      thumbDragging = true;
      thumbStartY = event.clientY;
      scrollStartTop = scroller.scrollTop;

      thumb.classList.add("dragging");
      thumb.setPointerCapture(event.pointerId);

      event.preventDefault();
      event.stopPropagation();
    });

    thumb.addEventListener("pointermove", event => {
      if(!thumbDragging) return;

      const railHeight = rail.clientHeight;
      const thumbHeight = thumb.offsetHeight;
      const maxThumbTravel = railHeight - thumbHeight;
      const maxScroll = scroller.scrollHeight - scroller.clientHeight;

      if(maxThumbTravel <= 0 || maxScroll <= 0) return;

      const dy = event.clientY - thumbStartY;

      scroller.scrollTop =
        scrollStartTop +
        (dy / maxThumbTravel) * maxScroll;

      event.preventDefault();
      event.stopPropagation();
    });

    const stopThumbDrag = event => {
      if(!thumbDragging) return;

      thumbDragging = false;
      thumb.classList.remove("dragging");

      event.preventDefault();
      event.stopPropagation();
    };

    thumb.addEventListener("pointerup", stopThumbDrag);
    thumb.addEventListener("pointercancel", stopThumbDrag);
  }

  let isDown = false;
  let startY = 0;
  let startScrollTop = 0;
  let activePointerId = null;

  scroller.addEventListener("pointerdown", event => {
    if(event.target.closest("button")) return;

    isDown = true;
    activePointerId = event.pointerId;
    startY = event.clientY;
    startScrollTop = scroller.scrollTop;

    scroller.classList.add("dragging");
    scroller.setPointerCapture(event.pointerId);
  });

  scroller.addEventListener("pointermove", event => {
    if(!isDown || event.pointerId !== activePointerId) return;

    const dy = event.clientY - startY;
    scroller.scrollTop = startScrollTop - dy;

    event.preventDefault();
  });

  const stopDrag = event => {
    if(
      event &&
      activePointerId !== null &&
      event.pointerId !== activePointerId
    ){
      return;
    }

    isDown = false;
    activePointerId = null;
    scroller.classList.remove("dragging");
  };

  scroller.addEventListener("pointerup", stopDrag);
  scroller.addEventListener("pointercancel", stopDrag);
}
