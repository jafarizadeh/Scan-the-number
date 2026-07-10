export function shutdownModalHTML(){
  return `
    <div id="shutdownModal" class="modal">
      <div class="modalBox shutdownBox">
        <div class="shutdownIcon">
          <svg viewBox="0 0 24 24">
            <path d="M12 2v10"></path>
            <path d="M6.7 6.7a7.5 7.5 0 1 0 10.6 0"></path>
          </svg>
        </div>

        <div class="shutdownTitle">Safe Power Off</div>

        <div class="shutdownText">
          This will safely shut down the Raspberry Pi. Wait until the screen goes off before unplugging power.
        </div>

        <div id="shutdownMessage" class="shutdownMessage">
          Confirm shutdown?
        </div>

        <div class="shutdownActions">
          <button id="cancelShutdownBtn" class="shutdownCancel">Cancel</button>
          <button id="confirmShutdownBtn" class="shutdownConfirm">Power Off</button>
        </div>
      </div>
    </div>
  `;
}

export function bindShutdownModal(actions){
  document.getElementById("cancelShutdownBtn").onclick = actions.closeShutdownModal;
  document.getElementById("confirmShutdownBtn").onclick = actions.confirmShutdown;
}

export function openShutdownModal(){
  document.getElementById("shutdownModal").classList.add("show");
}

export function closeShutdownModal(){
  document.getElementById("shutdownModal").classList.remove("show");
}

export function setShutdownMessage(text){
  const el = document.getElementById("shutdownMessage");
  if(el) el.textContent = text;
}
