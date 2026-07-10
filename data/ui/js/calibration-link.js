(function(){
  function makeButton(){
    const btn = document.createElement("button");
    btn.id = "openCalibrationBtn";
    btn.type = "button";
    btn.textContent = "Calibration";
    btn.style.height = "38px";
    btn.style.border = "0";
    btn.style.borderRadius = "14px";
    btn.style.padding = "0 14px";
    btn.style.fontWeight = "900";
    btn.style.background = "#315b7c";
    btn.style.color = "#fff";
    btn.style.boxShadow = "0 6px 16px rgba(31,64,98,.16)";
    btn.onclick = () => {
      window.location.href = "/calibrate.html?v=" + Date.now();
    };
    return btn;
  }

  function add(){
    if(document.getElementById("openCalibrationBtn")) return true;

    const controls = Array.from(document.querySelectorAll("button,a"));
    const settings = controls.find(el => {
      const t = (el.textContent || "").trim().toLowerCase();
      return t === "settings" || t.includes("settings");
    });

    const btn = makeButton();

    if(settings && settings.parentElement){
      settings.insertAdjacentElement("afterend", btn);
      return true;
    }

    const header =
      document.querySelector("header") ||
      document.querySelector(".headerActions") ||
      document.querySelector(".topActions") ||
      document.querySelector(".appHeader");

    if(header){
      header.appendChild(btn);
      return true;
    }

    btn.style.position = "fixed";
    btn.style.top = "10px";
    btn.style.right = "120px";
    btn.style.zIndex = "9999";
    document.body.appendChild(btn);
    return true;
  }

  const timer = setInterval(() => {
    if(add()) clearInterval(timer);
  }, 500);

  window.addEventListener("load", add);
})();
