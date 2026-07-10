(function(){
  function openCalibration(){
    window.location.href = "http://" + window.location.hostname + ":8081/";
  }

  function addCalibrationButton(){
    if(document.getElementById("calibrationFixedButton")) return;

    const btn = document.createElement("button");
    btn.id = "calibrationFixedButton";
    btn.type = "button";
    btn.textContent = "Calibration";
    btn.onclick = openCalibration;

    btn.style.position = "fixed";
    btn.style.top = "12px";
    btn.style.right = "150px";
    btn.style.zIndex = "999999";
    btn.style.height = "42px";
    btn.style.padding = "0 16px";
    btn.style.border = "0";
    btn.style.borderRadius = "16px";
    btn.style.background = "#315b7c";
    btn.style.color = "#ffffff";
    btn.style.fontSize = "14px";
    btn.style.fontWeight = "950";
    btn.style.boxShadow = "0 8px 20px rgba(31,64,98,.25)";
    btn.style.cursor = "pointer";

    document.body.appendChild(btn);
  }

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", addCalibrationButton);
  }else{
    addCalibrationButton();
  }

  setInterval(addCalibrationButton, 1000);
})();
