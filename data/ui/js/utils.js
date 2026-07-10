import { RED_NUMBERS } from "./state.js";

export function numberClass(n){
  n = Number(n);
  if(n === 0) return "numGreen";
  return RED_NUMBERS.has(n) ? "numRed" : "numBlack";
}

export function chip(n, extra = ""){
  return `<div class="numChip ${numberClass(n)} ${extra}">${n}</div>`;
}

export function shortTime(timestamp){
  return (timestamp || "").split("T").pop().slice(0, 8);
}

export function setSwitch(id, on){
  const el = document.getElementById(id);
  if(el){
    el.classList.toggle("on", !!on);
  }
}
