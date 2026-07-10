export async function getJSON(url){
  const response = await fetch(url);
  return await response.json();
}

export async function postJSON(url, data = {}){
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });
  return await response.json();
}
