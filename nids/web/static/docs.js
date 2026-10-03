"use strict";
const host = document.getElementById("docs-content");
function safe(value) { return String(value || "").replace(/[&<>"']/g, (x) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x])); }
async function load() {
  try {
    const response = await fetch("/api/openapi.json");
    if (!response.ok) throw new Error("OpenAPI specification could not be loaded.");
    const spec = await response.json();
    const groups = new Map();
    for (const [path, methods] of Object.entries(spec.paths || {})) {
      for (const [method, operation] of Object.entries(methods)) {
        if (!["get", "post", "put", "delete", "patch"].includes(method)) continue;
        const tag = operation.tags?.[0] || "other";
        if (!groups.has(tag)) groups.set(tag, []);
        groups.get(tag).push({path, method, operation});
      }
    }
    host.innerHTML = [...groups.entries()].map(([tag, items]) => `<section class="docs-group"><h2>${safe(tag.charAt(0).toUpperCase() + tag.slice(1))}</h2>${items.map(({path, method, operation}) => `<article class="endpoint-card"><span class="endpoint-method ${method === "post" ? "post" : ""}">${safe(method.toUpperCase())}</span><code class="endpoint-path">${safe(path)}</code><div class="endpoint-summary">${safe(operation.summary)}<small>${safe(operation.description || "See the OpenAPI JSON for schemas and response details.")}</small></div></article>`).join("")}</section>`).join("");
  } catch (error) {
    host.textContent = error.message;
  }
}
load();
