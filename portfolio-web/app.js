const form = document.getElementById("demo-form");
const resultState = document.getElementById("result-state");
const queueState = document.getElementById("queue-state");
const timeline = document.getElementById("timeline");
const taskId = document.getElementById("task-id");
const metaModel = document.getElementById("meta-model");
const model = document.getElementById("model");
const seed = document.getElementById("seed");
const promptInput = document.getElementById("prompt");
const resultImage = document.getElementById("result-image");

function paletteFromText(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  const a = (hash >>> 0).toString(16).padStart(8, "0");
  return [`#${a.slice(0, 6)}`, `#${a.slice(2, 8)}`, `#${a.slice(1, 7)}`];
}

function setTimeline(items) {
  timeline.innerHTML = items
    .map((item) => `<li class="${item.state}">${item.label}</li>`)
    .join("");
}

function renderResult() {
  const [a, b, c] = paletteFromText(`${promptInput.value}|${seed.value}|${model.value}`);
  resultImage.innerHTML = `
    <div class="mock-art" style="background:
      linear-gradient(145deg, ${a}, ${b} 48%, ${c}),
      repeating-linear-gradient(45deg, rgba(255,255,255,.14) 0 1px, transparent 1px 18px);">
      <span>${model.value} mock output</span>
      <strong>Seed ${seed.value || "auto"}</strong>
    </div>
  `;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  document.body.classList.add("is-running");
  resultState.textContent = "running";
  queueState.textContent = "running";
  taskId.textContent = `txt2img_demo_${seed.value || Date.now()}`;
  metaModel.textContent = model.value;
  setTimeline([
    { label: "Request validated", state: "done" },
    { label: "Task enqueued", state: "running" },
    { label: "Mock renderer waiting", state: "" },
    { label: "History + metadata pending", state: "" },
  ]);

  window.setTimeout(() => {
    setTimeline([
      { label: "Request validated", state: "done" },
      { label: "Task enqueued", state: "done" },
      { label: "Mock renderer completed", state: "done" },
      { label: "History + metadata saved", state: "done" },
    ]);
    renderResult();
    resultState.textContent = "completed";
    queueState.textContent = "completed";
    document.body.classList.remove("is-running");
  }, 850);
});
