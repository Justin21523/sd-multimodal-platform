import "./styles.css";

const project = {
  "slug": "sd-multimodal-platform",
  "title": "SD Multi-Modal Platform: Portfolio-Ready Stable Diffusion Platform",
  "summary": "SD Multi-Modal Platform is a FastAPI + React generative AI platform demo. The refreshed version keeps the full-mode Stable Diffusion / SDXL / ControlNet / post-processing architecture, while adding a mock-safe demo mode so the public GitHub Pages site, screenshots, walkthrough video, and local smoke tests work without GPU access, model weights, Redis, or external services.",
  "category": "ai-data",
  "year": 2026,
  "status": "completed",
  "technologies": [
    "Python",
    "FastAPI",
    "React",
    "TypeScript",
    "Vite",
    "PyTorch",
    "Diffusers",
    "Hugging Face Transformers",
    "Redis",
    "Celery",
    "Pydantic",
    "Pillow",
    "Docker",
    "GitHub Pages"
  ],
  "githubUrl": "https://github.com/Justin21523/sd-multimodal-platform",
  "readmeUrl": "https://github.com/Justin21523/sd-multimodal-platform#readme",
  "problem": "Stable Diffusion projects often remain stuck in a local-only state: reviewers need model weights, CUDA, Redis, Celery, post-processing weights, and environment-specific paths before they can understand the value. This repo already had many API and service modules, but it needed a stable public entrypoint for demos, screenshots, video capture, and smoke testing.",
  "solution": "I reorganized the project around two modes. Full mode preserves the ModelRegistry, Diffusers pipelines, Redis/Celery queue, assets/history, and post-processing services. Mock-safe mode uses deterministic PIL rendering behind the same FastAPI routes, so the workflow remains demonstrable without large models or GPU hardware. The React workbench can connect to the mock API, and the repo now includes a GitHub Pages static demo, Playwright screenshot/video capture, README Mermaid diagrams, and an interview runbook.",
  "architecture": "This case study is generated from the portfolio catalog pipeline using README, Git metadata, package/build configuration, and media signals. The final architecture narrative still needs source-level review. Current detected technology signals include: Python, FastAPI, React, TypeScript, Vite, PyTorch, Diffusers, Hugging Face Transformers, Redis, Celery, Pydantic, Pillow, Docker, GitHub Pages.",
  "setupGuide": "This project does not expose a verified runnable web command yet. Review the README/source tree and add exact install, run, test, and build commands before interview use.\nNo verified build command was detected. Treat the current portfolio page as a case-study placeholder until build steps are reviewed.",
  "features": [
    "Detected technical signals: Python, FastAPI, React, TypeScript, Vite, PyTorch, Diffusers, Hugging Face Transformers, Redis, Celery, Pydantic, Pillow, Docker, GitHub Pages,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability",
    "I reorganized the project around two modes",
    "Full mode preserves the ModelRegistry, Diffusers pipelines, Redis/Celery queue, assets/history, and post-processing services",
    "Mock-safe mode uses deterministic PIL rendering behind the same FastAPI routes, so the workflow remains demonstrable without large models or GPU hardware",
    "The React workbench can connect to the mock API, and the repo now includes a GitHub Pages static demo, Playwright screenshot/video capture, README Mermaid diagrams, and an interview runbook"
  ],
  "metrics": [
    {
      "label": "Demo Modules",
      "value": "5"
    },
    {
      "label": "Tech Stack",
      "value": "14"
    },
    {
      "label": "Mode",
      "value": "Fixture"
    },
    {
      "label": "Status",
      "value": "completed"
    }
  ],
  "records": [
    {
      "id": "flow-01",
      "name": "Detected technical signals: Python, FastAPI, React, TypeScript, Vite, PyTorch, Diffusers, Hugging Face Transformers, Redis, Celery, Pydantic, Pillow, Docker, GitHub Pages,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability",
      "status": "Ready",
      "owner": "Frontend"
    },
    {
      "id": "flow-02",
      "name": "I reorganized the project around two modes",
      "status": "Review",
      "owner": "Data"
    },
    {
      "id": "flow-03",
      "name": "Full mode preserves the ModelRegistry, Diffusers pipelines, Redis/Celery queue, assets/history, and post-processing services",
      "status": "Queued",
      "owner": "Automation"
    },
    {
      "id": "flow-04",
      "name": "Mock-safe mode uses deterministic PIL rendering behind the same FastAPI routes, so the workflow remains demonstrable without large models or GPU hardware",
      "status": "Ready",
      "owner": "Product"
    },
    {
      "id": "flow-05",
      "name": "The React workbench can connect to the mock API, and the repo now includes a GitHub Pages static demo, Playwright screenshot/video capture, README Mermaid diagrams, and an interview runbook",
      "status": "Review",
      "owner": "Quality"
    }
  ]
};

const state = {
  tab: "overview",
  query: "",
  selected: project.records[0]?.id ?? "",
};

function matches(record) {
  const q = state.query.trim().toLowerCase();
  if (!q) return true;
  return [record.name, record.status, record.owner].join(" ").toLowerCase().includes(q);
}

function renderMetrics() {
  return project.metrics.map((metric) => `
    <div class="metric">
      <span>${metric.label}</span>
      <strong>${metric.value}</strong>
    </div>
  `).join("");
}

function renderTabs() {
  return ["overview", "workflow", "data", "architecture"].map((tab) => `
    <button class="tab ${state.tab === tab ? "active" : ""}" data-tab="${tab}">${tab}</button>
  `).join("");
}

function renderOverview() {
  return `
    <section class="panel hero-panel">
      <div>
        <p class="eyebrow">${project.category} · ${project.year}</p>
        <h1>${project.title}</h1>
        <p class="lead">${project.summary}</p>
      </div>
      <div class="metrics">${renderMetrics()}</div>
    </section>
    <section class="panel split">
      <div>
        <h2>Problem</h2>
        <p>${project.problem}</p>
      </div>
      <div>
        <h2>Solution</h2>
        <p>${project.solution}</p>
      </div>
    </section>
  `;
}

function renderWorkflow() {
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Demo workflow</p>
          <h2>Interactive Review Flow</h2>
        </div>
        <button id="runDemo" class="primary">Run demo pass</button>
      </div>
      <div class="timeline">
        ${project.features.map((feature, index) => `
          <article class="step">
            <span>${String(index + 1).padStart(2, "0")}</span>
            <p>${feature}</p>
          </article>
        `).join("")}
      </div>
      <output id="demoOutput" class="output">Ready to run the guided demo.</output>
    </section>
  `;
}

function renderData() {
  const rows = project.records.filter(matches);
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Fixture data</p>
          <h2>Sample Records</h2>
        </div>
        <input id="search" value="${state.query}" placeholder="Filter records" />
      </div>
      <div class="table">
        ${rows.map((record) => `
          <button class="row ${state.selected === record.id ? "selected" : ""}" data-record="${record.id}">
            <span>${record.id}</span>
            <strong>${record.name}</strong>
            <em>${record.owner}</em>
            <b>${record.status}</b>
          </button>
        `).join("") || `<p class="empty">No records match this filter.</p>`}
      </div>
    </section>
  `;
}

function renderArchitecture() {
  return `
    <section class="panel split">
      <div>
        <p class="eyebrow">Architecture</p>
        <h2>How the demo is organized</h2>
        <p>${project.architecture}</p>
        <pre>demo-app/
  src/main.js
  src/styles.css
  index.html
  package.json</pre>
      </div>
      <div>
        <p class="eyebrow">Run guide</p>
        <h2>Local commands</h2>
        <pre>${project.setupGuide}</pre>
        <div class="chips">${project.technologies.slice(0, 12).map((tech) => `<span>${tech}</span>`).join("")}</div>
      </div>
    </section>
  `;
}

function render() {
  const views = {
    overview: renderOverview,
    workflow: renderWorkflow,
    data: renderData,
    architecture: renderArchitecture,
  };
  document.querySelector("#app").innerHTML = `
    <header class="topbar">
      <a href="${project.githubUrl}" class="brand">${project.title}</a>
      <nav>${renderTabs()}</nav>
      <a class="repo" href="${project.readmeUrl}">README</a>
    </header>
    <main>${views[state.tab]()}</main>
  `;

  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.tab = button.dataset.tab;
      render();
    });
  });
  document.querySelector("#search")?.addEventListener("input", (event) => {
    state.query = event.target.value;
    render();
    document.querySelector("#search")?.focus();
  });
  document.querySelectorAll("[data-record]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selected = button.dataset.record;
      render();
    });
  });
  document.querySelector("#runDemo")?.addEventListener("click", () => {
    const output = document.querySelector("#demoOutput");
    if (output) output.textContent = `${project.title}: ${project.records.length} fixture records processed and ${project.features.length} workflow checks completed.`;
  });
}

render();
