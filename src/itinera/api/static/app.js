const POLL_MS = 2500;
const SETTLED = ["awaiting_review", "completed", "failed", "cancelled"];

const STAGE_ORDER = [
  "pending",
  "researching",
  "planning",
  "awaiting_review",
  "revising",
  "finalizing",
  "completed",
];

const TONES = {
  pending: "run",
  researching: "run",
  planning: "run",
  revising: "run",
  finalizing: "run",
  awaiting_review: "wait",
  completed: "done",
  failed: "fail",
  cancelled: "fail",
};

const PHASE_PLAN = [
  { stage: "researching", label: "Researching the destination", seconds: 120 },
  { stage: "planning", label: "Drafting the itinerary", seconds: 100 },
  { stage: "revising", label: "Revising the itinerary", seconds: 120 },
  { stage: "finalizing", label: "Assembling the final plan", seconds: 50 },
];

const el = (sel, root = document) => root.querySelector(sel);
const els = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const role = (name, root = document) => root.querySelector(`[data-role="${name}"]`);

const panels = {
  request: el("#requestPanel"),
  run: el("#runPanel"),
  draft: el("#draftPanel"),
  review: el("#reviewPanel"),
  final: el("#finalPanel"),
};

let planId = null;
let pollTimer = null;
let tickTimer = null;
let busy = false;
let clock = { elapsed: 0, at: 0, running: false };

function show(panel, visible) {
  panel.hidden = !visible;
}

function toast(message, tone) {
  const node = el("#toast");
  node.textContent = message;
  if (tone) {
    node.dataset.tone = tone;
  } else {
    delete node.dataset.tone;
  }
  node.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    node.hidden = true;
  }, tone === "error" ? 7000 : 3500);
}

function money(value, currency) {
  if (value === null || value === undefined) return "—";
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: currency || "USD",
      maximumFractionDigits: 0,
    }).format(value);
  } catch {
    return `${Math.round(value)} ${currency || ""}`.trim();
  }
}

function clockTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString();
}

async function problemFrom(response) {
  try {
    const body = await response.json();
    return body.detail || body.title || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(await problemFrom(response));
  return response.json();
}

function readForm() {
  const value = (id) => el(`#${id}`).value.trim();
  const interests = value("interests")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  return {
    destination: value("destination"),
    start_date: value("start_date"),
    end_date: value("end_date"),
    budget_min: Number(value("budget_min")),
    budget_max: Number(value("budget_max")),
    currency: value("currency").toUpperCase() || "USD",
    interests,
    travelers: Number(value("travelers")),
  };
}

function localProblem(body) {
  if (!body.destination) return "Name a destination.";
  if (!body.start_date || !body.end_date) return "Both dates are required.";
  if (body.end_date < body.start_date) return "The end date falls before the start date.";
  if (!(body.budget_max > body.budget_min)) return "Budget max must exceed budget min.";
  if (!(body.travelers >= 1)) return "There must be at least one traveller.";
  return null;
}

async function submitRequest(event) {
  event.preventDefault();
  const body = readForm();
  const problem = localProblem(body);
  if (problem) {
    toast(problem, "error");
    return;
  }

  const button = el("#submitBtn");
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const created = await api("/plan", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": crypto.randomUUID(),
      },
      body: JSON.stringify(body),
    });
    adopt(created.plan_id);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Start planning";
  }
}

function adopt(id) {
  planId = id;
  window.location.hash = id;
  show(panels.request, false);
  show(panels.run, true);
  refresh();
}

const PHASE_TO_STAGE = {
  research: "researching",
  planner: "planning",
  review: "awaiting_review",
  finalize: "finalizing",
};

function effectiveStage(plan) {
  const stored = plan.status === "pending" ? "researching" : plan.status;
  if (SETTLED.includes(plan.status)) return stored;

  const notes = (plan.progress && plan.progress.notes) || [];
  const latest = notes[notes.length - 1];
  const fromNote = latest && PHASE_TO_STAGE[latest.phase];
  if (!fromNote) return stored;

  return STAGE_ORDER.indexOf(fromNote) > STAGE_ORDER.indexOf(stored) ? fromNote : stored;
}

function renderStages(effective) {
  const current = STAGE_ORDER.indexOf(effective);
  els("li", role("stages")).forEach((node) => {
    const at = STAGE_ORDER.indexOf(node.dataset.stage);
    let state = "todo";
    if (effective === "completed") {
      state = "done";
    } else if (at === current || (effective === "revising" && node.dataset.stage === "planning")) {
      state = "active";
    } else if (at > -1 && current > -1 && at < current) {
      state = "done";
    }
    node.dataset.state = state;
  });
}

function mmss(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function progressFraction(status, elapsed) {
  const stage = status === "pending" ? "researching" : status;
  if (stage === "completed") return 1;

  const index = PHASE_PLAN.findIndex((p) => p.stage === stage);
  if (index < 0) return null;

  const total = PHASE_PLAN.reduce((sum, p) => sum + p.seconds, 0);
  const before = PHASE_PLAN.slice(0, index).reduce((sum, p) => sum + p.seconds, 0);
  const phase = PHASE_PLAN[index];

  const into = Math.max(elapsed - before, 0);
  const share = 1 - Math.exp(-into / phase.seconds);
  return Math.min((before + share * phase.seconds) / total, 0.97);
}

function renderProgress(plan) {
  const holder = role("progress");
  const running = !["awaiting_review", "completed", "failed", "cancelled"].includes(
    plan.status
  );
  const progress = plan.progress;
  holder.hidden = !running;
  if (!running) {
    clock.running = false;
    return;
  }

  const stage = effectiveStage(plan);
  const elapsed = progress ? progress.elapsed_seconds : 0;
  clock = { elapsed, at: Date.now(), running: true, status: stage };
  const fraction = progressFraction(stage, elapsed);

  role("bar").style.width = `${Math.round((fraction ?? 0.05) * 100)}%`;
  role("elapsed").textContent = mmss(elapsed);

  const total = PHASE_PLAN.reduce((sum, p) => sum + p.seconds, 0);
  const left = total - elapsed;
  role("eta").textContent =
    left > 20 ? `· about ${mmss(left)} left` : "· taking longer than usual";

  const latest = progress && progress.notes && progress.notes[progress.notes.length - 1];
  const phase = PHASE_PLAN.find((p) => p.stage === stage);
  role("now").textContent = latest ? latest.message : (phase ? phase.label : "Working…");
}

function renderFeed(plan) {
  const holder = role("feed");
  const notes = (plan.progress && plan.progress.notes) || [];
  holder.hidden = notes.length === 0;
  if (!notes.length) return;

  role("feedCount").textContent = notes.length;
  role("notes").innerHTML = notes
    .slice(-12)
    .reverse()
    .map(
      (n) => `<li>
        <time>${mmss(n.elapsed_seconds)}</time>
        <span class="feed__phase">${escapeHtml(n.phase)}</span>
        <span class="feed__msg">${escapeHtml(n.message)}</span>
      </li>`
    )
    .join("");
}

function renderStatus(plan) {
  role("destination").textContent = plan.destination;
  role("planId").textContent = plan.plan_id;

  const badge = role("status");
  badge.textContent = plan.status.replace(/_/g, " ");
  badge.dataset.tone = TONES[plan.status] || "run";

  renderStages(effectiveStage(plan));

  const bits = [`<b>${plan.revision_count}</b> revision${plan.revision_count === 1 ? "" : "s"}`];
  const remaining = plan.review_request && plan.review_request.revisions_remaining;
  if (remaining !== undefined && remaining !== null) {
    bits.push(`<b>${remaining}</b> revisions remaining`);
  }
  bits.push(`updated <b>${clockTime(plan.updated_at)}</b>`);
  role("meta").innerHTML = bits.map((bit) => `<span>${bit}</span>`).join("");

  const error = role("error");
  error.hidden = !plan.error;
  if (plan.error) error.textContent = plan.error;

  role("events").innerHTML = (plan.events || [])
    .map(
      (e) =>
        `<li><time>${clockTime(e.recorded_at)}</time>${e.event.replace(/_/g, " ")}</li>`
    )
    .join("");
}

function renderBudget(budget) {
  const holder = role("budget");
  if (!budget || !budget.lines) {
    holder.innerHTML = "";
    return;
  }
  const cards = [
    `<div class="budget__item">
       <div class="budget__label">Total</div>
       <div class="budget__value">${money(budget.total_budget, budget.currency)}</div>
       <div class="budget__sub">${budget.travelers} travellers · ${budget.duration_days} days</div>
     </div>`,
  ];
  for (const line of budget.lines) {
    cards.push(
      `<div class="budget__item">
         <div class="budget__label">${line.category}</div>
         <div class="budget__value">${money(line.total, budget.currency)}</div>
         <div class="budget__sub">${money(line.per_person_per_day, budget.currency)} pp/day</div>
       </div>`
    );
  }
  holder.innerHTML = cards.join("");
}

function renderActivity(activity, index) {
  const facts = [];
  if (activity.start_time) facts.push(activity.start_time.slice(0, 5));
  if (activity.duration_hours) facts.push(`${activity.duration_hours}h`);
  if (activity.neighbourhood) facts.push(activity.neighbourhood);
  if (activity.estimated_cost_per_person)
    facts.push(`${activity.estimated_cost_per_person} pp`);
  if (activity.booking_note && activity.booking_note.toLowerCase() !== "none required.")
    facts.push(activity.booking_note);

  const cites = (activity.source_urls || [])
    .slice(0, 3)
    .map((u, i) => `<a class="act__cite" href="${u}" target="_blank" rel="noreferrer">[${i + 1}]</a>`)
    .join(" ");

  return `<div class="act">
    <div class="act__idx">${index + 1}</div>
    <div class="act__body">
      <div class="act__name">${escapeHtml(activity.name)}
        <span class="tag">${activity.kind}</span></div>
      ${activity.description ? `<div class="act__desc">${escapeHtml(activity.description)}</div>` : ""}
      <div class="act__facts">${facts.map(escapeHtml).join("<span>·</span>")} ${cites}</div>
    </div>
  </div>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function renderDraft(draft) {
  if (!draft) {
    show(panels.draft, false);
    return;
  }
  show(panels.draft, true);
  role("summary").textContent = draft.summary || "";
  renderBudget(draft.budget);

  role("days").innerHTML = (draft.days || [])
    .map(
      (day) => `<div class="day">
        <div class="day__head">
          <span class="day__no">Day ${day.day_number}</span>
          <span class="day__theme">${escapeHtml(day.theme || "")}</span>
          <span class="day__date">${day.day_date || ""}</span>
        </div>
        ${day.notes ? `<div class="day__notes">${escapeHtml(day.notes)}</div>` : ""}
        ${(day.activities || []).map(renderActivity).join("") ||
          `<div class="act"><div class="act__body act__desc">Nothing scheduled.</div></div>`}
      </div>`
    )
    .join("");

  const dining = (draft.days || []).flatMap((day) =>
    (day.dining || []).map((d) => ({ ...d, day: day.day_number }))
  );
  role("dining").innerHTML = dining.length
    ? `<h3 class="subhead">Dining</h3><ul>${dining
        .map(
          (d) =>
            `<li>${escapeHtml(d.name)} <span>· day ${d.day} · ${escapeHtml(
              d.cuisine || ""
            )} · ${escapeHtml(d.price_band || "")}</span></li>`
        )
        .join("")}</ul>`
    : "";
}

function renderReview(plan) {
  const open = plan.awaiting_review;
  show(panels.review, open);
  if (!open) return;
  const remaining = (plan.review_request || {}).revisions_remaining;
  role("revisions").textContent =
    remaining === undefined || remaining === null
      ? "The workflow is paused until you decide."
      : `The workflow is paused until you decide. ${remaining} revision${
          remaining === 1 ? "" : "s"
        } remaining before the plan is finalised automatically.`;
}

async function renderFinal() {
  const final = await api(`/plan/${planId}/final`);
  show(panels.final, true);
  show(panels.review, false);

  role("finalMeta").textContent = `Approved after ${final.revision_count} revision${
    final.revision_count === 1 ? "" : "s"
  }${final.revision_limit_reached ? " · revision limit reached" : ""}.`;

  role("packing").innerHTML = (final.packing_list || [])
    .map((i) => `<li>${escapeHtml(i)}</li>`)
    .join("");
  role("checklist").innerHTML = (final.preparation_checklist || [])
    .map((i) => `<li>${escapeHtml(i)}</li>`)
    .join("");
  role("weather").textContent = final.weather_summary || "";

  const sources = final.sources || [];
  role("sourceCount").textContent = sources.length;
  role("sources").innerHTML = sources
    .map((u) => `<li><a href="${u}" target="_blank" rel="noreferrer">${escapeHtml(u)}</a></li>`)
    .join("");

  renderDraft(final.itinerary);
}

function schedule(status) {
  clearTimeout(pollTimer);
  if (!SETTLED.includes(status)) pollTimer = setTimeout(refresh, POLL_MS);
}

function startTicking() {
  clearInterval(tickTimer);
  tickTimer = setInterval(() => {
    if (!clock.running) return;
    const elapsed = clock.elapsed + (Date.now() - clock.at) / 1000;
    role("elapsed").textContent = mmss(elapsed);
    const fraction = progressFraction(clock.status, elapsed);
    if (fraction !== null) role("bar").style.width = `${Math.round(fraction * 100)}%`;
  }, 1000);
}

async function refresh() {
  if (!planId || busy) return;
  try {
    const plan = await api(`/plan/${planId}`);
    renderStatus(plan);
    renderProgress(plan);
    renderFeed(plan);
    renderDraft(plan.draft);
    renderReview(plan);
    if (plan.status === "completed") await renderFinal();
    schedule(plan.status);
  } catch (error) {
    toast(error.message, "error");
    schedule("failed");
  }
}

function buildReviewBody(action) {
  if (action === "approve") return { action: "approve" };

  if (action === "reject") {
    const feedback = el("#feedback").value.trim();
    if (!feedback) throw new Error("Rejecting needs feedback the agents can act on.");
    return { action: "reject", feedback };
  }

  const operation = el("#operation").value;
  if (operation === "instruction") {
    const instruction = el("#instruction").value.trim();
    if (!instruction) throw new Error("Describe the change you want.");
    return { action: "modify", instruction };
  }

  const patch = { operation };
  if (["remove_activity", "add_activity", "replace_activity", "set_day_notes"].includes(operation)) {
    patch.day_number = Number(el("#dayNumber").value);
  }
  if (["remove_activity", "replace_activity", "add_activity"].includes(operation)) {
    patch.activity_index = Number(el("#activityIndex").value);
  }
  if (["set_day_notes", "set_lodging_preference"].includes(operation)) {
    const value = el("#patchValue").value.trim();
    if (!value) throw new Error("This operation needs a value.");
    patch.value = value;
  }
  if (["add_activity", "replace_activity"].includes(operation)) {
    const name = el("#actName").value.trim();
    if (!name) throw new Error("The activity needs a name.");
    patch.activity = {
      name,
      kind: el("#actKind").value,
      duration_hours: Number(el("#actHours").value) || 2,
      estimated_cost_per_person: Number(el("#actCost").value) || 0,
      neighbourhood: el("#actArea").value.trim() || null,
      description: el("#actDesc").value.trim(),
      source_urls: [],
    };
  }
  return { action: "modify", patches: [patch] };
}

async function submitReview(action) {
  if (busy) return;
  let body;
  try {
    body = buildReviewBody(action);
  } catch (error) {
    toast(error.message, "error");
    return;
  }

  busy = true;
  clearTimeout(pollTimer);
  const buttons = els("[data-action]");
  buttons.forEach((b) => (b.disabled = true));
  const pressed = el(`[data-action="${action}"]`);
  const label = pressed.textContent;
  pressed.textContent = action === "modify" ? "Applying…" : "Working…";

  try {
    const result = await api(`/plan/${planId}/review`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    toast(result.message || "Review recorded.");
    el("#feedback").value = "";
    el("#instruction").value = "";
  } catch (error) {
    toast(error.message, "error");
  } finally {
    busy = false;
    buttons.forEach((b) => (b.disabled = false));
    pressed.textContent = label;
    refresh();
  }
}

function syncOperands() {
  const operation = el("#operation").value;
  const needs = {
    remove_activity: ["day", "index"],
    set_day_notes: ["day", "value"],
    set_lodging_preference: ["value"],
    add_activity: ["day", "index", "activity"],
    replace_activity: ["day", "index", "activity"],
    instruction: ["instruction"],
  }[operation] || [];

  els("[data-operand]").forEach((node) => {
    node.hidden = !needs.includes(node.dataset.operand);
  });
}

function fillSample() {
  const start = new Date();
  start.setDate(start.getDate() + 45);
  const end = new Date(start);
  end.setDate(end.getDate() + 3);
  const iso = (d) => d.toISOString().slice(0, 10);

  el("#destination").value = "Kyoto, Japan";
  el("#start_date").value = iso(start);
  el("#end_date").value = iso(end);
  el("#budget_min").value = 1800;
  el("#budget_max").value = 2600;
  el("#currency").value = "USD";
  el("#interests").value = "temples, food, autumn foliage";
  el("#travelers").value = 2;
}

function reset() {
  clearTimeout(pollTimer);
  planId = null;
  window.location.hash = "";
  [panels.run, panels.draft, panels.review, panels.final].forEach((p) => show(p, false));
  show(panels.request, true);
}

function init() {
  el("#requestForm").addEventListener("submit", submitRequest);
  el("#sampleBtn").addEventListener("click", fillSample);
  el("#newPlanBtn").addEventListener("click", reset);
  el("#operation").addEventListener("change", syncOperands);

  els(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      els(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      els(".tabpanel").forEach((p) =>
        p.classList.toggle("is-active", p.dataset.panel === tab.dataset.tab)
      );
    });
  });

  els("[data-action]").forEach((button) => {
    button.addEventListener("click", () => submitReview(button.dataset.action));
  });

  syncOperands();
  startTicking();

  const existing = window.location.hash.slice(1);
  if (existing) adopt(existing);
}

document.addEventListener("DOMContentLoaded", init);