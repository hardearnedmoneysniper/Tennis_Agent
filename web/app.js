const state = {
  jobId: null,
  result: null,
  selectedFile: null,
  pollTimer: null,
  activeStrokeType: null,
  chatHistory: [],
  llm: { provider: "local", ready: false, model: null, reason: "" },
  sendingChat: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));

const pct = (value) => `${Math.max(0, Math.min(100, Number(value) || 0))}%`;

function formatNumber(value, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(decimals);
}

function formatTime(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(total / 60);
  const rest = total - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

function statusLabel(status) {
  return { good: "良好", ok: "可接受", poor: "需改进", unknown: "无数据" }[status] || "—";
}

function statusLabelLong(status) {
  return { good: "在参考区间", ok: "接近参考区间", poor: "偏离参考区间", unknown: "数据不足" }[status] || "";
}

function severityLabel(severity) {
  return { high: "高优先级", medium: "中优先级", low: "低优先级" }[severity] || "建议";
}

function providerLabel(llm) {
  if (llm && llm.provider === "claude") return `Claude · ${llm.model || "已连接"}`;
  return "本地规则教练";
}

function showView(name) {
  ["uploadView", "progressView", "reportView"].forEach((id) => {
    $(`#${id}`).classList.toggle("hidden", id !== `${name}View`);
  });
  window.scrollTo({ top: 0, behavior: "auto" });
}

function showToast(message, timeout = 4200) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => toast.classList.add("hidden"), timeout);
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  let data = null;
  try {
    data = await response.json();
  } catch (error) {
    data = null;
  }
  if (!response.ok) {
    const message = (data && data.error) || `请求失败（${response.status}）`;
    throw new Error(message);
  }
  return data;
}

/* ---------- upload ---------- */

function setSelectedFile(file) {
  state.selectedFile = file;
  const dropTitle = $("#dropTitle");
  const dropHint = $("#dropHint");
  const dropzone = $("#dropzone");
  const button = $("#analyzeBtn");

  if (!file) {
    dropTitle.textContent = "把击球视频拖到这里";
    dropHint.textContent = "或点击选择 · MP4 / MOV · 上限 300MB";
    dropzone.classList.remove("has-file");
    button.disabled = true;
    return;
  }

  if (file.size > 300 * 1024 * 1024) {
    showToast("视频超过 300MB，请先压缩或裁剪到 10–30 秒。");
    setSelectedFile(null);
    return;
  }
  dropTitle.textContent = file.name;
  dropHint.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · 点击可更换`;
  dropzone.classList.add("has-file");
  button.disabled = false;
}

function bindUpload() {
  const input = $("#videoInput");
  const dropzone = $("#dropzone");

  input.addEventListener("change", () => {
    setSelectedFile(input.files && input.files[0] ? input.files[0] : null);
  });

  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) {
      input.files = event.dataTransfer.files;
      setSelectedFile(file);
    }
  });

  $("#uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.selectedFile) return;
    const profile = {
      name: $("#playerName").value.trim(),
      handedness: $("#handedness").value,
      level: $("#level").value,
      goal: $("#goal").value.trim(),
    };
    const form = new FormData();
    form.append("video", state.selectedFile);
    form.append("profile", JSON.stringify(profile));

    showView("progress");
    updateProgress(1, "正在上传视频");
    $("#progressHint").textContent = "视频只保存在本机。第一次运行会加载姿态模型。";

    try {
      const data = await fetchJSON("/api/analyze", { method: "POST", body: form });
      state.jobId = data.job_id;
      pollJob();
    } catch (error) {
      updateProgress(0, "上传失败");
      $("#progressHint").textContent = error.message;
      showToast(error.message);
    }
  });

  $("#backToUpload").addEventListener("click", () => {
    clearTimeout(state.pollTimer);
    showView("upload");
  });
}

function updateProgress(percent, stage) {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  $("#progressBar").style.width = pct(value);
  $("#progressPct").textContent = `${Math.round(value)}%`;
  $("#progressStage").textContent = stage || "分析中";
  $("#progressTrack").setAttribute("aria-valuenow", String(Math.round(value)));
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const data = await fetchJSON(`/api/jobs/${state.jobId}`);
    updateProgress(data.progress, data.stage);
    if (data.status === "done" && data.result) {
      state.result = data.result;
      state.llm = data.result.llm || state.llm;
      updateLlmChip(state.llm);
      state.activeStrokeType = firstStrokeType(data.result);
      state.chatHistory = [];
      renderReport(data.result);
      showView("report");
      return;
    }
    if (data.status === "error") {
      $("#progressStage").textContent = "分析失败";
      $("#progressHint").textContent = data.error || "请调整拍摄条件后重试。";
      $("#progressPct").textContent = "";
      $("#progressBar").style.width = "0%";
      showToast(data.error || "分析失败");
      return;
    }
    state.pollTimer = setTimeout(pollJob, 900);
  } catch (error) {
    $("#progressHint").textContent = error.message;
    showToast(error.message);
    if (error.message.includes("任务不存在")) {
      showView("upload");
      return;
    }
    state.pollTimer = setTimeout(pollJob, 2500);
  }
}

/* ---------- LLM settings ---------- */

function updateLlmChip(llm) {
  state.llm = llm || { provider: "local", ready: false };
  const chip = $("#llmChip");
  chip.classList.toggle("live", Boolean(llm && llm.ready));
  chip.textContent = providerLabel(llm);
  chip.title = llm && llm.reason ? llm.reason : "当前反馈引擎";
}

async function loadLlmStatus() {
  try {
    const llm = await fetchJSON("/api/llm");
    updateLlmChip(llm);
  } catch (error) {
    updateLlmChip({ provider: "local", ready: false });
  }
}

function bindSettings() {
  const modal = $("#settingsModal");
  const open = () => {
    $("#settingsStatus").textContent =
      state.llm.ready && state.llm.provider === "claude"
        ? `已连接 ${state.llm.model || "Claude"}。新分析会由 Claude 生成反馈，也可以在当前报告里继续追问。`
        : "当前使用本地规则教练。填入 API Key 后，重新分析即可获得 Claude 的多模态反馈。";
    $("#apiKeyInput").value = "";
    modal.classList.remove("hidden");
    setTimeout(() => $("#apiKeyInput").focus(), 30);
  };
  const close = () => modal.classList.add("hidden");

  $("#settingsBtn").addEventListener("click", open);
  $("#closeSettings").addEventListener("click", close);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.classList.contains("hidden")) close();
  });

  $("#saveKeyBtn").addEventListener("click", async () => {
    const key = $("#apiKeyInput").value.trim();
    if (!key) {
      showToast("请先粘贴 API Key。");
      return;
    }
    try {
      const llm = await fetchJSON("/api/llm/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      updateLlmChip(llm);
      close();
      showToast("已连接 Claude。重新分析后生效。");
    } catch (error) {
      showToast(error.message);
    }
  });

  $("#clearKeyBtn").addEventListener("click", async () => {
    try {
      const llm = await fetchJSON("/api/llm/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: "" }),
      });
      updateLlmChip(llm);
      showToast("已切回本地规则教练。");
    } catch (error) {
      showToast(error.message);
    }
  });
}

/* ---------- report rendering ---------- */

function firstStrokeType(result) {
  const types = Object.keys(result.by_type || {});
  if (!types.length) return null;
  types.sort((a, b) => (result.by_type[b].count || 0) - (result.by_type[a].count || 0));
  return types[0];
}

function dimensionBars(dimensions) {
  const order = [
    ["contact", "击球点"],
    ["rotation", "转体与准备"],
    ["drive", "下肢与发力"],
    ["finish", "随挥与平衡"],
  ];
  const rows = order
    .filter(([key]) => dimensions && dimensions[key] !== undefined && dimensions[key] !== null)
    .map(([key, label]) => {
      const value = Number(dimensions[key]) || 0;
      return `<div class="dimension-row">
        <span>${esc(label)}</span>
        <span class="dimension-track"><span class="dimension-fill" style="width:${pct(value)}"></span></span>
        <span>${Math.round(value)}</span>
      </div>`;
    });
  if (!rows.length) return "";
  return `<div class="dimension-bars">${rows.join("")}</div>`;
}

function metaItems(result) {
  const items = [];
  const date = new Date((result.created_at || Date.now() / 1000) * 1000);
  items.push(`<span class="meta-item">${date.toLocaleDateString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" })}</span>`);
  items.push(`<span class="meta-item"><strong>${esc(result.video?.filename || "视频")}</strong> · ${formatTime(result.video?.duration || 0)}</span>`);
  if (result.video?.truncated) {
    items.push(`<span class="meta-item">仅分析前 ${Math.round(result.video.analyzed_seconds || 0)} 秒</span>`);
  }
  items.push(`<span class="meta-item">${esc(result.dominant_hand_label || "")}${result.handedness === "auto" ? "" : "持拍"}</span>`);
  if (result.consistency !== null && result.consistency !== undefined) {
    items.push(`<span class="meta-item">动作稳定性 <strong>${Math.round(result.consistency)}</strong></span>`);
  }
  items.push(`<span class="meta-item">反馈引擎：<strong>${esc(providerLabel(result.llm))}</strong></span>`);
  return items.join("");
}

function renderReport(result) {
  const feedback = result.feedback || {};
  const level = result.level || {};
  const overall = level.score ?? result.overall_score;
  const score = overall === null || overall === undefined ? "—" : Math.round(overall);
  const dimensions = result.dimensions || {};
  const noStrokes = result.no_strokes || result.stroke_count === 0;
  const issues = feedback.issues || [];
  const strengths = feedback.strengths || [];

  const head = `
    <header class="report-head">
      <div class="report-head-copy">
        <p class="eyebrow">分析报告</p>
        <h1 class="report-title">${esc(feedback.headline || "本次击球分析")}</h1>
        <p class="report-overall">${esc(feedback.overall || feedback.level_assessment || "")}</p>
        <div class="meta-row">${metaItems(result)}</div>
      </div>
      <aside class="match-card">
        <div class="score-line">
          <span class="score-value">${score}</span>
          <span class="score-denom">/100</span>
          <span class="score-caption">综合评分</span>
        </div>
        <div class="level-line">
          <span class="level-ntrp">${level.ntrp ? `NTRP ${esc(level.ntrp)}` : "样本不足"}</span>
          <span class="level-label">${esc(level.label || "暂不评估")}</span>
        </div>
        ${dimensionBars(dimensions)}
      </aside>
    </header>
  `;

  const notice = noStrokes
    ? `<section class="notice">
        <strong>没有识别到明确挥拍</strong>
        <p>${esc(feedback.overall || "")}</p>
        <ul>${(result.camera_notes || []).map((note) => `<li>${esc(note)}</li>`).join("")}</ul>
      </section>`
    : "";

  const videoUrl = result.annotated_video && result.annotated_video.url;
  const videoBlock = videoUrl
    ? `<div class="video-shell"><video id="reportVideo" controls preload="metadata" playsinline src="${esc(videoUrl)}"></video>
        <div class="video-actions"><a href="${esc(videoUrl)}" download>下载标注视频</a><span>${result.annotated_video.web_compatible === false ? "当前为兼容模式，如无法播放请下载。" : ""}</span></div>
      </div>`
    : `<div class="video-empty">标注视频没有生成，仍可查看下面的指标与反馈。</div>`;

  const timeline = !noStrokes
    ? `<div class="shot-timeline" id="shotTimeline">
        <div class="timeline-line"></div>
        ${(result.strokes || []).map((stroke) => {
          const left = result.video && result.video.analyzed_seconds
            ? (stroke.contact_time / result.video.analyzed_seconds) * 100
            : 0;
          return `<button class="timeline-marker" type="button" data-time="${stroke.contact_time}" style="left:${Math.max(1, Math.min(99, left))}%" aria-label="跳到 ${formatTime(stroke.contact_time)} ${esc(stroke.type_label)}">
            <span class="marker-label">${esc(stroke.type_label)} · ${formatTime(stroke.contact_time)}</span>
          </button>`;
        }).join("")}
        <div class="timeline-caption"><span>0:00</span><span>${formatTime(result.video?.analyzed_seconds || 0)}</span></div>
      </div>`
    : "";

  const shotIndex = !noStrokes
    ? `<aside class="shot-index">
        <div class="shot-index-head">
          <h2>击球索引</h2>
          <p>${result.stroke_count} 次 · ${esc(result.dominant_hand_label || "")}</p>
        </div>
        <ul class="shot-list">
          ${(result.strokes || []).map((stroke) => `
            <li>
              <button class="shot-item" type="button" data-time="${stroke.contact_time}">
                <span class="shot-time">${formatTime(stroke.contact_time)}</span>
                <span class="shot-type">${esc(stroke.type_label)}<small>腕速 ${formatNumber(stroke.peak_speed_mps, 1)} m/s</small></span>
                <span class="shot-score">${stroke.scoring?.score ?? "—"}</span>
              </button>
            </li>`).join("")}
        </ul>
      </aside>`
    : `<aside class="shot-index"><div class="shot-index-head"><h2>击球索引</h2></div><p class="chat-empty">暂无可列出的击球。</p></aside>`;
  const issuesHtml = issues.length
    ? issues.map((issue, index) => {
      const priority = String(issue.priority || index + 1).padStart(2, "0");
      const severity = issue.severity || "medium";
      return `<article class="issue">
        <div class="issue-priority">${priority}</div>
        <div>
          <div class="issue-head">
            <h3>${esc(issue.title)}</h3>
            <span class="issue-stroke">${esc(issue.stroke || "")}</span>
            <span class="severity ${esc(severity)}">${severityLabel(severity)}</span>
          </div>
          <p class="issue-observation">${esc(issue.observation || "")}</p>
          <div class="issue-grid">
            <div class="issue-cell"><span>为什么重要</span><p>${esc(issue.why || "")}</p></div>
            <div class="issue-cell"><span>怎么改</span><p>${esc(issue.fix || "")}</p></div>
            <div class="issue-cell"><span>练什么</span><p>${esc(issue.drill || "")}</p></div>
          </div>
        </div>
      </article>`;
    }).join("")
    : `<p class="chat-empty">这次没有触发高优先级问题。保持当前动作框架，重点放在连续性和比赛中的稳定性。</p>`;

  const strengthsHtml = strengths.length
    ? `<div class="strength-list">${strengths.map((item) => `<p><strong>${esc(item.title)}</strong> ${esc(item.detail)}</p>`).join("")}</div>`
    : "";

  const diagnosis = noStrokes ? "" : `
    <section class="section-block">
      <div class="section-head">
        <h2>诊断</h2>
        <p>按改进收益排序。每个问题都对应一个可以单独练的动作和练习。</p>
      </div>
      <div class="issue-list">${issuesHtml}</div>
      ${strengthsHtml}
    </section>
  `;

  const metrics = noStrokes ? "" : `
    <section class="section-block">
      <div class="section-head">
        <h2>身体数据</h2>
        <p>黄色区间是业余球员的参考范围，圆点是你这次的平均值。先看偏离最大的两项。</p>
      </div>
      <div class="stroke-tabs" id="strokeTabs">${renderStrokeTabs(result)}</div>
      <div id="metricsBody">${renderMetrics(result, state.activeStrokeType)}</div>
    </section>
  `;

  const keyframes = (result.keyframes || []).length
    ? `<section class="section-block">
        <div class="section-head">
          <h2>关键帧</h2>
          <p>打开对照骨骼线看触球位置，比只看视频更容易发现问题。</p>
        </div>
        <div class="keyframe-row">
          ${result.keyframes.map((keyframe) => `
            <a class="keyframe" href="${esc(keyframe.url)}" target="_blank" rel="noopener">
              <img src="${esc(keyframe.url)}" alt="${esc(keyframe.label)}" loading="lazy">
              <span>${esc(keyframe.label)} · ${formatTime(keyframe.time)}</span>
            </a>`).join("")}
        </div>
      </section>`
    : "";

  const plan = renderPlan(feedback.training_plan || {});
  const mainGrid = noStrokes
    ? ""
    : `<div class="report-grid"><div class="video-column">${videoBlock}${timeline}</div>${shotIndex}</div>`;

  const chat = `
    <section class="section-block chat-section">
      <div class="section-head">
        <h2>问教练</h2>
        <p>可以追问某个指标、某次击球，或让教练把训练计划改得更适合你的时间。</p>
      </div>
      <div class="chat-log" id="chatLog"></div>
      <form class="chat-form" id="chatForm">
        <input id="chatInput" type="text" maxlength="2000" placeholder="例如：为什么我的击球点总是偏后？" ${result.llm?.provider === "claude" ? "" : "disabled"}>
        <button class="primary-button" type="submit" ${result.llm?.provider === "claude" ? "" : "disabled"}>发送</button>
      </form>
      <p class="chat-note">${result.llm?.provider === "claude" ? "回答会引用本次分析数据。" : "配置 Claude API Key 后，可以在这里继续追问。"}</p>
    </section>
  `;

  const disclaimer = `
    <p class="report-disclaimer">这是基于单机位人体姿态估计的训练建议：没有追踪球和球拍，无法测量球速、旋转和落点；触球点是腕速与手臂伸展的近似估计。报告不能替代现场教练或医学诊断。${result.llm?.reason ? ` ${esc(result.llm.reason)}` : ""}</p>
  `;

  $("#reportContent").innerHTML = `
    <div class="report">
      ${head}
      ${notice}
      ${mainGrid}
      ${diagnosis}
      ${metrics}
      ${keyframes}
      ${plan}
      ${chat}
      ${disclaimer}
    </div>
  `;

  bindReport();
  renderChat();
}

function renderStrokeTabs(result) {
  const types = Object.keys(result.by_type || {});
  types.sort((a, b) => (result.by_type[b].count || 0) - (result.by_type[a].count || 0));
  return types.map((type) => {
    const data = result.by_type[type];
    return `<button class="stroke-tab ${type === state.activeStrokeType ? "active" : ""}" type="button" data-type="${esc(type)}">
      ${esc(data.label || strokeTypeLabel(type))}<small>${data.count} 次</small>
    </button>`;
  }).join("");
}

function strokeTypeLabel(type) {
  return { forehand: "正手", backhand: "反手", serve: "发球", overhead: "高压" }[type] || type;
}

function renderMetrics(result, type) {
  const data = type ? result.by_type[type] : null;
  if (!data) return `<p class="chat-empty">暂无可展示的指标。</p>`;
  const rows = (data.metrics || []).map((metric) => {
    const status = metric.status || "unknown";
    const dotClass = status === "good" ? "good" : status === "ok" ? "ok" : status === "poor" ? "poor" : "";
    return `<div class="metric-row">
      <div class="metric-label">
        <span>${esc(metric.label)}</span>
        <small>${esc(metric.unit)} · 参考 ${formatNumber(metric.ideal_low, 1)}–${formatNumber(metric.ideal_high, 1)}</small>
      </div>
      ${renderGauge(metric, dotClass)}
      <span class="metric-value">${formatNumber(metric.value, metric.decimals ?? 1)}</span>
      <span class="metric-status ${status}" title="${esc(statusLabelLong(status))}">${esc(statusLabel(status))} · ${metric.score ?? "—"}</span>
    </div>`;
  }).join("");
  return `
    <div class="dimension-bars metric-dimensions">${dimensionBars(data.dimensions)}</div>
    <div class="metric-table">${rows}</div>
  `;
}

function renderGauge(metric, dotClass) {
  const value = Number(metric.value);
  const low = Number(metric.ideal_low);
  const high = Number(metric.ideal_high);
  const idealWidth = Math.max(high - low, Math.abs(value) * 0.1, 0.2);
  let min = Math.min(low, value) - idealWidth * 0.6;
  let max = Math.max(high, value) + idealWidth * 0.6;
  if (low >= 0 && value >= 0) min = Math.max(0, min);
  if (max <= min) max = min + 1;
  const position = (input) => Math.max(0, Math.min(100, ((input - min) / (max - min)) * 100));
  const start = position(low);
  const end = position(high);
  return `<span class="metric-gauge" aria-label="实测 ${formatNumber(value, metric.decimals ?? 1)} ${esc(metric.unit)}，参考 ${formatNumber(low, 1)} 到 ${formatNumber(high, 1)}">
    <span class="metric-track"></span>
    <span class="metric-ideal" style="left:${start}%;width:${Math.max(1, end - start)}%"></span>
    <span class="metric-dot ${dotClass}" style="left:${position(value)}%"></span>
  </span>`;
}

function renderPlan(plan) {
  const weeks = plan.weeks || [];
  if (!weeks.length) {
    return `<section class="section-block">
      <div class="section-head"><h2>训练计划</h2></div>
      <p class="plan-focus"><strong>${esc(plan.focus || "先拍一段能被稳定分析的视频")}</strong></p>
      ${renderSuccess(plan.success_metrics)}
    </section>`;
  }
  const weekHtml = weeks.map((week) => `
    <div class="week">
      <div class="week-head">
        <h3>${esc(week.theme)}</h3>
        <span>WEEK ${esc(week.week)}</span>
      </div>
      ${(week.sessions || []).map((session) => `
        <div class="session">
          <div class="session-head">
            <span class="session-day">${esc(session.day)}</span>
            <span class="session-title">${esc(session.title)}</span>
            <span class="session-duration">${esc(session.duration_min)} 分钟</span>
          </div>
          ${(session.drills || []).map((drill) => `
            <div class="drill">
              <span class="drill-name">${esc(drill.name)}</span>
              <span class="drill-sets">${esc(drill.sets)}</span>
              <span class="drill-notes">${esc(drill.notes)}</span>
            </div>`).join("")}
        </div>`).join("")}
    </div>`).join("");

  return `<section class="section-block">
    <div class="section-head">
      <h2>两周训练计划</h2>
      <p>每周 3 次，每次 40–60 分钟。一次只盯一个重点，练完用视频对照。</p>
    </div>
    <p class="plan-focus"><strong>本周重点：</strong>${esc(plan.focus || "")}</p>
    <div class="weeks">${weekHtml}</div>
    ${renderSuccess(plan.success_metrics)}
  </section>`;
}

function renderSuccess(items) {
  if (!items || !items.length) return "";
  return `<ul class="success-list">${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>`;
}

function bindReport() {
  const video = $("#reportVideo");
  if (video) {
    video.addEventListener("error", () => {
      const shell = video.closest(".video-shell");
      if (!shell) return;
      const url = state.result?.annotated_video?.url;
      shell.innerHTML = `<div class="video-empty">当前浏览器无法直接播放标注视频。${url ? `请<a href="${esc(url)}" download>下载 MP4</a> 后观看，或换用 Chrome / Edge。` : ""}</div>`;
    });
  }
  const seek = (time, trigger) => {
    if (trigger) {
      $$(".timeline-marker, .shot-item").forEach((item) => item.classList.toggle("active", item === trigger));
    }
    if (!video) return;
    video.currentTime = Math.max(0, Number(time) - 0.6);
    video.play().catch(() => {});
    video.focus({ preventScroll: true });
  };

  $$(".timeline-marker, .shot-item").forEach((button) => {
    button.addEventListener("click", () => seek(button.dataset.time, button));
  });

  $$(".stroke-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.activeStrokeType = tab.dataset.type;
      $$(".stroke-tab").forEach((item) => item.classList.toggle("active", item === tab));
      const body = $("#metricsBody");
      if (body) body.innerHTML = renderMetrics(state.result, state.activeStrokeType);
    });
  });

  const chatForm = $("#chatForm");
  if (chatForm) {
    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (state.sendingChat) return;
      const input = $("#chatInput");
      const question = input.value.trim();
      if (!question) return;
      state.chatHistory.push({ role: "user", content: question });
      input.value = "";
      renderChat();
      state.sendingChat = true;
      try {
        const payload = {
          job_id: state.result.job_id,
          messages: state.chatHistory,
        };
        const data = await fetchJSON("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        state.chatHistory.push({ role: "assistant", content: data.reply || "我没有生成有效回答。" });
      } catch (error) {
        state.chatHistory.push({ role: "assistant", content: `暂时无法回答：${error.message}` });
      } finally {
        state.sendingChat = false;
        renderChat();
      }
    });
  }

  updateChatAvailability();
}

function updateChatAvailability() {
  const provider = state.result?.llm?.provider;
  const input = $("#chatInput");
  const button = $("#chatForm button");
  if (!input || !button) return;
  const enabled = provider === "claude";
  input.disabled = !enabled;
  button.disabled = !enabled;
  if (enabled) input.placeholder = "例如：为什么我的击球点总是偏后？";
}

function renderChat() {
  const log = $("#chatLog");
  if (!log) return;
  if (!state.chatHistory.length) {
    log.innerHTML = `<p class="chat-empty">${state.result?.llm?.provider === "claude" ? "问一个具体问题，教练会结合这次击球数据回答。" : "本地规则模式不支持自由问答。配置 Claude 后可以继续追问。"}</p>`;
    return;
  }
  log.innerHTML = state.chatHistory.map((message) => `
    <div class="chat-message ${message.role === "user" ? "user" : ""}">${esc(message.content)}</div>
  `).join("");
  if (state.sendingChat) {
    log.innerHTML += `<div class="chat-message">教练正在看数据…</div>`;
  }
  log.scrollTop = log.scrollHeight;
}

/* ---------- boot ---------- */

function bindNavigation() {
  $("#brandLink").addEventListener("click", (event) => {
    event.preventDefault();
    clearTimeout(state.pollTimer);
    showView("upload");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindNavigation();
  bindUpload();
  bindSettings();
  loadLlmStatus();
  const job = new URLSearchParams(window.location.search).get("job");
  if (job) {
    state.jobId = job;
    showView("progress");
    updateProgress(0, "正在加载分析结果");
    pollJob();
  }
});
