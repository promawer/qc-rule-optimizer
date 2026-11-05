(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const TERMINAL = new Set(["reached", "exhausted", "failed", "cancelled", "evaluated"]);

  const state = {
    rules: [],
    currentId: null,
    dataset: null,
    settings: null,
    dirty: false,
    job: null,
    jobId: null,
    timer: null,
    caseRows: [],
    caseFilter: "all",
    caseSplit: "TRAIN",
    caseWithReason: true,
    prompt: null,
    activeTab: "rule",
  };

  // ---------------------------------------------------------------- 工具
  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function pct(value) {
    return value == null || Number.isNaN(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
  }

  function delta(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    const points = Number(value) * 100;
    return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pt`;
  }

  function fmtTime(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString("zh-CN", { hour12: false });
  }

  function timeOnly(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString("zh-CN", { hour12: false });
  }

  function elapsedText(job) {
    const start = job.startedAt || job.createdAt;
    if (!start) return "";
    const end = job.finishedAt || new Date().toISOString();
    const seconds = Math.max(0, (new Date(end) - new Date(start)) / 1000);
    if (seconds < 60) return `${seconds.toFixed(0)} 秒`;
    return `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`;
  }

  const STATUS = {
    queued: ["排队中", ""],
    running: ["运行中", "badge-mock"],
    reached: ["已达标", "badge-ok"],
    exhausted: ["未达标", "badge-fail"],
    cancelled: ["已取消", ""],
    failed: ["失败", "badge-fail"],
    evaluated: ["已评测", "badge-ok"],
  };

  function statusInfo(status) {
    const item = STATUS[status] || [status || "未知", ""];
    return { text: item[0], cls: item[1] };
  }

  function statusBadge(status) {
    const info = statusInfo(status);
    return `<span class="badge ${info.cls}">${esc(info.text)}</span>`;
  }

  function labelPill(value) {
    if (value === "VIOLATION") return '<span class="pill pill-v">违规</span>';
    if (value === "NORMAL") return '<span class="pill pill-n">正常</span>';
    return '<span class="pill pill-u">未知</span>';
  }

  function predPill(value) {
    if (value === true) return '<span class="pill pill-v">违规</span>';
    if (value === false) return '<span class="pill pill-n">正常</span>';
    return '<span class="pill pill-u">未判定</span>';
  }

  function stat(label, value, cls) {
    return `<div class="stat"><div class="label">${esc(label)}</div><div class="value ${cls || ""}">${esc(value)}</div></div>`;
  }

  function transcriptBlock(text) {
    const safe = esc(text || "");
    const short = safe.length > 110 ? `${safe.slice(0, 110)}…` : safe;
    return `<span>${short}</span><div class="transcript-full" hidden>${safe}</div>`;
  }

  let toastTimer = null;
  function toast(message, type = "") {
    const el = $("#toast");
    el.textContent = message;
    el.className = `toast ${type}`;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch { data = text; }
    }
    if (!response.ok) {
      let detail = data && data.detail ? data.detail : `请求失败（HTTP ${response.status}）`;
      if (Array.isArray(detail)) detail = detail.map((item) => item.msg || JSON.stringify(item)).join("；");
      throw new Error(String(detail));
    }
    return data;
  }

  async function withBusy(button, label, action) {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = label;
    try {
      return await action();
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  function markDirty() { state.dirty = true; }

  // ---------------------------------------------------------------- 启动
  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    boot().catch((error) => toast(error.message, "error"));
  });

  async function boot() {
    await loadSettings();
    await loadRules();
    await loadPrompt();
    if (state.rules.length) {
      await selectRule(state.rules[0].id);
    } else {
      clearRuleForm();
    }
    const initial = location.hash.replace("#", "");
    if (["rule", "data", "optimize", "prompt"].includes(initial)) setTab(initial);
  }

  function bindEvents() {
    $("#settingsBtn").addEventListener("click", openSettings);
    $("#closeSettingsBtn").addEventListener("click", () => $("#settingsDialog").close());
    $("#saveSettingsBtn").addEventListener("click", saveSettings);
    $("#testConnBtn").addEventListener("click", testConnection);

    $("#newRuleBtn").addEventListener("click", () => {
      if (state.dirty && !confirm("当前修改尚未保存，仍要新建规则吗？")) return;
      clearRuleForm();
      setTab("rule");
      $("#ruleName").focus();
    });
    $("#ruleSearch").addEventListener("input", renderRuleList);

    window.addEventListener("hashchange", () => {
      const tab = location.hash.replace("#", "");
      if (["rule", "data", "optimize"].includes(tab) && tab !== state.activeTab) setTab(tab);
    });

    $("#tabs").addEventListener("click", (event) => {
      const tab = event.target.closest(".tab");
      if (tab) setTab(tab.dataset.tab);
    });

    $("#saveRuleBtn").addEventListener("click", saveRule);
    $("#testRegexBtn").addEventListener("click", testRegex);
    $("#deleteRuleBtn").addEventListener("click", deleteRule);

    ["#ruleCode", "#ruleName", "#ruleDesc", "#ruleRegex", "#ruleSemantic", "#ruleLogic"].forEach((selector) => {
      $(selector).addEventListener("input", markDirty);
      $(selector).addEventListener("change", markDirty);
    });

    $("#uploadBtn").addEventListener("click", uploadDataset);
    $("#exportLink").addEventListener("click", (event) => event.stopPropagation());

    $("#startBtn").addEventListener("click", () => startJob("optimize"));
    $("#evaluateBtn").addEventListener("click", () => startJob("evaluate"));
    $("#cancelBtn").addEventListener("click", cancelJob);

    $("#ruleList").addEventListener("click", (event) => {
      const item = event.target.closest(".rule-item");
      if (!item || item.dataset.rule === state.currentId) return;
      if (state.dirty && !confirm("当前修改尚未保存，仍要切换规则吗？")) return;
      selectRule(item.dataset.rule).catch((error) => toast(error.message, "error"));
    });

    $("#jobResult").addEventListener("click", (event) => {
      const button = event.target.closest("[data-apply]");
      if (button) applyRun(button.dataset.apply, button);
    });
    $("#jobResult").addEventListener("change", (event) => {
      if (event.target.id === "applyForce") {
        const button = $("#applyBtn");
        if (button) button.disabled = !event.target.checked;
      }
    });
    $("#historyList").addEventListener("click", (event) => {
      const button = event.target.closest("[data-apply]");
      if (button) applyRun(button.dataset.apply, button);
    });

    $("#tab-optimize").addEventListener("click", (event) => {
      const filter = event.target.closest("[data-filter]");
      if (filter) {
        state.caseFilter = filter.dataset.filter;
        $$("#tab-optimize .filter").forEach((item) => item.classList.toggle("active", item === filter));
        renderCases();
        return;
      }
      const splitButton = event.target.closest("[data-case-split]");
      if (splitButton) {
        state.caseSplit = splitButton.dataset.caseSplit;
        $$("#caseSplitTabs .split").forEach((item) => item.classList.toggle("active", item === splitButton));
        if (state.jobId) loadJobCases(state.jobId);
      }
    });

    $("#savePromptBtn").addEventListener("click", savePrompt);

    ["#datasetTable", "#casesTable"].forEach((selector) => {
      $(selector).addEventListener("click", (event) => {
        const row = event.target.closest("tr.clickable");
        if (!row) return;
        const full = row.querySelector(".transcript-full");
        if (full) full.hidden = !full.hidden;
      });
    });

    $("#settingsDialog").addEventListener("click", (event) => {
      if (event.target === $("#settingsDialog")) $("#settingsDialog").close();
    });
  }

  // ---------------------------------------------------------------- 规则
  async function loadRules() {
    state.rules = await api("/api/rules");
    renderRuleList();
    if (state.currentId) {
      const rule = state.rules.find((item) => item.id === state.currentId);
      if (rule) renderRuleHeader(rule);
      else clearRuleForm();
    }
  }

  function renderRuleList() {
    const keyword = ($("#ruleSearch").value || "").trim().toLowerCase();
    const rules = state.rules.filter((rule) => !keyword || rule.name.toLowerCase().includes(keyword));
    const list = $("#ruleList");
    if (!rules.length) {
      list.innerHTML = '<div class="empty">还没有匹配的规则<br />点击右上角「新建」创建一条</div>';
      return;
    }
    list.innerHTML = rules.map((rule) => {
      const dataset = rule.dataset || {};
      const accuracy = dataset.autoAccuracy == null ? "" : ` · 自动 ${pct(dataset.autoAccuracy)}`;
      return `<div class="rule-item ${rule.id === state.currentId ? "active" : ""}" data-rule="${rule.id}">
        <div class="name"><span>${esc(rule.name)}</span><span class="logic">${esc(rule.logic)}</span></div>
        <div class="sub">${esc(rule.code)} · ${dataset.total || 0} 通通话${accuracy}</div>
      </div>`;
    }).join("");
  }

  function nextRuleCode() {
    let max = 0;
    state.rules.forEach((rule) => {
      const matched = /^R(\d+)$/i.exec(rule.code || "");
      if (matched) max = Math.max(max, parseInt(matched[1], 10));
    });
    return `R${String(max + 1).padStart(3, "0")}`;
  }

  function renderRuleHeader(rule) {
    $("#ruleTitle").textContent = rule.name;
    const parts = [rule.code, `v${rule.version}`, `${rule.logic === "AND" ? "AND 与" : "OR 或"}`];
    if (rule.description) parts.push(rule.description);
    $("#ruleMeta").textContent = parts.join(" · ");
    $("#deleteRuleBtn").hidden = false;
  }

  async function selectRule(id) {
    const rule = state.rules.find((item) => item.id === id);
    if (!rule) return;
    clearTimeout(state.timer);
    state.currentId = id;
    state.dirty = false;
    state.job = null;
    state.jobId = null;
    state.dataset = null;
    fillRuleForm(rule);
    renderRuleHeader(rule);
    renderJob(null);
    $("#casesCard").hidden = true;
    await loadDataset();
    await loadRuns();
  }

  function renderChangeLog(rule) {
    const card = $("#changeLogCard");
    const list = $("#changeLogList");
    const log = (rule.changeLog || []).slice().reverse();
    if (!log.length) {
      card.hidden = true;
      list.innerHTML = "";
      return;
    }
    card.hidden = false;
    const sourceLabel = { manual: "人工修改", optimize: "优化应用" };
    const fieldLabel = {
      code: "编号", name: "名称", description: "说明",
      regex: "正则", semantic: "语义", logic: "逻辑",
    };
    list.innerHTML = log.map((entry) => {
      const fields = (entry.changedFields || []).map((key) => fieldLabel[key] || key).join("、");
      return `<div class="history-item">
        <span class="muted">v${entry.version} · ${fmtTime(entry.at)}</span>
        <span class="grow"><b>${esc(sourceLabel[entry.source] || entry.source)}</b>：${esc(entry.reason)}</span>
        <span class="chip">${esc(fields || "无字段变化")}</span>
      </div>`;
    }).join("");
  }

  function fillRuleForm(rule) {
    $("#ruleCode").value = rule.code || "";
    $("#ruleName").value = rule.name || "";
    $("#ruleDesc").value = rule.description || "";
    $("#ruleRegex").value = rule.regex || "";
    $("#ruleSemantic").value = rule.semantic || "";
    $("#ruleLogic").value = rule.logic || "OR";
    $("#ruleChangeReason").value = "";
    $("#ruleVersion").textContent = `版本 v${rule.version} · 更新于 ${fmtTime(rule.updatedAt)}`;
    $("#deleteRuleBtn").hidden = false;
    renderChangeLog(rule);
  }

  function clearRuleForm() {
    clearTimeout(state.timer);
    state.currentId = null;
    state.dirty = false;
    state.dataset = null;
    state.job = null;
    state.jobId = null;
    $("#ruleTitle").textContent = "新建规则";
    $("#ruleMeta").textContent = "填写规则内容后点击保存";
    $("#deleteRuleBtn").hidden = true;
    $("#ruleCode").value = nextRuleCode();
    $("#ruleName").value = "";
    $("#ruleDesc").value = "";
    $("#ruleRegex").value = "";
    $("#ruleSemantic").value = "";
    $("#ruleLogic").value = "OR";
    $("#ruleChangeReason").value = "";
    $("#ruleVersion").textContent = "";
    $("#changeLogCard").hidden = true;
    $("#changeLogList").innerHTML = "";
    $("#datasetStats").innerHTML = "";
    $("#datasetCount").textContent = "";
    $("#datasetTable").innerHTML = '<div class="empty">保存规则并导入数据后显示</div>';
    $("#exportLink").hidden = true;
    $("#jobResult").innerHTML = "";
    $("#progressCard").hidden = true;
    $("#casesCard").hidden = true;
    $("#historyCard").hidden = true;
    setTab("rule");
  }

  async function saveRule() {
    const payload = {
      code: $("#ruleCode").value.trim().toUpperCase(),
      name: $("#ruleName").value.trim(),
      description: $("#ruleDesc").value.trim(),
      regex: $("#ruleRegex").value.trim(),
      semantic: $("#ruleSemantic").value.trim(),
      logic: $("#ruleLogic").value,
      changeReason: $("#ruleChangeReason").value.trim(),
    };
    if (!payload.code) return toast("请填写规则编号，例如 R003", "error");
    if (!payload.name) return toast("请填写规则名称", "error");
    if (!payload.regex && !payload.semantic) return toast("正则表达式和语义描述至少填写一项", "error");
    if (state.currentId && !payload.changeReason) return toast("修改规则内容时请填写本次修改理由", "error");

    const button = $("#saveRuleBtn");
    try {
      await withBusy(button, "保存中…", async () => {
        const saved = state.currentId
          ? await api(`/api/rules/${state.currentId}`, {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            })
          : await api("/api/rules", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
        state.dirty = false;
        await loadRules();
        await selectRule(saved.id);
        toast("规则已保存", "success");
      });
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function deleteRule() {
    if (!state.currentId) return;
    const rule = state.rules.find((item) => item.id === state.currentId);
    if (!confirm(`确定删除规则「${rule ? rule.name : ""}」及其导入的数据？`)) return;
    try {
      await api(`/api/rules/${state.currentId}`, { method: "DELETE" });
      state.currentId = null;
      state.dirty = false;
      await loadRules();
      if (state.rules.length) await selectRule(state.rules[0].id);
      else clearRuleForm();
      toast("规则已删除", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function testRegex() {
    if (!state.currentId) return toast("请先保存规则", "error");
    const result = $("#testResult");
    result.textContent = "测试中…";
    try {
      const data = await api(`/api/rules/${state.currentId}/test-regex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: $("#testTranscript").value }),
      });
      result.textContent = data.error
        ? data.error
        : data.regexHit
          ? "命中：按当前逻辑会判为违规"
          : "未命中：正则部分不会判违规";
    } catch (error) {
      result.textContent = error.message;
    }
  }

  // ---------------------------------------------------------------- 数据
  async function loadDataset() {
    if (!state.currentId) return;
    const data = await api(`/api/rules/${state.currentId}/dataset?limit=200`);
    state.dataset = data;
    renderDataset(data);
    updateOptimizeHint();
  }

  function renderDataset(data) {
    if (!data) {
      $("#datasetStats").innerHTML = "";
      $("#datasetTable").innerHTML = '<div class="empty">还没有数据</div>';
      $("#exportLink").hidden = true;
      return;
    }
    const labels = data.labels || {};
    const calls = data.callTotal == null ? data.total : data.callTotal;
    const splitSummary = data.splitSummary || {};
    const splitText = ["TRAIN", "VALIDATION", "TEST"]
      .map((key) => {
        const item = splitSummary[key] || {};
        return `${key === "TRAIN" ? "训练" : key === "VALIDATION" ? "验证" : "测试"} ${item.total || 0}`;
      }).join(" / ");
    $("#datasetStats").innerHTML = [
      stat("通话总数", calls),
      stat(`本规则违规（${data.ruleCode || ""}）`, labels.VIOLATION || 0, "red"),
      stat("本规则正常", labels.NORMAL || 0, "green"),
      stat("自动质检准确率", data.autoAccuracy == null ? "—" : pct(data.autoAccuracy), "blue"),
    ].join("");
    $("#datasetCount").textContent = `共 ${calls} 通通话；${splitText}；当前规则派生 ${data.total} 个样本${data.rows.length < data.total ? `（显示前 ${data.rows.length} 条）` : ""}`;
    $("#exportLink").hidden = calls === 0;
    $("#exportLink").href = "/api/dataset/export.csv";

    if (!data.total) {
      $("#datasetTable").innerHTML = '<div class="empty">还没有通话数据，先导入 CSV / JSON / XLSX。</div>';
      return;
    }
    const codeText = (codes) => {
      const list = codes || [];
      return list.length ? list.map((code) => esc(code)).join("、") : '<span class="muted">—</span>';
    };
    const splitLabel = { TRAIN: "训练", VALIDATION: "验证", TEST: "测试" };
    $("#datasetTable").innerHTML = `<table>
      <thead><tr><th>通话ID</th><th>集合</th><th>本规则人工</th><th>本规则自动</th><th>自动是否正确</th><th>人工违规规则</th><th>自动违规规则</th><th>通话文本</th></tr></thead>
      <tbody>${data.rows.map((row) => {
        const autoCorrect = row.autoResult === row.humanResult;
        return `<tr class="clickable">
          <td class="mono">${esc(row.callId)}</td>
          <td><span class="chip">${splitLabel[row.split] || "训练"}</span></td>
          <td>${labelPill(row.humanResult)}</td>
          <td>${labelPill(row.autoResult)}</td>
          <td>${autoCorrect ? '<span class="pill pill-ok">正确</span>' : '<span class="pill pill-bad">错误</span>'}</td>
          <td class="mono">${codeText(row.humanRules)}</td>
          <td class="mono">${codeText(row.autoRules)}</td>
          <td class="transcript">${transcriptBlock(row.transcript)}</td>
        </tr>`;
      }).join("")}</tbody></table>`;
  }

  async function uploadDataset() {
    if (!state.currentId) return toast("请先保存规则", "error");
    const file = $("#fileInput").files[0];
    if (!file) return toast("请选择要导入的文件", "error");
    const form = new FormData();
    form.append("file", file);
    form.append("mode", $("#importMode").value);
    const button = $("#uploadBtn");
    try {
      await withBusy(button, "导入中…", async () => {
        const data = await api("/api/dataset/import", { method: "POST", body: form });
        const report = data.report || {};
        const positives = report.rulePositives || {};
        const positiveText = Object.keys(positives).length
          ? "；各规则人工违规数：" + Object.entries(positives).map(([code, count]) => `${code}=${count}`).join("，")
          : "";
        $("#importReport").hidden = false;
        $("#importReport").className = "report";
        $("#importReport").innerHTML = `<b>导入完成</b>：新增 ${report.imported || 0} 通通话，跳过 ${report.skipped || 0} 通；
          其中人工判定存在违规的 ${report.callsWithViolation || 0} 通，当前共 ${data.callTotal || 0} 通。${esc(positiveText)}
          ${(report.warnings || []).length ? `<div class="muted">提示：${esc((report.warnings || []).join("；"))}</div>` : ""}`;
        await loadDataset();
        await loadRules();
        toast("通话数据已导入", "success");
      });
    } catch (error) {
      $("#importReport").hidden = false;
      $("#importReport").className = "report error";
      $("#importReport").textContent = error.message;
      toast(error.message, "error");
    }
  }

  // ---------------------------------------------------------------- 通用提示词
  async function loadPrompt() {
    const data = await api("/api/semantic-prompt");
    state.prompt = data;
    $("#semanticPrompt").value = data.prompt || "";
    $("#promptVersion").textContent = `当前 v${data.version}`;
    $("#promptPreview").textContent = data.preview || "";
    const history = (data.history || []).slice().reverse();
    $("#promptHistoryCard").hidden = !history.length;
    $("#promptHistoryList").innerHTML = history.map((entry) => `
      <div class="history-item">
        <span class="muted">v${entry.version} · ${fmtTime(entry.at)}</span>
        <span class="grow"><b>${entry.source === "optimize" ? "优化应用" : "人工修改"}</b>：${esc(entry.reason)}</span>
      </div>`).join("");
    $("#promptStatus").textContent = "";
  }

  async function savePrompt() {
    const prompt = $("#semanticPrompt").value;
    const reason = $("#promptChangeReason").value.trim();
    if (!prompt.trim()) return toast("通用语义提示词不能为空", "error");
    if (!reason) return toast("修改通用语义提示词时请填写理由", "error");
    try {
      await withBusy($("#savePromptBtn"), "保存中…", async () => {
        await api("/api/semantic-prompt", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, reason }),
        });
        $("#promptChangeReason").value = "";
        await loadPrompt();
        toast("通用语义提示词已保存", "success");
      });
    } catch (error) {
      $("#promptStatus").textContent = error.message;
      toast(error.message, "error");
    }
  }

  // ---------------------------------------------------------------- 页签
  function setTab(name) {
    state.activeTab = name;
    $$("#tabs .tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
    ["rule", "data", "optimize", "prompt"].forEach((key) => {
      $(`#tab-${key}`).classList.toggle("hidden", key !== name);
    });
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    if (name === "data" && state.currentId) loadDataset().catch((error) => toast(error.message, "error"));
    if (name === "optimize") {
      updateOptimizeHint();
      loadRuns().catch(() => {});
    }
    if (name === "prompt") loadPrompt().catch((error) => toast(error.message, "error"));
  }

  function updateOptimizeHint() {
    const hint = $("#optimizeHint");
    if (!state.currentId) {
      hint.textContent = "先保存一条规则并导入数据，再开始优化。";
      return;
    }
    const total = state.dataset ? state.dataset.total : 0;
    if (total < 5) {
      hint.textContent = `当前只有 ${total} 通通话，至少需要 5 通（且人工列中要同时出现违规和正常）。`;
      return;
    }
    if (state.settings && state.settings.mode === "deepseek" && !state.settings.hasApiKey) {
      hint.textContent = "真实模型模式尚未配置 API Key，请先在「模型设置」中填写。";
      return;
    }
    if (state.settings && state.settings.mode === "mock") {
      hint.textContent = `当前为模拟演示：不调用大模型，使用本地启发式算法；语义部分不参与判定。当前 ${total} 通通话。`;
    } else {
      hint.textContent = `已就绪：${total} 通通话，将调用 ${state.settings.model} 迭代优化。`;
    }
  }

  // ---------------------------------------------------------------- 优化任务
  async function startJob(kind) {
    if (!state.currentId) return toast("请先保存规则", "error");
    if (state.job && ["queued", "running"].includes(state.job.status)) return toast("已有任务正在运行", "error");

    const targetPct = Number($("#targetAccuracy").value);
    const maxRounds = Number($("#maxRounds").value);
    const sampleLimit = Number($("#sampleLimit").value);
    if (kind === "optimize") {
      if (!(targetPct >= 50 && targetPct <= 100)) return toast("目标准确率需在 50% 到 100% 之间", "error");
      if (!(maxRounds >= 1 && maxRounds <= 10)) return toast("轮次需在 1 到 10 之间", "error");
    }

    $("#jobResult").innerHTML = "";
    $("#casesCard").hidden = true;
    state.job = null;
    $("#startBtn").disabled = true;
    $("#evaluateBtn").disabled = true;

    try {
      const data = kind === "evaluate"
        ? await api(`/api/rules/${state.currentId}/evaluate`, { method: "POST" })
        : await api("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ruleId: state.currentId,
              targetAccuracy: targetPct / 100,
              maxRounds,
              sampleLimit,
            }),
          });
      state.jobId = data.jobId;
      pollJob(data.jobId);
    } catch (error) {
      $("#startBtn").disabled = false;
      $("#evaluateBtn").disabled = false;
      toast(error.message, "error");
    }
  }

  async function pollJob(jobId) {
    clearTimeout(state.timer);
    try {
      const job = await api(`/api/jobs/${jobId}`);
      state.job = job;
      renderJob(job);
      if (!TERMINAL.has(job.status)) {
        state.timer = setTimeout(() => pollJob(jobId), 900);
        return;
      }
      $("#startBtn").disabled = false;
      $("#evaluateBtn").disabled = false;
      if (job.status === "failed") {
        toast(job.error || "任务失败", "error");
      } else if (job.status === "reached") {
        toast("优化达标，结果已生成", "success");
      } else if (job.status === "exhausted") {
        toast("优化结束，未达到目标准确率");
      } else if (job.status === "cancelled") {
        toast("任务已取消");
      } else {
        toast("评测完成");
      }
      if (job.hasCases) await loadJobCases(jobId);
      await loadRuns();
      await loadRules();
    } catch (error) {
      $("#startBtn").disabled = false;
      $("#evaluateBtn").disabled = false;
      toast(error.message, "error");
    }
  }

  async function cancelJob() {
    if (!state.jobId) return;
    try {
      await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
      toast("已请求取消");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function renderJob(job) {
    const grid = $("#progressCard").parentElement;
    if (!job) {
      $("#progressCard").hidden = true;
      grid.classList.add("one");
      $("#jobResult").innerHTML = "";
      $("#casesCard").hidden = true;
      return;
    }
    grid.classList.remove("one");
    $("#progressCard").hidden = false;
    $("#jobPhase").textContent = job.phase || "准备中";
    const info = statusInfo(job.status);
    $("#jobStatus").textContent = info.text;
    $("#jobStatus").className = `badge ${info.cls}`;
    const done = job.progress ? job.progress.done : 0;
    const total = job.progress ? job.progress.total : 0;
    const terminal = TERMINAL.has(job.status);
    $("#jobBar").style.width = terminal ? "100%" : total ? `${Math.min(100, (done / total) * 100)}%` : "0%";
    $("#jobProgressText").textContent = terminal ? "已完成" : total ? `${done} / ${total} 条` : "准备中";
    $("#jobElapsed").textContent = elapsedText(job);
    $("#jobLog").textContent = (job.logs || []).slice(-12)
      .map((item) => `${timeOnly(item.at)}  ${item.message}`).join("\n");
    $("#cancelBtn").hidden = !["queued", "running"].includes(job.status);
    renderJobResult(job);
  }

  function compareBox(title, rule) {
    if (!rule) return `<div class="box"><h4>${esc(title)}</h4><pre>（暂无）</pre></div>`;
    const logic = rule.logic === "AND" ? "AND 与" : "OR 或";
    return `<div class="box"><h4>${esc(title)} · ${logic}</h4>
      <pre>正则：${esc(rule.regex || "（空）")}

语义：${esc(rule.semantic || "（空）")}</pre></div>`;
  }

  function promptBox(title, text) {
    return `<div class="box"><h4>${esc(title)}</h4><pre>${esc(text || "（空）")}</pre></div>`;
  }

  function renderGate(gate) {
    const statusMap = {
      PASS: ["通过", "badge-ok"],
      FAIL: ["未通过", "badge-fail"],
      INSUFFICIENT: ["样本不足", "badge-mock"],
    };
    const [label, cls] = statusMap[gate.status] || [gate.status, ""];
    const reasons = (gate.reasons || []).map((reason) => `<li>${esc(reason)}</li>`).join("");
    const checks = (gate.checks || []).map((check) => `
      <div class="gate-check ${check.pass ? "ok" : "bad"}">
        <span class="gate-icon">${check.pass ? "✓" : "✗"}</span>
        <span class="gate-label">${esc(check.label)}</span>
        <span class="muted">${esc(check.detail)}</span>
      </div>`).join("");
    return `<div class="card gate-card">
      <div class="card-head">
        <h3>泛化验收</h3>
        <span class="badge ${cls}">${label}</span>
      </div>
      <ul class="gate-reasons">${reasons}</ul>
      <div class="gate-checks">${checks}</div>
    </div>`;
  }

  function renderJobResult(job) {
    const container = $("#jobResult");
    if (!job.baseline) {
      container.innerHTML = job.error ? `<div class="card"><p class="muted">${esc(job.error)}</p></div>` : "";
      return;
    }
    const baselineTrain = job.baseline;
    const bestTrain = job.best || baselineTrain;
    const baselineValidation = job.baselineValidation || baselineTrain;
    const bestValidation = job.bestValidation || baselineValidation;
    const baselineTest = job.baselineTest || baselineTrain;
    const bestTest = job.bestTest || baselineTest;
    const target = job.target || 0;
    const summary = job.splitSummary || {};
    const gap = Math.abs((bestValidation.accuracy || 0) - (bestTest.accuracy || 0));
    const gate = job.gate;

    let html = "";
    if (gate) html += renderGate(gate);

    const splitRows = [
      ["TRAIN", "训练集", baselineTrain, bestTrain, summary.TRAIN],
      ["VALIDATION", "验证集", baselineValidation, bestValidation, summary.VALIDATION],
      ["TEST", "测试集", baselineTest, bestTest, summary.TEST],
    ];
    const tableRows = splitRows.map(([key, label, base, best, counts]) => {
      const item = counts || {};
      return `<tr>
        <td><span class="chip">${label}</span></td>
        <td>${item.total || 0} 条（违规 ${item.positives || 0} / 正常 ${item.negatives || 0}）</td>
        <td>${pct(base.accuracy)}</td>
        <td><b>${pct(best.accuracy)}</b></td>
        <td>${best.fp} / ${best.fn}</td>
      </tr>`;
    }).join("");

    html += `<div class="card">
      <div class="card-head">
        <h3>${job.kind === "evaluate" ? "当前规则评测" : "训练 / 验证 / 测试"}</h3>
        <span class="muted">测试集只在最后评一次，不参与选择候选</span>
      </div>
      <div class="stats">
        ${stat("验证集准确率", pct(bestValidation.accuracy), target && bestValidation.accuracy >= target ? "green" : "red")}
        ${stat("测试集准确率", pct(bestTest.accuracy), target && bestTest.accuracy >= target - 0.05 ? "green" : "red")}
        ${stat("测试集基线→最优", `${pct(baselineTest.accuracy)} → ${pct(bestTest.accuracy)}`, "blue")}
        ${stat("验证-测试差距", `${(gap * 100).toFixed(1)} pt`, gap <= 0.15 ? "green" : "red")}
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>集合</th><th>样本</th><th>基线准确率</th><th>最优准确率</th><th>误报 / 漏报</th></tr></thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
      <div class="compare">${compareBox("基线规则", job.baselineRule)}${compareBox("最优规则", job.bestRule)}</div>
    </div>`;

    if (job.iterations && job.iterations.length) {
      html += `<div class="card"><div class="card-head"><h3>迭代过程</h3>
        <span class="muted">共 ${job.iterations.length} 轮</span></div>${renderIterations(job)}</div>`;
    }

    const improved = (bestTest.accuracy || 0) > (baselineTest.accuracy || 0)
      || (bestValidation.accuracy || 0) > (baselineValidation.accuracy || 0);
    if (job.bestRule && ["reached", "exhausted"].includes(job.status)) {
      const gatePass = gate && gate.status === "PASS";
      const disabled = !improved || !gatePass;
      const last = job.iterations && job.iterations.length
        ? job.iterations[job.iterations.length - 1]
        : null;
      const defaultReason = last && last.rationale
        ? last.rationale
        : "应用优化后的规则，原因见迭代记录";
      html += `<div class="card best-panel">
        <div class="best-head">
          <div><h3>可应用的最优版本</h3>
            <p class="muted">验证集 ${pct(bestValidation.accuracy)}，测试集 ${pct(bestTest.accuracy)}${improved ? "" : "，与基线持平"}</p>
          </div>
          <span class="badge ${gatePass ? "badge-ok" : "badge-fail"}">${gatePass ? "验收通过" : "验收未通过"}</span>
        </div>
        <label class="full">
          <span>应用理由 <em class="muted">必填，会写入规则的修改历史</em></span>
          <textarea id="applyReason" rows="2">${esc(defaultReason)}</textarea>
        </label>
        ${gatePass ? "" : `<label class="checkline">
          <input id="applyForce" type="checkbox" />
          我已知晓泛化验收未通过（${esc((gate && gate.status) || "未知")}），仍要人工确认应用
        </label>`}
        <div class="form-actions">
          <button class="btn primary" id="applyBtn" data-apply="${esc(job.id)}" ${disabled ? "disabled" : ""}>应用为当前规则</button>
          ${!improved ? '<span class="muted">相对基线没有提升，不能应用</span>' : ""}
        </div>
        ${job.promptChanged ? `<p class="muted">本轮同时修改了通用语义提示词，应用后会更新所有规则共用的提示词。</p>
          <div class="compare">${promptBox("提示词修改前", job.baselinePrompt)}${promptBox("提示词修改后", job.bestPrompt)}</div>` : ""}
        <div class="compare">${compareBox("待应用规则", job.bestRule)}</div>
      </div>`;
    }
    container.innerHTML = html;
  }

  function renderIterations(job) {
    let previous = job.baselineRule || {};
    let previousPrompt = job.baselinePrompt || "";
    return job.iterations.map((iteration) => {
      const before = previous;
      const after = iteration.rule;
      const promptBefore = previousPrompt;
      if (iteration.accepted) {
        previous = after;
        if (iteration.promptChanged && iteration.semanticPrompt) previousPrompt = iteration.semanticPrompt;
      }
      const badge = iteration.accepted
        ? '<span class="badge badge-accepted">已采纳</span>'
        : '<span class="badge badge-rejected">未超过最优</span>';
      const metrics = iteration.metrics || {};
      const validation = iteration.validationMetrics || {};
      const promptBlock = iteration.promptChanged
        ? `<div class="rationale"><b>提示词修改理由：</b>${esc(iteration.promptRationale || "未说明")} ${iteration.accepted ? "" : "（本轮未采纳）"}</div>
           <div class="compare">${promptBox("提示词修改前", promptBefore)}${promptBox("提示词修改后", iteration.semanticPrompt)}</div>`
        : "";
      return `<div class="iteration ${iteration.accepted ? "accepted" : "rejected"}">
        <div class="iteration-head">
          <span class="title">第 ${iteration.index} 轮</span>
          <span class="acc">验证 ${pct(validation.accuracy)}</span>
          <span class="muted">训练 ${pct(metrics.accuracy)}</span>
          ${badge}
          <span class="muted">耗时 ${((iteration.durationMs || 0) / 1000).toFixed(1)} 秒</span>
        </div>
        <div class="rationale"><b>修改理由：</b>${esc(iteration.rationale || "无修改说明")}</div>
        ${iteration.expectedEffect ? `<div class="expected">预期效果：${esc(iteration.expectedEffect)}</div>` : ""}
        <div class="compare">${compareBox("修改前", before)}${compareBox("修改后", after)}</div>
        ${promptBlock}
        <div class="chip-row">
          <span class="chip">验证精确率 ${pct(validation.precision)}</span>
          <span class="chip">验证召回率 ${pct(validation.recall)}</span>
          <span class="chip">验证误报 ${validation.fp}</span>
          <span class="chip">验证漏报 ${validation.fn}</span>
          <span class="chip">${iteration.source === "mock" ? "本地启发式" : "大模型"}</span>
        </div>
      </div>`;
    }).join("");
  }

  async function loadJobCases(jobId) {
    try {
      const split = state.caseSplit || "TRAIN";
      const [best, baseline] = await Promise.all([
        api(`/api/jobs/${jobId}/cases?which=best&split=${split}`),
        api(`/api/jobs/${jobId}/cases?which=baseline&split=${split}`).catch(() => null),
      ]);
      state.caseWithReason = Boolean(best && best.withReason);
      const baseMap = new Map(((baseline && baseline.rows) || []).map((row) => [row.callId, row]));
      state.caseRows = (best.rows || []).map((row) => ({
        ...row,
        baselinePredicted: baseMap.has(row.callId) ? baseMap.get(row.callId).predicted : row.predicted,
      }));
      $("#casesCard").hidden = !state.caseRows.length;
      renderCases();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function renderCases() {
    const rows = state.caseFilter === "errors"
      ? state.caseRows.filter((row) => row.correct === false || row.predicted == null)
      : state.caseRows;
    if (!rows.length) {
      $("#casesTable").innerHTML = '<div class="empty">没有符合条件的记录</div>';
      return;
    }
    $("#casesTable").innerHTML = `<table>
      <thead><tr><th>通话ID</th><th>人工复核</th><th>基线判定</th><th>最优判定</th><th>结果</th><th>正则理由</th><th>语义理由</th><th>通话文本</th></tr></thead>
      <tbody>${rows.map((row) => {
        const result = row.predicted == null
          ? '<span class="pill pill-u">未判定</span>'
          : row.correct
            ? '<span class="pill pill-ok">正确</span>'
            : '<span class="pill pill-bad">错误</span>';
        const regexReason = row.regexReason
          || (row.regexHit === true ? "正则命中" : row.regexHit === false ? "正则未命中" : "未配置正则");
        let semanticReason = row.semanticReason || "";
        if (!state.caseWithReason) semanticReason = "未生成（仅正则）";
        else if (!semanticReason) semanticReason = row.semanticHit == null ? "未调用语义" : "未生成";
        return `<tr class="clickable">
          <td class="mono">${esc(row.callId)}</td>
          <td>${labelPill(row.humanResult)}</td>
          <td>${predPill(row.baselinePredicted)}</td>
          <td>${predPill(row.predicted)}</td>
          <td>${result}</td>
          <td class="reason-cell">${esc(regexReason)}</td>
          <td class="reason-cell">${esc(semanticReason)}</td>
          <td class="transcript">${transcriptBlock(row.transcript)}</td>
        </tr>`;
      }).join("")}</tbody></table>`;
  }

  async function applyRun(runId, button) {
    if (!runId) return;
    const reason = ($("#applyReason") ? $("#applyReason").value : "").trim();
    if (!reason) return toast("请填写应用理由", "error");
    const force = $("#applyForce") ? $("#applyForce").checked : false;
    const original = button.textContent;
    button.disabled = true;
    button.textContent = "应用中…";
    try {
      await api(`/api/runs/${runId}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason, force }),
      });
      toast("已应用为当前规则，修改理由已记录", "success");
      state.dirty = false;
      await loadRules();
      await loadPrompt();
      if (state.currentId) await selectRule(state.currentId);
      await loadRuns();
    } catch (error) {
      toast(error.message, "error");
      button.disabled = false;
      button.textContent = original;
    }
  }

  async function loadRuns() {
    if (!state.currentId) {
      $("#historyCard").hidden = true;
      return;
    }
    const runs = await api(`/api/runs?ruleId=${state.currentId}`);
    if (!runs.length) {
      $("#historyCard").hidden = true;
      return;
    }
    $("#historyCard").hidden = false;
    $("#historyList").innerHTML = runs.slice(0, 8).map((run) => {
      const accuracy = run.best ? pct(run.best.accuracy) : "—";
      const apply = run.bestRule && !run.applied
        ? `<button class="btn small" data-apply="${esc(run.id)}" type="button">应用</button>`
        : run.applied ? '<span class="badge badge-ok">已应用</span>' : "";
      return `<div class="history-item">
        <span class="muted">${fmtTime(run.finishedAt || run.createdAt)}</span>
        <span class="grow">${statusBadge(run.status)} 最优准确率 ${accuracy}</span>
        ${apply}
      </div>`;
    }).join("");
  }

  // ---------------------------------------------------------------- 设置
  async function loadSettings() {
    state.settings = await api("/api/settings");
    renderMode();
  }

  function renderMode() {
    const mode = (state.settings && state.settings.mode) || "mock";
    const badge = $("#modeBadge");
    badge.textContent = mode === "deepseek" ? "真实模型" : "模拟演示";
    badge.className = `badge ${mode === "deepseek" ? "badge-live" : "badge-mock"}`;
  }

  function openSettings() {
    const settings = state.settings || {};
    $("#settingMode").value = settings.mode || "mock";
    $("#settingBaseUrl").value = settings.baseUrl || "https://api.deepseek.com";
    $("#settingModel").value = settings.model || "deepseek-chat";
    $("#settingApiKey").value = "";
    $("#settingApiKey").placeholder = settings.hasApiKey ? "已配置，留空表示不修改" : "请输入 API Key";
    $("#settingsStatus").textContent = "";
    $("#settingsDialog").showModal();
  }

  async function saveSettings() {
    const mode = $("#settingMode").value;
    const payload = {
      mode,
      baseUrl: $("#settingBaseUrl").value.trim() || "https://api.deepseek.com",
      model: $("#settingModel").value.trim() || "deepseek-chat",
    };
    const key = $("#settingApiKey").value.trim();
    if (key) payload.apiKey = key;
    try {
      await withBusy($("#saveSettingsBtn"), "保存中…", async () => {
        state.settings = await api("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        renderMode();
        $("#settingsDialog").close();
        updateOptimizeHint();
        toast("模型设置已保存", "success");
      });
    } catch (error) {
      $("#settingsStatus").textContent = error.message;
      toast(error.message, "error");
    }
  }

  async function testConnection() {
    const payload = {
      mode: $("#settingMode").value,
      baseUrl: $("#settingBaseUrl").value.trim() || "https://api.deepseek.com",
      model: $("#settingModel").value.trim() || "deepseek-chat",
    };
    const key = $("#settingApiKey").value.trim();
    if (key) payload.apiKey = key;
    $("#settingsStatus").textContent = "正在测试连接…";
    try {
      const result = await api("/api/settings/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      $("#settingsStatus").textContent = result.ok ? "连接成功，模型可用。" : "连接失败。";
    } catch (error) {
      $("#settingsStatus").textContent = `连接失败：${error.message}`;
    }
  }
})();
