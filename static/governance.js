(() => {
  "use strict";
  const {state, api, esc, toast} = window.QC;
  const $ = s => document.querySelector(s);
  let data = null, refreshTimer = null;
  const labels = {DRAFT:"候选经验", APPROVED:"已审核", REJECTED:"已驳回", RETIRED:"已停用", FROZEN:"已冻结",
    EVALUATING:"独立评测中", EVALUATED:"独立验收通过", EVALUATION_FAILED:"独立验收未通过", ERROR:"评测异常",
    INTERRUPTED:"评测已中断", SHADOW:"影子运行", CANARY:"灰度运行", PUBLISHED:"正式运行", ROLLED_BACK:"已回滚",
    PENDING:"待人工复核", ACCEPTED:"已纳入数据"};
  const badge = status => `<span class="chip">${esc(labels[status] || status)}</span>`;
  const empty = text => `<div class="empty small">${text}</div>`;
  const splitIds = value => String(value || "").split(/[\s,;，；、]+/).filter(Boolean);
  const post = (url, payload, method="POST") => api(url, {method, headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  const actorOptions = [{value:"UNKNOWN",label:"待复核 / 无法判断"},{value:"VIOLATION",label:"确认违规"},{value:"NORMAL",label:"确认正常"}];
  function selectedRule() {
    const rule = state.rules.find(r => r.id === state.currentId);
    if (!rule) throw new Error("请先选择并保存一条规则");
    return rule;
  }
  function fieldHtml(f) {
    let control;
    const value = f.value ?? "";
    const attrs = `${f.optional ? "" : "required"} ${f.readonly ? "readonly" : ""}`;
    if (f.type === "textarea") control = `<textarea name="${f.name}" rows="${f.rows || 3}" ${attrs}>${esc(value)}</textarea>`;
    else if (f.type === "select") control = `<select name="${f.name}" ${attrs}>${(f.options || []).map(o => `<option value="${esc(o.value)}" ${String(o.value)===String(value)?"selected":""}>${esc(o.label)}</option>`).join("")}</select>`;
    else if (f.type === "checkbox") control = `<input name="${f.name}" type="checkbox" ${value ? "checked" : ""} />`;
    else control = `<input name="${f.name}" type="${f.type || "text"}" ${f.type === "file" ? 'accept=".csv,.json,.xlsx"' : `value="${esc(value)}"`} ${attrs} ${f.min != null ? `min="${f.min}"` : ""} ${f.max != null ? `max="${f.max}"` : ""} />`;
    return `<label class="${f.type === "checkbox" ? "checkbox-field" : ""}"><span>${esc(f.label)}</span>${control}${f.hint ? `<small class="muted">${esc(f.hint)}</small>` : ""}</label>`;
  }
  function modal(title, fields, submit, note="") {
    const dialog = document.createElement("dialog");
    dialog.className = "governance-dialog";
    dialog.innerHTML = `<form><h3>${esc(title)}</h3>${note ? `<p class="hint">${esc(note)}</p>` : ""}<div class="governance-fields">${fields.map(fieldHtml).join("")}</div><p class="form-error" role="alert"></p><div class="dialog-actions"><button class="btn primary" type="submit">确认保存</button><button class="btn ghost" type="button" data-close>取消</button></div></form>`;
    document.body.append(dialog);
    dialog.addEventListener("close", () => dialog.remove());
    dialog.querySelector("[data-close]").onclick = () => dialog.close();
    dialog.querySelector("form").onsubmit = async event => {
      event.preventDefault();
      const form = event.target, values = {};
      for (const f of fields) {
        const input = form.elements.namedItem(f.name);
        values[f.name] = f.type === "checkbox" ? input.checked : f.type === "file" ? input.files[0] : String(input.value).trim();
      }
      const button = form.querySelector('[type="submit"]');
      button.disabled = true;
      form.querySelector(".form-error").textContent = "";
      try { await submit(values); dialog.close(); await refresh(); toast("已保存", "success"); }
      catch (e) { form.querySelector(".form-error").textContent = e.message; }
      finally { button.disabled = false; }
    };
    dialog.showModal();
    return dialog;
  }
  const namedFields = () => [{name:"reviewer",label:"审核 / 操作人"},{name:"reason",label:"依据与理由",type:"textarea"}];
  async function populateKnowledge(rule) {
    try {
      const result = await api("/api/governance");
      if (rule.id && rule.id !== state.currentId) return;
      const linked = new Set(rule.knowledgeIds || []);
      $("#ruleKnowledge").innerHTML = result.knowledge.filter(k => k.status === "APPROVED" || linked.has(k.id)).map(k =>
        `<option value="${esc(k.id)}" ${linked.has(k.id)?"selected":""}>${esc(k.title)}${k.status !== "APPROVED" ? "（已失效，请移除）" : ""}</option>`).join("");
    } catch (e) { toast(e.message, "error"); }
  }
  async function refresh() {
    clearTimeout(refreshTimer);
    data = await api("/api/governance");
    $("#governanceSummary").innerHTML = `<div class="chip-row"><span class="chip">数据版本 ${data.datasetVersion}</span><span class="chip">历史快照 ${data.datasetVersions.length}</span><span class="chip">知识 ${data.knowledge.length}</span><span class="chip">冻结版本 ${data.releases.length}</span></div>`;
    $("#migrationWarnings").innerHTML = (data.migrationWarnings || []).map(w => `<p class="hint">${esc(w.message || w)}</p>`).join("");
    $("#knowledgeList").innerHTML = data.knowledge.map(k => `<article class="asset-item"><div class="card-head"><b>${esc(k.title)}</b>${badge(k.status)}</div><p>${esc(k.content)}</p><p class="muted">范围：${esc(k.scope)} · 例外：${esc(Array.isArray(k.exceptions) ? k.exceptions.join("；") : k.exceptions)}</p><p class="muted">来源 ${esc((k.sourceCallIds || []).join("、"))}；反例 ${esc((k.counterExamples || []).join("、"))}</p><div class="link-row"><code>${esc(k.id)}</code>${k.status==="DRAFT" ? `<button class="btn small" data-knowledge="${k.id}" data-action="APPROVE">审核通过</button><button class="btn small ghost" data-knowledge="${k.id}" data-action="REJECT">驳回</button>` : k.status==="APPROVED" ? `<button class="btn small ghost" data-knowledge="${k.id}" data-action="RETIRE">停用</button>` : ""}</div></article>`).join("") || empty("还没有专家知识。用已复核训练案例提炼经验，并提供正确反例。");
    $("#holdoutList").innerHTML = data.holdouts.map(h => `<article class="asset-item"><b>${esc(h.name)}</b><span class="chip">${h.count} 通</span><p class="muted">${esc(h.id)} · ${h.usedBy ? `已绑定 ${esc(h.usedBy)}` : "尚未用于验收"}</p></article>`).join("") || empty("请导入与开发数据独立的新通话。封存集一旦使用，只能重试同一冻结版本。");
    $("#releaseList").innerHTML = data.releases.map(r => {
      const actions = {FROZEN:["EVALUATE","REJECT"],ERROR:["EVALUATE"],INTERRUPTED:["EVALUATE"],EVALUATED:["APPROVE","REJECT"],APPROVED:["SHADOW","REJECT"],SHADOW:["CANARY","ROLLBACK"],CANARY:["PUBLISH","ROLLBACK"],PUBLISHED:["ROLLBACK"]}[r.status] || [];
      const actionNames = {EVALUATE:"独立验收",APPROVE:"人工审核",REJECT:"驳回",SHADOW:"进入影子",CANARY:"进入灰度",PUBLISH:"正式发布",ROLLBACK:"回滚"};
      const m = r.evaluation?.candidate;
      const metrics = m ? `<p class="chip-row"><span class="chip">准确率 ${(m.accuracy*100).toFixed(1)}%</span><span class="chip">召回率 ${(m.recall*100).toFixed(1)}%</span><span class="chip">覆盖率 ${(m.coverage*100).toFixed(1)}%</span></p>` : "";
      return `<article class="asset-item"><div class="card-head"><b>${esc(r.ruleName)} · ${esc(r.id)}</b>${badge(r.status)}</div><p class="muted">提交人 ${esc(r.author)} · 数据版本 ${r.datasetVersion} · 包指纹 ${esc(r.bundleHash.slice(0,14))}</p>${metrics}${r.error ? `<p class="form-error">${esc(r.error)}</p>` : ""}${(r.evaluation?.gate?.reasons || []).map(x=>`<p class="hint">${esc(x)}</p>`).join("")}<div class="link-row">${actions.map(a=>`<button class="btn small ${a==="ROLLBACK"?"danger ghost":"ghost"}" data-release="${r.id}" data-action="${a}">${actionNames[a]}</button>`).join("")}<a class="link" href="/api/releases/${r.id}/manifest" target="_blank" rel="noopener">查看冻结清单</a>${["SHADOW","CANARY","PUBLISHED"].includes(r.status) ? `<button class="btn small ghost" data-runtime="${r.ruleId}">运行试判</button>` : ""}</div><details><summary>审核与操作记录</summary>${(r.history || []).map(h=>`<p class="muted">${esc(h.at)} · ${esc(h.reviewer)} · ${esc(h.action)}：${esc(h.reason)}</p>`).join("")}</details></article>`;
    }).join("") || empty("冻结当前能力或开发验收通过的候选后，使用封存数据完成独立验收。");
    $("#feedbackList").innerHTML = data.feedback.map(f => `<article class="asset-item"><b>${esc(f.callId)}</b>${badge(f.status)}<p>${esc(f.reason)} · 人工标签 ${esc(f.label)} · 提交人 ${esc(f.reviewer)}</p><details><summary>查看反馈原文与版本</summary><p>${esc(f.releaseId)}</p><pre>${esc(f.transcript)}</pre></details>${f.status==="PENDING" ? `<button class="btn small ghost" data-feedback="${f.id}">复核后入库</button>` : ""}</article>`).join("") || empty("反馈须有人工标签并由另一位复核人确认，才会形成新数据版本。");
    if (state.activeTab === "governance" && data.releases.some(r=>r.status==="EVALUATING")) refreshTimer = setTimeout(()=>refresh().catch(e=>toast(e.message,"error")), 2500);
  }
  async function annotate() {
    const rule = selectedRule();
    const ds = await api(`/api/rules/${rule.id}/dataset?limit=500`);
    while (ds.rows.length < ds.callTotal) {
      const page = await api(`/api/rules/${rule.id}/dataset?limit=500&offset=${ds.rows.length}`);
      if (!page.rows.length) break;
      ds.rows.push(...page.rows);
    }
    if (!ds.rows.length) throw new Error("请先导入通话");
    const dialog = modal("人工标注复核",[
      {name:"callId",label:"通话",type:"select",options:ds.rows.map(c=>({value:c.callId,label:`${c.callId} · ${{NORMAL:"正常",VIOLATION:"违规",UNKNOWN:"待复核"}[c.humanResult]} · ${{TRAIN:"训练",VALIDATION:"验证",TEST:"测试"}[c.split]}`}))},
      {name:"transcript",label:"原始通话",type:"textarea",rows:6,readonly:true,optional:true,value:ds.rows[0].transcript},
      {name:"label",label:"人工结论",type:"select",options:actorOptions,value:ds.rows[0].humanResult},
      {name:"quote",label:"证据原文（可选，每行一段）",type:"textarea",optional:true},
      ...[["complaintIntent","客户有投诉意图"],["adequateResponse","坐席回应适当"],["timelyResponse","坐席回应及时"]].map(([name,label]) => ({name,label:label+"（可选人工标注）",type:"select",optional:true,options:[{value:"",label:"未标注 / 不适用"},{value:"true",label:"是"},{value:"false",label:"否"}]})),
      ...namedFields()
    ], async values => {
      const steps = Object.fromEntries(["complaintIntent","adequateResponse","timelyResponse"].filter(k=>values[k]).map(k=>[k,values[k]==="true"]));
      const existing = ds.rows.find(c=>c.callId===values.callId)?.humanEvidence || [];
      const evidence = values.quote === existing.map(e=>e.quote).join("\n").trim() ? existing :
        values.quote.split(/\r?\n/).map(quote=>quote.trim()).filter(Boolean).map(quote=>({quote}));
      await post("/api/annotations",{...values,steps,ruleId:rule.id,evidence},"PUT");
      await window.QC.loadRules(); await window.QC.loadDataset();
    }, "标注保存会形成新数据版本，使旧实验的应用资格失效。");
    const showAnnotation = c => {
      dialog.querySelector('[name="transcript"]').value=c.transcript;
      dialog.querySelector('[name="label"]').value=c.humanResult;
      dialog.querySelector('[name="quote"]').value=(c.humanEvidence || []).map(e=>e.quote).join("\n");
      for (const name of ["complaintIntent", "adequateResponse", "timelyResponse"]) {
        const value = c.humanSteps?.[name];
        dialog.querySelector(`[name="${name}"]`).value=typeof value === "boolean" ? String(value) : "";
      }
      dialog.querySelector('[name="reason"]').value="";
    };
    dialog.querySelector('[name="callId"]').onchange = e => showAnnotation(ds.rows.find(x=>x.callId===e.target.value));
    showAnnotation(ds.rows[0]);
  }
  function newKnowledge() {
    const rule = selectedRule();
    const proposal = state.job?.iterations?.slice().reverse().find(x=>x.proposedKnowledge)?.proposedKnowledge || {};
    modal("提炼候选经验",[
      {name:"title",label:"经验名称",value:proposal.title},{name:"content",label:"经验内容",type:"textarea",value:proposal.content},
      {name:"scope",label:"适用范围",type:"textarea",value:proposal.scope},
      {name:"exceptions",label:"排除条件",type:"textarea",value:Array.isArray(proposal.exceptions)?proposal.exceptions.join("；"):proposal.exceptions},
      {name:"sourceCallIds",label:"已复核训练来源（通话ID，逗号分隔）",value:(proposal.sourceCallIds||[]).join(",")},
      {name:"counterExamples",label:"正确反例（训练通话ID，逗号分隔）",value:(proposal.counterExamples||[]).join(",")},
      {name:"reason",label:"提炼依据",type:"textarea"}
    ], v => post("/api/knowledge",{...v,ruleId:rule.id,sourceCallIds:splitIds(v.sourceCallIds),counterExamples:splitIds(v.counterExamples)}),"候选经验需审核后在规则配置中明确关联；模型生成的提案不会自动生效。");
  }
  async function freeze() {
    const rule=selectedRule();
    const runs=await api(`/api/runs?ruleId=${encodeURIComponent(rule.id)}`);
    modal("冻结能力版本",[
      {name:"runId",label:"候选来源",type:"select",optional:true,options:[{value:"",label:"当前已保存的开发草稿"},...runs.filter(r=>r.gate?.status==="PASS").map(r=>({value:r.id,label:`实验 ${r.id}`}))]},
      {name:"author",label:"提交人"},{name:"reason",label:"冻结理由",type:"textarea"}
    ],v=>post("/api/releases",{...v,runId:v.runId||null,ruleId:rule.id}),"冻结后不可编辑，验收失败需重新冻结。共享规则、已批准知识、模型配置和指标要求一并固定。");
  }
  function holdout() {
    modal("导入封存独立集",[
      {name:"name",label:"独立集名称"},{name:"file",label:"CSV / JSON / XLSX",type:"file"},
      {name:"fullyReviewed",label:"确认完整复核当前全部规则（否则使用文件中的已复核范围）",type:"checkbox"}
    ],v=>{const form=new FormData();Object.entries(v).forEach(([k,x])=>form.append(k,x));return api("/api/holdouts/import",{method:"POST",body:form});},"请提供未参与开发和知识提炼的新数据。已知重复通话、文本及同源分组会被拒绝。");
  }
  async function releaseAction(id,action) {
    if (action === "EVALUATE") {
      await refresh();
      const choices=data.holdouts.filter(h=>!h.usedBy||h.usedBy===id).map(h=>({value:h.id,label:`${h.name} · ${h.count} 通`}));
      if (!choices.length) throw new Error("请先导入新的独立验收集");
      modal("独立验收",[{name:"holdoutId",label:"封存数据",type:"select",options:choices}],v=>post(`/api/releases/${id}/evaluate`,v),"将调用真实模型。封存数据仅用于冻结版本的独立验收，不提供给优化器。");
    } else {
      const fields=namedFields();
      if(action==="CANARY") fields.push({name:"percent",label:"候选流量比例（%）",type:"number",min:1,max:50,value:10});
      modal("版本操作 · "+action,fields,v=>post(`/api/releases/${id}/transition`,{...v,action,percent:Number(v.percent||10)}),"审核人与提交人需不同。此操作控制本地 /api/runtime/judge 的运行版本，不会自动部署到外部生产系统。");
    }
  }
  async function newFeedback() {
    await refresh();
    if(!data.releases.length) throw new Error("请先建立冻结能力版本");
    modal("提交人工反馈",[
      {name:"releaseId",label:"实际使用的版本",type:"select",options:data.releases.map(r=>({value:r.id,label:`${r.ruleName} · ${r.id}`}))},
      {name:"callId",label:"通话ID"},{name:"transcript",label:"通话原文",type:"textarea",rows:6},
      {name:"label",label:"人工结论",type:"select",options:actorOptions},
      {name:"transcriptComplete",label:"确认转写包含完整通话结尾",type:"checkbox"},...namedFields()
    ],v=>post("/api/feedback",v));
  }
  function runtime(ruleId) {
    modal("运行版本试判",[{name:"callId",label:"通话ID（用于稳定灰度分流）"},{name:"transcript",label:"带角色的通话原文",type:"textarea",rows:6},{name:"transcriptComplete",label:"确认包含完整通话结尾",type:"checkbox"}], async v=>{
      const result=await post("/api/runtime/judge",{...v,ruleId});
      const dialog=document.createElement("dialog");dialog.className="governance-dialog";
      dialog.innerHTML=`<h3>运行判定</h3><pre>${esc(JSON.stringify(result,null,2))}</pre><button class="btn" type="button">关闭</button>`;
      document.body.append(dialog);dialog.querySelector("button").onclick=()=>{dialog.close();dialog.remove();};dialog.showModal();
    },"将调用真实模型；待复核结果会保持未知，影子结果不会替换主结果。");
  }
  function evidenceHtml(row) {
    if(!Object.keys(row.steps || {}).length && !(row.evidence || []).length && !row.reviewRequired) return "";
    const names={complaintIntent:"投诉意图",adequateResponse:"回应适当",timelyResponse:"回应及时"};
    return `<details class="evidence-details"><summary>${row.reviewRequired?"待复核":"查看步骤与证据"}</summary>${Object.entries(row.steps||{}).map(([k,v])=>`<p>${esc(names[k]||k)}：${v===null?"未知/不适用":v?"是":"否"}</p>`).join("")}${(row.evidence||[]).map(e=>`<p>${esc(e.turnId)} ${esc(e.role)}：「${esc(e.quote)}」</p>`).join("")}${(row.evidenceErrors||[]).map(e=>`<p class="form-error">${esc(e)}</p>`).join("")}</details>`;
  }
  window.QCGovernance={refresh,populateKnowledge,evidenceHtml};
  document.addEventListener("DOMContentLoaded",()=>{
    const invoke = action => () => Promise.resolve().then(action).catch(e=>toast(e.message,"error"));
    $("#annotateBtn").onclick=invoke(annotate);
    $("#newKnowledgeBtn").onclick=invoke(newKnowledge);
    $("#freezeReleaseBtn").onclick=invoke(freeze);
    $("#importHoldoutBtn").onclick=invoke(holdout);
    $("#newFeedbackBtn").onclick=invoke(newFeedback);
    $("#refreshGovernanceBtn").onclick=invoke(refresh);
    $("#tab-governance").addEventListener("click",event=>{
      const b=event.target.closest("button");if(!b)return;
      if(b.dataset.knowledge) modal("专家知识审核",namedFields(),v=>post(`/api/knowledge/${b.dataset.knowledge}/review`,{...v,action:b.dataset.action}));
      if(b.dataset.release) invoke(()=>releaseAction(b.dataset.release,b.dataset.action))();
      if(b.dataset.feedback) modal("反馈复核入库",namedFields(),async v=>{await post(`/api/feedback/${b.dataset.feedback}/accept`,v);await window.QC.loadRules();await window.QC.loadDataset();});
      if(b.dataset.runtime) runtime(b.dataset.runtime);
    });
  });
})();
