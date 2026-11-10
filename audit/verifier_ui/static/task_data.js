/* Task data is isolated from the existing verifier/candidate state. */
window.TaskDataUI = (() => {
  const q = s => document.querySelector(s);
  const h = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const state = {catalog: null, items: [], image: '', direction: '', index: 0, reviewFilter: '', busy: false};
  const params = () => new URLSearchParams(location.search);
  const pass = () => params().get('pass') === '2' ? 2 : 1;
  const tab = () => params().get('taskTab') === 'components' ? 'components' : 'review';
  let savedActor = '';
  try { savedActor = localStorage.getItem('task-data-annotator') || ''; } catch (_) {}
  const actor = () => {
    const value = q('#td-actor')?.value.trim() || savedActor.trim();
    if (!value) throw new Error('请先填写标注人，再保存。');
    return value;
  };
  async function api(path, body) {
    const r = await fetch('/api/task-data/' + path, body === undefined ? {} : {
      method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify(body)});
    const data = await r.json();
    if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请检查输入');
    return data;
  }
  function status(message, error=false) {
    const node = q('#td-status');
    if (node) { node.textContent = message; node.classList.toggle('td-error', error); }
  }
  async function action(fn) {
    if (state.busy) return;
    state.busy = true;
    try { await fn(); } catch (error) { status(error.message, true); }
    finally { state.busy = false; }
  }
  const options = (values, selected, empty) => `<option value="">${h(empty)}</option>` + values.map(v =>
    `<option value="${h(v)}" ${v === selected ? 'selected' : ''}>${h(v)}</option>`).join('');
  const picture = (kind, id, key, caption) => `<figure><img loading="lazy" alt="${h(caption)}" src="/api/task-data/image/${kind}/${encodeURIComponent(id)}/${key}"><figcaption>${h(caption)}</figcaption></figure>`;
  const nameOf = (task, group) => task + (group ? ' · ' + (state.catalog.mappings.find(m => m.purpose === group)?.name || ({water:'水面',playground:'运动场'}[group] || group)) : '');
  const setText = value => `{${value.t1.join(', ')}} → {${value.t2.join(', ')}}`;
  const mappingForCard = group => {
    const m = state.catalog.mappings.find(m => m.purpose === group);
    const changed = JSON.stringify(m.value) !== JSON.stringify(m.generated_value);
    return `目的：${m.name}；生成此真值时：${setText(m.generated_value)}` +
      (changed ? `；当前映射：${setText(m.value)}（需要重新生成真值后再审）` : '');
  };
  function go(taskTab, number=1) {
    const p = params(); p.set('mode', 'task-data'); p.set('taskTab', taskTab); p.set('pass', number);
    history.pushState({}, '', '?' + p); state.index = 0; state.image = ''; state.direction = '';
    render().catch(e => status(e.message, true));
  }
  function shell() {
    const meta = state.catalog.metadata;
    q('#main').innerHTML = `<section class="td"><div class="td-toolbar">
      <div class="td-tabs"><button id="td-review-tab" class="${tab()==='review'?'on':''}">真值审查</button><button id="td-component-tab" class="${tab()==='components'?'on':''}">组件标注</button></div>
      <div class="td-row"><span class="td-source">${meta.source==='synthetic'?'合成演示数据':meta.preview?'真实数据 · 轻量预览':'真实任务数据'}</span><label>标注人 <input class="td-actor" id="td-actor" value="${h(savedActor)}" placeholder="填写姓名"></label>
      <a href="/api/task-data/export/task_review.json" download>导出审查</a><a href="/api/task-data/export/second_component_labels.json" download>导出标注</a></div></div>
      <div id="td-status" class="td-status" role="status" aria-live="polite"></div>
      ${(meta.notes||[]).map(n=>`<div class="td-notice">${h(n)}</div>`).join('')}
      <div id="td-content"></div></section>`;
    q('#td-review-tab').onclick = () => go('review');
    q('#td-component-tab').onclick = () => go('components');
    q('#td-actor').oninput = e => { savedActor = e.target.value; try {localStorage.setItem('task-data-annotator', savedActor);} catch (_) {} };
    q('#main').querySelectorAll('img').forEach(bindImageError);
  }
  function bindImageError(image) {
    image.onerror = () => { image.alt = '图片读取失败，请刷新重试'; status('有图片读取失败，服务端已重试；请刷新或检查登记路径。', true); };
  }
  function mappingPanel() {
    return `<details class="card td-mapping" open><summary>R1 映射表 · 6 个目的</summary>
      <p class="meta">仅在两期非零且地类不同的变化像素上取下列集合。0 表示未变化且没有地类。</p>
      <table><thead><tr><th>目的</th><th>T1 类别集合</th><th>→ T2 类别集合</th><th>修改</th></tr></thead><tbody>
      ${state.catalog.mappings.map(m => `<tr data-purpose="${h(m.purpose)}"><td><b>${h(m.name)}</b></td>
        ${['t1','t2'].map(k=>`<td>${state.catalog.legend.slice(1).map(c=>`<label title="${h(c.name)}"><input type="checkbox" data-side="${k}" value="${c.code}" ${m.value[k].includes(c.code)?'checked':''}>${c.code}</label>`).join('')}</td>`).join('')}
        <td><button class="td-action" data-save-mapping="${h(m.purpose)}">保存</button>${JSON.stringify(m.value)!==JSON.stringify(m.generated_value)?'<span class="meta"> 待重新生成真值</span>':''}</td></tr>`).join('')}</tbody></table>
      <div class="td-legend">${state.catalog.legend.slice(1).map(c=>`<span>${c.code} ${h(c.name)}</span>`).join('')}</div>
      <div class="td-notice">映射改动需要重新生成真值后再审。保存映射会记录修改人和时间，不会自动重算当前真值。</div></details>`;
  }
  function reviewView() {
    const all = state.catalog.examples;
    const groups = [...new Set(all.map(e => `${e.task}|${e.group}`))];
    const selected = state.reviewFilter ? all.filter(e => `${e.task}|${e.group}` === state.reviewFilter) : all;
    q('#td-content').innerHTML = `<div class="td-eyebrow">TASK DATA / GROUND TRUTH</div><h2>真值审查</h2>
      <p class="td-intro">核对指令、两期标签和真值。每个任务选取大、小与随机例子；R1 按目的、R3 按参照物分别抽样。</p>
      <div class="td-toolbar"><label>任务 <select id="td-review-filter">${options(groups, state.reviewFilter, '全部任务')}</select></label><span>${all.filter(e=>e.verdict).length} / ${all.length} 例已审</span></div>
      <div class="meta">${(state.catalog.metadata.sampling||[]).map(s=>`${h(nameOf(s.task,s.group))}：${s.selected}/3 例${s.insufficient?`（非空条目仅 ${s.available} 个，数量不足）`:''}`).join(' · ')}</div>
      ${mappingPanel()}
      <div class="td-legend">${state.catalog.legend.map(c=>`<span><i class="td-swatch" style="background:rgb(${c.rgb.join(',')})"></i>${c.code} ${h(c.name)}</span>`).join('')}</div>
      <p class="meta">真值叠加：黄色半透明填充、红色轮廓。显示尺寸最多 512 像素，面积按原始真值计算。</p>
      ${selected.map(e=>`<article class="card td-example" data-example="${h(e.id)}"><div class="td-row"><h3>${h(nameOf(e.task,e.group))} · ${h(e.task_type)}</h3><span class="td-source">${h(e.selection)}</span><span>真值面积 <b>${e.area.toLocaleString()}</b> px</span></div>
        <div class="meta">${h(e.image)}${e.task==='T4.1'?' · LEVIR 验证集；面积阈值真值仅来自 LEVIR 标签':''}</div>
        <p class="td-instruction">${h(e.instruction)}</p>
        ${e.task==='R1'?`<div>${h(mappingForCard(e.group))}</div>`:''}
        <div class="td-grid">${picture('example',e.id,'t1','T1 原图')}${picture('example',e.id,'t2','T2 原图')}${e.semantic_available?picture('example',e.id,'s1','T1 语义标签')+picture('example',e.id,'s2','T2 语义标签'):'<div class="td-notice">LEVIR 仅有二值变化标签，无两期语义地类标签。</div>'}${picture('example',e.id,'overlay','真值叠加在 T2')}</div>
        <div class="td-row td-card-actions"><button class="td-choice ${e.verdict==='reasonable'?'on':''}" data-verdict="reasonable">合理</button><button class="td-choice ${e.verdict==='unreasonable'?'on':''}" data-verdict="unreasonable">不合理</button><input class="td-note" aria-label="一句话备注" placeholder="可留一句话备注" value="${h(e.note)}"><button class="td-action" data-note>保存备注</button><button class="td-action" data-undo-review>撤销</button></div>
        <div class="meta">${e.reviewed_at?`${h(e.annotator)} · ${h(e.reviewed_at)}`:'未审查'}</div></article>`).join('') || '<p class="empty">尚无审查例子，请先导入。</p>'}`;
    // Render human-readable task labels while keeping stable filter values.
    q('#td-review-filter').querySelectorAll('option').forEach(o=>{if(o.value){const [t,g]=o.value.split('|');o.textContent=nameOf(t,g);}});
    q('#td-review-filter').onchange = e => {state.reviewFilter=e.target.value;reviewView();};
    document.querySelectorAll('[data-save-mapping]').forEach(button=>button.onclick=()=>action(async()=>{
      const who=actor(), row=button.closest('tr'), body={annotator:who};
      for(const side of ['t1','t2']) body[side]=[...row.querySelectorAll(`input[data-side="${side}"]:checked`)].map(x=>Number(x.value));
      await api('mappings/'+encodeURIComponent(button.dataset.saveMapping),body);
      state.catalog=await api('catalog');reviewView();status('映射已保存并记录修改历史；需要重新生成真值后再审。');
    }));
    document.querySelectorAll('[data-example]').forEach(card=>{
      const ident=card.dataset.example, record=all.find(e=>e.id===ident);
      const save=verdict=>action(async()=>{await api('reviews/'+encodeURIComponent(ident),{verdict,note:card.querySelector('.td-note').value,annotator:actor()});state.catalog=await api('catalog');reviewView();status('审查已保存。');});
      card.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>save(b.dataset.verdict));
      card.querySelector('[data-note]').onclick=()=>save(record.verdict);
      card.querySelector('[data-undo-review]').onclick=()=>action(async()=>{const r=await api('undo',{kind:'review',target:ident,annotator:actor()});state.catalog=await api('catalog');reviewView();status(r.target?'已撤销最近一次审查修改。':'没有可撤销的修改。');});
    });
    q('#td-content').querySelectorAll('img').forEach(bindImageError);
  }
  async function componentView(restoreId) {
    const result = await api('components?pass_number='+pass());
    state.items = result.items;
    const filtered = state.items.filter(c=>(!state.image||c.image===state.image)&&(!state.direction||c.direction===state.direction));
    if(restoreId) {const i=filtered.findIndex(c=>c.id===restoreId);if(i>=0)state.index=i;}
    state.index = Math.max(0,Math.min(state.index,filtered.length-1));
    const c=filtered[state.index], count=filtered.filter(x=>x.labeled_at).length, blind=pass()===2;
    q('#td-content').innerHTML=`<div class="td-eyebrow">TASK DATA / COMPONENT LABELS</div><h2>${blind?'第二遍 · 盲标':'建筑变化组件标注'}</h2>
      <p class="td-intro">8 连通组件，面积 ≥64 px。新建主要看 T2，拆除主要看 T1；两期近景使用相同裁剪与红色轮廓。</p>
      <div class="td-toolbar"><div class="td-tabs"><button id="td-first-pass" class="${!blind?'on':''}">第一遍</button><button id="td-second-pass" class="${blind?'on':''}">第二遍</button></div>
      <div class="td-row"><label>图 <select id="td-image">${options([...new Set(state.items.map(x=>x.image))],state.image,'全部图')}</select></label><label>方向 <select id="td-direction"><option value="">全部方向</option><option value="new" ${state.direction==='new'?'selected':''}>新建</option><option value="demolished" ${state.direction==='demolished'?'selected':''}>拆除</option></select></label></div>
      <div><progress class="td-progress" max="${Math.max(1,filtered.length)}" value="${count}"></progress> 已标 ${count} / ${filtered.length}<span class="meta">（本遍总计 ${result.labeled}/${result.total}）</span></div></div>
      ${blind?'<div class="td-notice">第二遍按种子 1016 抽取 20%。本视图不显示第一遍结果和颜色预填，请独立选择两个属性。</div>':'<p class="meta">颜色来自组件像素 HSV 中位数，仅作为初值；子类请人工选择。多栋混色或无法判断的组件，导出时标记为忽略。</p>'}
      <div id="td-current-card">${c?`<article class="card"><div class="td-row td-component-title"><h3>${h(c.image)} · ${c.direction==='new'?'新建':'拆除'}</h3><span>${state.index+1} / ${filtered.length} · 面积 <b>${c.area.toLocaleString()}</b> px · bbox [${c.bbox.join(', ')}]</span></div>
      <div class="td-component-layout"><div><div class="td-closeups">${picture('component',c.id,'t1','T1 近景'+(c.direction==='demolished'?' · 主要判读期':''))}${picture('component',c.id,'t2','T2 近景'+(c.direction==='new'?' · 主要判读期':''))}</div><div class="td-position">${picture('component',c.id,'thumb','整图位置 · 红框与红色轮廓')}</div><div class="meta">组件 ${h(c.id)} · ${h(c.pool)}</div></div>
      <aside><b>屋顶颜色</b><div id="td-colors" class="td-attributes">${state.catalog.colors.map((color,i)=>`<button class="td-choice" data-color="${h(color)}"><kbd>${i+1}</kbd>${h(color)}</button>`).join('')}</div><b>子类</b><div id="td-subtypes" class="td-attributes">${state.catalog.subtypes.map((subtype,i)=>`<button class="td-choice" data-subtype="${h(subtype)}"><kbd>${['Q','W','E','R'][i]}</kbd>${h(subtype)}</button>`).join('')}</div>
      ${!blind?`<p class="meta" id="td-prefill">HSV 预填：${h(c.prefill_color)}（可改）</p>`:''}<p class="meta">${c.labeled_at?`本遍已保存：${h(c.annotator)} · ${h(c.labeled_at)}`:'本遍未保存'}</p></aside></div>
      <div class="td-row td-card-actions"><button id="td-prev" class="td-action" ${state.index===0?'disabled':''}>上一张</button><button id="td-save" class="td-choice">保存并下一张 <kbd>空格</kbd></button><button id="td-next" class="td-action" ${state.index===filtered.length-1?'disabled':''}>下一张</button><button id="td-undo" class="td-action">撤销 <kbd>Z</kbd></button></div></article>`:'<div class="card empty">当前范围没有组件。请先导入数据或调整筛选条件。</div>'}</div>`;
    q('#td-first-pass').onclick=()=>go('components',1);q('#td-second-pass').onclick=()=>go('components',2);
    for(const [id,key] of [['#td-image','image'],['#td-direction','direction']])q(id).onchange=e=>{state[key]=e.target.value;state.index=0;action(()=>componentView());};
    if(!c)return;
    state.current = {id:c.id,color:c.color || (!blind?c.prefill_color:null),subtype:c.subtype};
    function highlight(){document.querySelectorAll('[data-color]').forEach(b=>b.classList.toggle('on',b.dataset.color===state.current.color));document.querySelectorAll('[data-subtype]').forEach(b=>b.classList.toggle('on',b.dataset.subtype===state.current.subtype));}
    document.querySelectorAll('[data-color]').forEach(b=>b.onclick=()=>{state.current.color=b.dataset.color;highlight();});
    document.querySelectorAll('[data-subtype]').forEach(b=>b.onclick=()=>{state.current.subtype=b.dataset.subtype;highlight();});highlight();
    q('#td-prev').onclick=()=>action(async()=>{state.index--;await componentView();});
    q('#td-next').onclick=()=>action(async()=>{state.index++;await componentView();});
    q('#td-save').onclick=()=>action(async()=>{
      if(!state.current.color||!state.current.subtype)throw new Error('请同时选择屋顶颜色和子类。');
      await api('labels/'+encodeURIComponent(c.id),{...state.current,pass_number:pass(),annotator:actor()});
      const atEnd=state.index===filtered.length-1;if(!atEnd)state.index++;await componentView();status(atEnd?'已保存；已到当前筛选的最后一张。':'已保存，进入下一张。');
    });
    q('#td-undo').onclick=()=>action(async()=>{const r=await api('undo',{kind:'component',pass_number:pass(),annotator:actor()});await componentView(r.target);status(r.target?'已撤销最近一次保存，并返回该组件。':'没有可撤销的保存。');});
    q('#td-content').querySelectorAll('img').forEach(bindImageError);
  }
  async function render(){shell();if(tab()==='review')reviewView();else await componentView();}
  async function start(){try{state.catalog=await api('catalog');await render();}catch(error){q('#main').textContent='任务数据加载失败：'+error.message;}}
  window.addEventListener('popstate',()=>{if(params().get('mode')==='task-data'){state.index=0;render().catch(e=>status(e.message,true));}});
  window.addEventListener('keydown',event=>{
    if(params().get('mode')!=='task-data'||tab()!=='components'||state.busy||event.ctrlKey||event.metaKey||event.altKey||event.repeat)return;
    if(event.target.closest('input,textarea,select,[contenteditable="true"]'))return;
    const key=event.key.toLowerCase();let button;
    if(/^[1-7]$/.test(key))button=document.querySelectorAll('[data-color]')[Number(key)-1];
    else if(['q','w','e','r'].includes(key))button=document.querySelectorAll('[data-subtype]')[['q','w','e','r'].indexOf(key)];
    else if(key===' ')button=q('#td-save');else if(key==='z')button=q('#td-undo');
    if(button){event.preventDefault();button.click();}
  });
  return {start};
})();
