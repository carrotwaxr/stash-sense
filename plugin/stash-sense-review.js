/**
 * Stash Sense review framework UI (#127).
 *
 * Renders any review type generically from its preview: evidence, the exact
 * outcome (fields before/after, files kept/deleted, bytes reclaimed) and the
 * choices the user may change. Accept sends back the plan token it showed;
 * deletes need a confirm that names every file. Also the batch accept and the
 * apply log with revert.
 */
(function () {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }
  const esc = (v) => SS.escapeHtml(v == null ? '' : String(v));

  // ==================== API ====================

  async function call(method, path, body) {
    const settings = await SS.getSettings();
    const result = await SS.runPluginOperation('review', {
      sidecar_url: settings.sidecarUrl, method, path, body,
    });
    if (result && result.error) {
      const err = new Error(result.error);
      err.status = result.status;
      err.detail = result.detail || {};
      throw err;
    }
    return result;
  }

  const ReviewAPI = {
    types: () => call('GET', '/review/types'),
    preview: (recId, option) => call('POST', `/review/${recId}/preview`, { option: option || null }),
    accept: (recId, option, planToken, confirmDelete) => call('POST', `/review/${recId}/accept`, {
      option, plan_token: planToken, confirm_delete: confirmDelete || [],
    }),
    batchPreview: (type) => call('POST', `/review/types/${encodeURIComponent(type)}/batch/preview`),
    batchAccept: (type, items) => call('POST', `/review/types/${encodeURIComponent(type)}/batch/accept`, { items }),
    applyLog: (params = {}) => {
      const qs = new URLSearchParams();
      if (params.type) qs.set('type', params.type);
      qs.set('limit', params.limit || 50);
      qs.set('offset', params.offset || 0);
      return call('GET', `/review/apply-log?${qs}`);
    },
    revert: (entryId, force) => call('POST', `/review/apply-log/${entryId}/revert`, { force: !!force }),
    async still(args) {
      const settings = await SS.getSettings();
      const r = await SS.runPluginOperation('review_still', { sidecar_url: settings.sidecarUrl, ...args });
      if (r.error) throw new Error(r.error);
      return r.data_uri;
    },
  };

  async function dismiss(recId) {
    const settings = await SS.getSettings();
    const r = await SS.runPluginOperation('rec_dismiss', { sidecar_url: settings.sidecarUrl, rec_id: recId });
    if (r.error) throw new Error(r.error);
  }

  async function listRecs(type, status, limit, offset) {
    const settings = await SS.getSettings();
    const r = await SS.runPluginOperation('rec_list', {
      sidecar_url: settings.sidecarUrl, type, status, limit, offset,
    });
    if (r.error) throw new Error(r.error);
    return r;
  }

  // ==================== Modal ====================

  /** A modal with arbitrary HTML and buttons. Resolves with the clicked button's value. */
  function modal({ title, html, buttons }) {
    return new Promise((resolve) => {
      const overlay = SS.createElement('div', { className: 'ss-modal-overlay ss-review-modal-overlay' });
      const box = SS.createElement('div', { className: 'ss-review-modal' });
      box.innerHTML = `<h3>${esc(title)}</h3><div class="ss-review-modal-body">${html}</div>`;
      const actions = SS.createElement('div', { className: 'ss-review-modal-actions' });
      for (const b of buttons) {
        const btn = SS.createElement('button', { className: `ss-btn ${b.className || 'ss-btn-secondary'}`, textContent: b.label });
        btn.addEventListener('click', () => { overlay.remove(); resolve(b.value); });
        actions.appendChild(btn);
      }
      box.appendChild(actions);
      overlay.appendChild(box);
      overlay.addEventListener('click', (e) => { if (e.target === overlay) { overlay.remove(); resolve(null); } });
      document.body.appendChild(overlay);
    });
  }

  function fileListHtml(files) {
    return `<table class="ss-review-files"><tbody>${files.map(f => `
      <tr><td class="ss-review-path">${esc(f.path)}</td><td class="ss-review-size">${SS.formatSize(f.size || 0)}</td></tr>
    `).join('')}</tbody></table>`;
  }

  function destinationText(trashPath, count) {
    const it = count === 1 ? 'it' : 'them';
    return trashPath
      ? `Stash moves ${it} to its trash folder <code>${esc(trashPath)}</code>.`
      : `Stash has no trash folder configured: <strong>${count === 1 ? 'it is' : 'they are'} deleted permanently</strong>.`;
  }

  /** The named delete confirm. Resolves true only on an explicit confirm. */
  async function confirmDelete(files, trashPath, intro) {
    const total = files.reduce((n, f) => n + (f.size || 0), 0);
    const v = await modal({
      title: `Delete ${files.length} file${files.length === 1 ? '' : 's'} (${SS.formatSize(total)})?`,
      html: `${intro ? `<p>${intro}</p>` : ''}${fileListHtml(files)}<p>${destinationText(trashPath, files.length)}</p>`,
      buttons: [
        { label: 'Cancel', value: false },
        { label: `Delete ${files.length} file${files.length === 1 ? '' : 's'}`, value: true, className: 'ss-btn-danger' },
      ],
    });
    return v === true;
  }

  function notice(container, text, kind = 'info') {
    const el = SS.createElement('div', { className: `ss-review-notice ss-review-notice-${kind}` });
    el.innerHTML = text;
    container.prepend(el);
    return el;
  }

  // ==================== Card ====================

  function fmtValue(v) {
    if (v === null || v === undefined || v === '') return '<em>empty</em>';
    if (Array.isArray(v)) return v.length ? esc(v.join(', ')) : '<em>none</em>';
    return esc(v);
  }

  const FLAG_LABELS = {
    supercut_candidate: 'longer cut kept too',
    upscale_pair: 'upscale + original',
    held_unhealthy: 'held: no healthy file',
    tie: 'practically equal',
  };

  function fmtMbps(bitRate) {
    return bitRate ? `${(bitRate / 1e6).toFixed(1)} Mbps` : '';
  }

  function fmtFps(fps) {
    return fps ? `${Number(fps).toFixed(Number.isInteger(Number(fps)) ? 0 : 2)} fps` : '';
  }

  function fmtOffset(s) {
    return s == null ? '' : ` at ${s >= 0 ? '+' : '-'}${Math.abs(s).toFixed(1)} s`;
  }

  /** One file of a duplicate set: size, codec, resolution, bitrate, fps, duration, health and what the plan does with it. */
  function dupFileHtml(f, keepIds, deleteIds) {
    const id = String(f.file_id);
    const fate = keepIds.has(id) ? '<span class="ss-review-badge ss-review-badge-keep">keep</span>'
      : deleteIds.has(id) ? '<span class="ss-review-badge ss-review-badge-delete">delete</span>' : '';
    const unhealthy = f.health && f.health !== 'ok'
      ? `<span class="ss-review-badge ss-review-badge-warn">${esc(f.health)}</span>` : '';
    const facts = [
      SS.formatSize(f.size || 0),
      f.video_codec ? esc(f.video_codec) : '',
      f.width ? `${esc(f.width)}×${esc(f.height)}` : '',
      esc(fmtMbps(f.bit_rate)),
      esc(fmtFps(f.frame_rate)),
      f.duration ? SS.formatDuration(f.duration) : '',
    ].filter(Boolean).join(' · ');
    return `<div class="ss-review-dup-file ${keepIds.has(id) ? 'is-keep' : deleteIds.has(id) ? 'is-delete' : ''}" data-file-id="${esc(f.file_id)}">
      <div>${fate}${f.is_primary ? '<span class="ss-review-badge">primary</span>' : ''}${unhealthy}</div>
      <div class="ss-review-path">${esc(f.path)}</div>
      <div class="ss-review-dup-facts">${facts}</div>
    </div>`;
  }

  function dupPairsHtml(ev) {
    const pairs = ev.pairs || [];
    if (!pairs.length) return '';
    const rows = pairs.map(pr => {
      const e = pr.evidence || {};
      const sprite = Array.isArray(e.sprite) ? `d85 ${esc(e.sprite[0])} / d80 ${esc(e.sprite[1])}`
        : (e.sprite && typeof e.sprite === 'object' ? `d85 ${esc(e.sprite.d85)} / d80 ${esc(e.sprite.d80)}` : '');
      const audio = e.ber == null ? '' : `${esc(Number(e.ber).toFixed(2))}${esc(fmtOffset(e.ber_offset_s))}`;
      const kf = e.keyframe_fraction == null ? '' : `${esc(Math.round(e.keyframe_fraction * 100))}%${esc(fmtOffset(e.keyframe_offset_s))}`;
      const src = Array.isArray(pr.sources) ? pr.sources.join(', ') : (pr.sources || '');
      return `<tr><td>#${esc(pr.scene_a)} ↔ #${esc(pr.scene_b)}</td><td>${esc(pr.cls)}</td>
        <td>${audio}</td><td>${sprite}</td><td>${kf}</td><td>${esc(src)}</td></tr>`;
    }).join('');
    return `<table class="ss-review-ev-files ss-review-dup-pairs"><thead><tr>
      <th>Scenes</th><th>Class</th><th>Audio BER</th><th>Sprites</th><th>Keyframes</th><th>Found by</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  /** A duplicate set: one column per scene, the keeper reason, flags and the evidence per pair. */
  function dupEvidenceHtml(ev, p) {
    const keepIds = new Set((p.plan.keep_files || []).map(f => String(f.file_id)));
    const deleteIds = new Set((p.plan.delete_files || []).map(f => String(f.file_id)));
    const cols = ev.scenes.map(s => {
      const badges = [
        s.play_count ? `<span class="ss-review-badge">${esc(s.play_count)} plays</span>` : '',
        s.o_counter ? `<span class="ss-review-badge">O ${esc(s.o_counter)}</span>` : '',
        s.rating100 != null ? `<span class="ss-review-badge">rated ${esc(s.rating100)}</span>` : '',
        s.organized ? '<span class="ss-review-badge ss-review-badge-ok">organized</span>' : '',
      ].join('');
      return `<div class="ss-review-dup-scene">
        <div class="ss-review-ev-scene"><a href="/scenes/${esc(s.id)}" target="_blank">#${esc(s.id)}</a> ${esc(s.title || '(untitled)')}</div>
        <div class="ss-review-dup-meta">${s.date ? `${esc(s.date)} ` : ''}${badges}</div>
        ${(s.files || []).map(f => dupFileHtml(f, keepIds, deleteIds)).join('')}
      </div>`;
    }).join('');
    const flags = (ev.flags || []).map(f => `<span class="ss-review-badge ss-review-badge-warn">${esc(FLAG_LABELS[f] || f)}</span>`).join('');
    return `<div class="ss-review-dup-scenes">${cols}</div>
      ${ev.keeper_reason ? `<div class="ss-review-reason">${esc(ev.keeper_reason)}${flags ? `<div class="ss-review-flags">${flags}</div>` : ''}</div>` : ''}
      ${dupPairsHtml(ev)}`;
  }

  /** mm:ss, or h:mm:ss; unlike SS.formatDuration it prints 0:00 for t = 0. */
  function fmtT(t) {
    return SS.formatDuration(Math.max(0, Number(t) || 0)).replace('Unknown', '0:00');
  }

  /** The still nearest to time t, or null. */
  function nearestStill(stills, t) {
    let best = null;
    for (const s of stills) {
      if (best === null || Math.abs(s.t - t) < Math.abs(best.t - t)) best = s;
    }
    return best;
  }

  /** A split card: file header, one row per cut (stills and playback fill in from loadMedia), leftovers. */
  function splitEvidenceHtml(ev) {
    const f = ev.file || {};
    const head = `<div class="ss-review-split-head">
      ${ev.scene ? `<div class="ss-review-ev-scene">Scene <a href="/scenes/${esc(ev.scene.id)}" target="_blank">#${esc(ev.scene.id)}</a> ${esc(ev.scene.title || '(untitled)')}</div>` : ''}
      <div class="ss-review-path">${esc(f.path || '')}</div>
      <div class="ss-review-dup-meta">${ev.duration ? `${esc(fmtT(ev.duration))} · ` : ''}confidence
        <span class="ss-review-badge ${ev.confidence === 'high' ? 'ss-review-badge-ok' : 'ss-review-badge-warn'}">${esc(ev.confidence || 'low')}</span></div>
      ${(ev.reasons || []).length ? `<ul class="ss-review-split-reasons">${ev.reasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
    </div>`;

    if (ev.redundant) {
      const rows = (ev.covering || []).map(c => `<li><a href="/scenes/${esc(c.scene_id)}" target="_blank">Scene #${esc(c.scene_id)}</a>
        ${c.start != null && c.end != null ? `<span class="ss-review-dup-meta">${esc(fmtT(c.start))} to ${esc(fmtT(c.end))}</span>` : ''}</li>`).join('');
      return `${head}<div class="ss-review-split-covering"><strong>Held by</strong><ul>${rows}</ul></div>`;
    }

    const rows = (ev.cuts || []).map((c, i) => `<div class="ss-review-split-cut" data-cut="${i}">
      <div class="ss-review-split-cut-head">
        <span class="ss-review-split-time">${esc(fmtT(c.t))}</span>
        <span class="ss-review-badge">${esc(c.kind || 'cut')}</span>
        <span class="ss-review-split-evidence">${esc(c.evidence || '')}</span>
      </div>
      <div class="ss-review-split-stills"></div>
    </div>`).join('');
    const snapped = (ev.snapped || []).map(x =>
      `<div class="ss-review-dup-meta">${esc(fmtT(x.from))} snapped to ${esc(fmtT(x.to))}, the start of the black</div>`).join('');
    const left = (ev.leftovers || []).length
      ? `<div class="ss-review-split-leftovers"><strong>Unconfirmed blacks</strong><ul>${ev.leftovers.map(l =>
        `<li>${esc(fmtT(l.t))} <span class="ss-review-badge">${esc(l.kind || 'black')}</span> ${esc(l.evidence || '')}</li>`).join('')}</ul></div>`
      : '';
    return `${head}${rows || '<div class="ss-review-noop">No cuts: the file stays whole.</div>'}${snapped}${left}`;
  }

  function evidenceHtml(ev, p) {
    if (Array.isArray(ev.scenes) && p) return dupEvidenceHtml(ev, p);
    if (Array.isArray(ev.cuts)) return splitEvidenceHtml(ev);
    const parts = [];
    if (ev.scene) {
      parts.push(`<div class="ss-review-ev-scene">Scene <a href="/scenes/${esc(ev.scene.id)}" target="_blank">#${esc(ev.scene.id)}</a> ${esc(ev.scene.title)}</div>`);
    }
    if (Array.isArray(ev.files)) {
      parts.push(`<table class="ss-review-ev-files"><thead><tr>
        <th>File</th><th>Size</th><th>Video</th><th>Duration</th><th></th></tr></thead><tbody>
        ${ev.files.map(f => `<tr data-file-id="${esc(f.file_id)}">
          <td class="ss-review-path">${esc(f.path)}</td>
          <td>${SS.formatSize(f.size || 0)}</td>
          <td>${esc(f.video_codec || '')} ${f.width ? `${esc(f.width)}×${esc(f.height)}` : ''}</td>
          <td>${f.duration ? SS.formatDuration(f.duration) : ''}</td>
          <td>${f.is_primary ? '<span class="ss-review-badge">primary</span>' : ''}
              ${f.identical_to_keeper ? '<span class="ss-review-badge ss-review-badge-ok">identical</span>' : ''}</td>
        </tr>`).join('')}</tbody></table>`);
    }
    return parts.join('');
  }

  function outcomeHtml(p) {
    const plan = p.plan;
    const rows = [];
    for (const c of plan.field_changes) {
      rows.push(`<tr><td>${esc(c.field)}</td><td class="ss-review-before">${fmtValue(c.before)}</td>
        <td>→</td><td class="ss-review-after">${fmtValue(c.after)}</td></tr>`);
    }
    if (plan.primary_file) {
      rows.push(`<tr><td>primary file</td><td class="ss-review-before">#${esc(plan.primary_file.before)}</td>
        <td>→</td><td class="ss-review-after">#${esc(plan.primary_file.after)}</td></tr>`);
    }
    let mergeHtml = '';
    if (plan.merge) {
      const m = plan.merge;
      const vals = Object.entries(m.values || {}).map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td class="ss-review-after">${fmtValue(v)}</td></tr>`).join('');
      mergeHtml = `<div class="ss-review-merge">Merge scenes ${(m.sources || []).map(i => `#${esc(i)}`).join(', ')} into #${esc(m.destination)}
        ${vals ? `<table class="ss-review-changes"><tbody>${vals}</tbody></table>` : ''}</div>`;
    }
    let splitHtml = '';
    if (plan.split) {
      const ranges = plan.split.ranges || [];
      splitHtml = `<div class="ss-review-split-handoff"><strong>Hand off to stash-curator: split into ${ranges.length} scene${ranges.length === 1 ? '' : 's'}</strong>
        <ol>${ranges.map(r => `<li>${esc(fmtT(r.start))} – ${esc(fmtT(r.end))} (${esc(fmtT(r.end - r.start))})</li>`).join('')}</ol>
        <div class="ss-review-dup-meta">Nothing changes in Stash until stash-curator runs the split.</div></div>`;
    }
    const redundant = !!(p.evidence && p.evidence.redundant);
    const leaveWhole = p.rec_type === 'scene_splits' && !plan.split && !redundant;
    const empty = !rows.length && !plan.delete_files.length && !plan.merge && !plan.split;
    return `
      ${mergeHtml}
      ${splitHtml}
      ${rows.length ? `<table class="ss-review-changes"><tbody>${rows.join('')}</tbody></table>` : ''}
      ${plan.keep_files.length ? `<div class="ss-review-keep"><strong>Keep</strong>${fileListHtml(plan.keep_files)}</div>` : ''}
      ${plan.delete_files.length ? `<div class="ss-review-delete"><strong>Delete</strong>${fileListHtml(plan.delete_files)}
        <div class="ss-review-reclaim">Reclaims ${SS.formatSize(plan.bytes_reclaimed)}</div></div>` : ''}
      ${leaveWhole ? '<p class="ss-review-noop">Leave whole: accepting marks this file done; nothing changes in Stash.</p>'
        : empty ? '<p class="ss-review-noop">No change: accepting only resolves it.</p>' : ''}
    `;
  }

  function choicesHtml(p) {
    return p.choices.map(c => {
      const cur = p.option[c.field];
      if (c.kind === 'pick') {
        return `<div class="ss-review-choice"><label>${esc(c.label)}</label><select data-field="${esc(c.field)}">
          ${c.values.map(v => `<option value="${esc(v.value)}" ${String(v.value) === String(cur) ? 'selected' : ''}>
            ${esc(v.label)}${String(v.value) === String(p.default_option[c.field]) ? ' (suggested)' : ''}</option>`).join('')}
        </select></div>`;
      }
      return `<div class="ss-review-choice"><label>${esc(c.label)}</label>
        <input type="text" data-field="${esc(c.field)}" value="${esc(cur)}"></div>`;
    }).join('');
  }

  /** Per cut: the stills nearest t - 3 and t + 3, and a button playing Stash's stream from t - 5 in place. */
  function loadSplitMedia(card, ev) {
    const stills = ev.stills || [];
    const sceneId = ev.scene ? ev.scene.id : (stills[0] || {}).scene_id;
    (ev.cuts || []).forEach((c, i) => {
      const row = card.querySelector(`.ss-review-split-cut[data-cut="${i}"] .ss-review-split-stills`);
      if (!row) return;
      for (const want of [c.t - 3, c.t + 3]) {
        const s = nearestStill(stills, want);
        if (!s) continue;
        const fig = SS.createElement('figure', { className: 'ss-review-still' });
        fig.innerHTML = `<div class="ss-review-still-ph">…</div><figcaption>${want < c.t ? 'before' : 'after'} @ ${esc(fmtT(s.t))}</figcaption>`;
        row.appendChild(fig);
        ReviewAPI.still({ scene_id: s.scene_id, file_id: s.file_id, t: s.t, width: 320 })
          .then(uri => { fig.querySelector('.ss-review-still-ph').outerHTML = `<img src="${uri}" alt="">`; })
          .catch(e => { fig.querySelector('.ss-review-still-ph').textContent = e.message; });
      }
      const from = (ev.playback_points || [])[i] != null ? ev.playback_points[i] : Math.max(0, c.t - 5);
      const btn = SS.createElement('button', { className: 'ss-btn ss-btn-secondary ss-btn-sm',
        textContent: `▶ from ${fmtT(from)}` });
      btn.addEventListener('click', () => {
        // Stash's own stream, seeked with a media fragment: no clip rendering.
        const video = SS.createElement('video', { className: 'ss-review-video', attrs: {
          controls: '', autoplay: '', src: `/scene/${encodeURIComponent(sceneId)}/stream#t=${from}`,
        } });
        btn.replaceWith(video);
      });
      row.appendChild(btn);
    });
  }

  async function loadMedia(card, ev) {
    const strip = card.querySelector('.ss-review-media');
    if (!strip) return;
    if (Array.isArray(ev.cuts)) {
      // The split card draws its stills inside each cut row, not in the generic strip.
      loadSplitMedia(card, ev);
      return;
    }
    let stills = ev.stills || [];
    if (Array.isArray(ev.scenes)) {
      // Side by side: the n-th file's frame of every scene forms one row.
      const per = ev.scenes.map(sc => stills.filter(s => String(s.scene_id) === String(sc.id)));
      const known = new Set(per.flat());
      const rest = stills.filter(s => !known.has(s));
      stills = [];
      for (let i = 0; per.some(g => i < g.length); i++) {
        for (const g of per) if (i < g.length) stills.push(g[i]);
      }
      stills.push(...rest);
      strip.classList.add('ss-review-media-dup');
    }
    for (const s of stills) {
      const fig = SS.createElement('figure', { className: 'ss-review-still' });
      const cap = Array.isArray(ev.scenes) ? `scene #${esc(s.scene_id)} @ ${SS.formatDuration(s.t)}`
        : `file #${esc(s.file_id)} @ ${SS.formatDuration(s.t)}`;
      fig.innerHTML = `<div class="ss-review-still-ph">…</div><figcaption>${cap}</figcaption>`;
      strip.appendChild(fig);
      ReviewAPI.still({ scene_id: s.scene_id, file_id: s.file_id, t: s.t, width: 320 })
        .then(uri => { fig.querySelector('.ss-review-still-ph').outerHTML = `<img src="${uri}" alt="">`; })
        .catch(e => { fig.querySelector('.ss-review-still-ph').textContent = e.message; });
    }
    if (ev.playback) {
      const btn = SS.createElement('button', { className: 'ss-btn ss-btn-secondary ss-btn-sm', textContent: '▶ Play here' });
      btn.addEventListener('click', () => {
        // Stash's own stream, seeked with a media fragment: no clip rendering.
        const video = SS.createElement('video', { className: 'ss-review-video', attrs: {
          controls: '', autoplay: '', src: `/scene/${encodeURIComponent(ev.playback.scene_id)}/stream#t=${ev.playback.t}`,
        } });
        btn.replaceWith(video);
      });
      strip.appendChild(btn);
    }
  }

  /** Render one recommendation as a review card into el. onDone() runs after accept/dismiss. */
  async function renderCard(el, recId, onDone, option, preset) {
    el.className = 'ss-review-card';
    el.innerHTML = '<div class="ss-loading-inline"><div class="ss-spinner"></div></div>';
    let p = preset;
    try {
      if (!p) p = await ReviewAPI.preview(recId, option);
    } catch (e) {
      el.innerHTML = `<div class="ss-review-error">Rec #${esc(recId)}: ${esc(e.message)}</div>`;
      return;
    }
    el.innerHTML = `
      <div class="ss-review-head">
        <span class="ss-review-title">#${esc(recId)} · ${esc(p.summary)}</span>
        ${p.high_confidence ? '<span class="ss-review-badge ss-review-badge-ok">high confidence</span>' : ''}
      </div>
      <div class="ss-review-body">
        <section class="ss-review-evidence"><h4>Evidence</h4>${evidenceHtml(p.evidence, p)}<div class="ss-review-media"></div></section>
        <section class="ss-review-outcome"><h4>Outcome</h4>${outcomeHtml(p)}</section>
      </div>
      ${p.choices.length ? `<div class="ss-review-choices">${choicesHtml(p)}</div>` : ''}
      <div class="ss-review-actions">
        <button class="ss-btn ss-btn-primary" data-act="accept">Accept</button>
        <button class="ss-btn ss-btn-secondary" data-act="dismiss">Dismiss</button>
      </div>`;
    loadMedia(el, p.evidence);

    // Changing a choice rebuilds the preview: the card always shows what accept does.
    el.querySelectorAll('[data-field]').forEach(input => {
      input.addEventListener('change', () => {
        const next = { ...p.option };
        el.querySelectorAll('[data-field]').forEach(i => { next[i.dataset.field] = i.value; });
        renderCard(el, recId, onDone, next);
      });
    });

    el.querySelector('[data-act="dismiss"]').addEventListener('click', async () => {
      try {
        await dismiss(recId);
      } catch (e) {
        notice(el, esc(e.message), 'error');
        return;
      }
      el.remove();
      onDone && onDone();
    });

    el.querySelector('[data-act="accept"]').addEventListener('click', async (ev) => {
      ev.target.disabled = true;
      const del = p.plan.delete_files;
      if (del.length && !(await confirmDelete(del, p.trash_path))) {
        ev.target.disabled = false;
        return;
      }
      try {
        const out = await ReviewAPI.accept(recId, p.option, p.plan_token, del.map(f => f.file_id));
        const warn = (out.warnings || []).map(w => `<div class="ss-review-error">${esc(w)}</div>`).join('');
        el.innerHTML = `<div class="ss-review-done">Applied (log #${esc(out.entry.id)})${out.label_id ? ' · your correction was saved as a labeled example' : ''}</div>${warn}`;
        if (!warn) setTimeout(() => { el.remove(); onDone && onDone(); }, 1500);
      } catch (e) {
        if (e.detail && e.detail.code === 'stale') {
          await renderCard(el, recId, onDone, null, e.detail.preview);
          notice(el, 'The library changed since this preview. This is the updated plan; nothing was applied.', 'warn');
        } else {
          ev.target.disabled = false;
          notice(el, esc(e.message), 'error');
        }
      }
    });
  }

  // ==================== Type list + batch ====================

  async function runBatch(typeInfo, refresh) {
    let bp;
    try {
      bp = await ReviewAPI.batchPreview(typeInfo.type_id);
    } catch (e) {
      await modal({ title: 'Batch accept', html: esc(e.message), buttons: [{ label: 'OK', value: true }] });
      return;
    }
    const s = bp.summary;
    if (!s.items) {
      await modal({ title: 'Batch accept', html: `<p>No high-confidence items to accept (${s.skipped} need individual review).</p>`,
        buttons: [{ label: 'OK', value: true }] });
      return;
    }
    const files = bp.items.flatMap(i => i.delete_files);
    const ok = await modal({
      title: `Accept ${s.items} high-confidence item${s.items === 1 ? '' : 's'}?`,
      html: `<ul>
          <li>${s.field_changes} field change(s)</li>
          <li>${s.files_deleted} file(s) deleted, ${SS.formatSize(s.bytes_reclaimed)} reclaimed</li>
          <li>${s.skipped} item(s) left for individual review</li>
        </ul>
        ${files.length ? `<p>Files deleted:</p>${fileListHtml(files)}<p>${destinationText(bp.trash_path, files.length)}</p>` : ''}`,
      buttons: [
        { label: 'Cancel', value: false },
        { label: files.length ? `Accept and delete ${files.length} file(s)` : 'Accept all', value: true,
          className: files.length ? 'ss-btn-danger' : 'ss-btn-primary' },
      ],
    });
    if (ok !== true) return;
    try {
      const out = await ReviewAPI.batchAccept(typeInfo.type_id, bp.items.map(i => ({
        rec_id: i.rec_id, plan_token: i.plan_token, confirm_delete: i.delete_files.map(f => f.file_id),
      })));
      const problems = out.results.filter(r => r.status !== 'applied');
      await modal({
        title: `Applied ${out.applied} of ${out.results.length}`,
        html: problems.length ? `<p>Not applied:</p><ul>${problems.map(r => `<li>#${esc(r.rec_id)}: ${esc(r.message || r.status)}</li>`).join('')}</ul>` : '<p>All applied.</p>',
        buttons: [{ label: 'OK', value: true }],
      });
    } catch (e) {
      await modal({ title: 'Batch refused', html: esc(e.message), buttons: [{ label: 'OK', value: true }] });
    }
    refresh();
  }

  async function renderTypeList(container, typeInfo, onBack) {
    container.innerHTML = `
      <div class="ss-list-header">
        <button class="ss-btn ss-btn-back" data-act="back">← Back</button>
        <h1>${esc(typeInfo.display_name)}</h1>
      </div>
      <p class="ss-review-desc">${esc(typeInfo.description)}</p>
      <div class="ss-list-actions">
        <button class="ss-btn ss-btn-primary" data-act="batch">Accept all high-confidence…</button>
      </div>
      <div class="ss-review-cards"></div>
      <div class="ss-review-more"></div>`;
    container.querySelector('[data-act="back"]').addEventListener('click', onBack);
    const refresh = () => renderTypeList(container, typeInfo, onBack);
    container.querySelector('[data-act="batch"]').addEventListener('click', () => runBatch(typeInfo, refresh));

    const cards = container.querySelector('.ss-review-cards');
    const pageSize = 10;
    let offset = 0;
    async function loadPage() {
      const r = await listRecs(typeInfo.type_id, 'pending', pageSize, offset);
      const recs = r.recommendations || [];
      if (!recs.length && offset === 0) {
        cards.innerHTML = '<div class="ss-empty-state">Nothing to review.</div>';
      }
      for (const rec of recs) {
        const el = SS.createElement('div');
        cards.appendChild(el);
        renderCard(el, rec.id, null);
      }
      offset += recs.length;
      const more = container.querySelector('.ss-review-more');
      more.innerHTML = '';
      if (offset < (r.total || 0)) {
        const btn = SS.createElement('button', { className: 'ss-btn ss-btn-secondary', textContent: `Load more (${r.total - offset} left)` });
        btn.addEventListener('click', loadPage);
        more.appendChild(btn);
      }
    }
    loadPage().catch(e => { cards.innerHTML = `<div class="ss-review-error">${esc(e.message)}</div>`; });
  }

  // ==================== Apply log ====================

  function entrySummary(e) {
    const r = e.result || {};
    const bits = [];
    if (e.action === 'revert') bits.push(`revert of #${e.reverts_id}`);
    if (r.fields_changed) bits.push(`${r.fields_changed} field(s)`);
    if (r.files_deleted) bits.push(`${r.files_deleted} file(s) deleted, ${SS.formatSize(r.bytes_reclaimed || 0)}`);
    if (e.status === 'failed') bits.push(`failed: ${e.error}`);
    return bits.join(' · ') || 'no change';
  }

  function priorStateHtml(e) {
    const ps = e.prior_state || {};
    const fields = (ps.fields || []).map(f => `<tr><td>${esc(f.entity_type)} #${esc(f.entity_id)}</td><td>${esc(f.field)}</td><td>${fmtValue(f.value)}</td></tr>`).join('');
    const files = (ps.deleted_files || []).map(f => `<tr><td class="ss-review-path">${esc(f.path)}</td><td>${SS.formatSize(f.size)}</td>
      <td>${Object.entries(f.fingerprints || {}).map(([k, v]) => `${esc(k)}: ${esc(v)}`).join('<br>')}</td></tr>`).join('');
    return `
      ${fields ? `<h5>Values before</h5><table class="ss-review-changes"><tbody>${fields}</tbody></table>` : ''}
      ${files ? `<h5>Deleted files${e.trash_path ? ` (to trash ${esc(e.trash_path)})` : ' (permanent)'}</h5><table class="ss-review-files"><tbody>${files}</tbody></table>` : ''}`;
  }

  async function renderApplyLog(container, onBack, offset = 0) {
    const limit = 50;
    container.innerHTML = `
      <div class="ss-list-header">
        <button class="ss-btn ss-btn-back" data-act="back">← Back</button>
        <h1>Apply log</h1>
      </div>
      <p class="ss-review-desc">Every change accepted through review, with the state before it. Metadata changes can be reverted; deleted files are recorded by path, size and hash.</p>
      <div class="ss-review-log"><div class="ss-loading-inline"><div class="ss-spinner"></div></div></div>`;
    container.querySelector('[data-act="back"]').addEventListener('click', onBack);
    const box = container.querySelector('.ss-review-log');
    let data;
    try {
      data = await ReviewAPI.applyLog({ limit, offset });
    } catch (e) {
      box.innerHTML = `<div class="ss-review-error">${esc(e.message)}</div>`;
      return;
    }
    if (!data.entries.length) {
      box.innerHTML = '<div class="ss-empty-state">Nothing applied yet.</div>';
      return;
    }
    box.innerHTML = '';
    for (const e of data.entries) {
      // A failed apply can still have written fields before it stopped; those revert too.
      const wroteFields = e.status === 'applied'
        ? (e.plan.field_changes || []).length > 0
        : e.status === 'failed' && ((e.result || {}).steps || []).some(st => st.step === 'update_scene');
      const revertible = e.action === 'apply' && !e.reverted_by && wroteFields;
      const row = SS.createElement('details', { className: `ss-review-log-entry ss-review-log-${e.status}` });
      row.innerHTML = `
        <summary>
          <span class="ss-review-log-id">#${esc(e.id)}</span>
          <span>${esc(SS.formatDate(String(e.created_at).replace(' ', 'T') + 'Z'))}</span>
          <span>${esc(e.rec_type)}</span>
          <span>${esc(e.target_type || '')} ${e.target_type === 'scene' ? `<a href="/scenes/${esc(e.target_id)}" target="_blank">#${esc(e.target_id)}</a>` : esc(e.target_id || '')}</span>
          <span class="ss-review-log-sum">${esc(entrySummary(e))}</span>
          ${e.batch_id ? `<span class="ss-review-badge">batch ${esc(e.batch_id)}</span>` : ''}
          ${e.reverted_by ? `<span class="ss-review-badge">reverted by #${esc(e.reverted_by)}</span>` : ''}
        </summary>
        <div class="ss-review-log-detail">${priorStateHtml(e)}
          ${revertible ? '<button class="ss-btn ss-btn-secondary ss-btn-sm" data-act="revert">Revert metadata</button>' : ''}
        </div>`;
      const btn = row.querySelector('[data-act="revert"]');
      if (btn) btn.addEventListener('click', () => doRevert(e, () => renderApplyLog(container, onBack, offset)));
      box.appendChild(row);
    }
    if (data.total > offset + limit || offset > 0) {
      const nav = SS.createElement('div', { className: 'ss-review-pager' });
      if (offset > 0) {
        const prev = SS.createElement('button', { className: 'ss-btn ss-btn-secondary', textContent: '← Newer' });
        prev.addEventListener('click', () => renderApplyLog(container, onBack, Math.max(0, offset - limit)));
        nav.appendChild(prev);
      }
      if (data.total > offset + limit) {
        const next = SS.createElement('button', { className: 'ss-btn ss-btn-secondary', textContent: 'Older →' });
        next.addEventListener('click', () => renderApplyLog(container, onBack, offset + limit));
        nav.appendChild(next);
      }
      box.appendChild(nav);
    }
  }

  async function doRevert(entry, refresh, force = false) {
    if (!force) {
      const files = (entry.prior_state.deleted_files || []);
      const ok = await modal({
        title: `Revert apply #${entry.id}?`,
        html: `<p>Writes the "before" values back and reopens the recommendation.</p>
          ${files.length ? `<p>${files.length} deleted file(s) are <strong>not</strong> restored by a revert.</p>` : ''}`,
        buttons: [{ label: 'Cancel', value: false }, { label: 'Revert', value: true, className: 'ss-btn-primary' }],
      });
      if (ok !== true) return;
    }
    try {
      await ReviewAPI.revert(entry.id, force);
    } catch (e) {
      if (e.detail && e.detail.code === 'drifted') {
        const rows = e.detail.fields.map(f => `<tr><td>${esc(f.field)}</td><td>applied ${fmtValue(f.applied)}</td><td>now ${fmtValue(f.current)}</td></tr>`).join('');
        const again = await modal({
          title: 'Changed since this apply',
          html: `<p>These fields were edited after the apply. Reverting overwrites those edits.</p><table class="ss-review-changes"><tbody>${rows}</tbody></table>`,
          buttons: [{ label: 'Cancel', value: false }, { label: 'Revert anyway', value: true, className: 'ss-btn-danger' }],
        });
        if (again === true) return doRevert(entry, refresh, true);
        return;
      }
      await modal({ title: 'Revert failed', html: esc(e.message), buttons: [{ label: 'OK', value: true }] });
      return;
    }
    refresh();
  }

  window.StashSenseReview = { API: ReviewAPI, renderCard, renderTypeList, renderApplyLog };
  console.log(`[${SS.PLUGIN_NAME}] Review module loaded`);
})();
