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

  function evidenceHtml(ev) {
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
    return `
      ${rows.length ? `<table class="ss-review-changes"><tbody>${rows.join('')}</tbody></table>` : ''}
      ${plan.keep_files.length ? `<div class="ss-review-keep"><strong>Keep</strong>${fileListHtml(plan.keep_files)}</div>` : ''}
      ${plan.delete_files.length ? `<div class="ss-review-delete"><strong>Delete</strong>${fileListHtml(plan.delete_files)}
        <div class="ss-review-reclaim">Reclaims ${SS.formatSize(plan.bytes_reclaimed)}</div></div>` : ''}
      ${!rows.length && !plan.delete_files.length ? '<p class="ss-review-noop">No change: accepting only resolves it.</p>' : ''}
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

  async function loadMedia(card, ev) {
    const strip = card.querySelector('.ss-review-media');
    if (!strip) return;
    for (const s of ev.stills || []) {
      const fig = SS.createElement('figure', { className: 'ss-review-still' });
      fig.innerHTML = `<div class="ss-review-still-ph">…</div><figcaption>file #${esc(s.file_id)} @ ${SS.formatDuration(s.t)}</figcaption>`;
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
        <section class="ss-review-evidence"><h4>Evidence</h4>${evidenceHtml(p.evidence)}<div class="ss-review-media"></div></section>
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
