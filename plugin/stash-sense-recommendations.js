/**
 * Stash Sense Recommendations Module
 * Dashboard UI for viewing and acting on recommendations
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  // ==================== Recommendations API ====================
  // All API calls go through the Python backend to bypass CSP restrictions

  async function apiCall(mode, params = {}) {
    const settings = await SS.getSettings();
    const result = await SS.runPluginOperation(mode, {
      sidecar_url: settings.sidecarUrl,
      ...params,
    });
    if (result.error) {
      throw new Error(result.error);
    }
    return result;
  }

  const RecommendationsAPI = {
    async getCounts() {
      return apiCall('rec_counts');
    },

    async getList(params = {}) {
      return apiCall('rec_list', {
        status: params.status,
        type: params.type,
        limit: params.limit || 100,
        offset: params.offset || 0,
      });
    },

    async getOne(id) {
      return apiCall('rec_get', { rec_id: id });
    },

    async resolve(id, action, details = null) {
      return apiCall('rec_resolve', {
        rec_id: id,
        action,
        details,
      });
    },

    async dismiss(id, reason = null) {
      return apiCall('rec_dismiss', {
        rec_id: id,
        reason,
      });
    },

    async getAnalysisTypes() {
      return apiCall('rec_analysis_types');
    },

    async runAnalysis(type) {
      return apiCall('rec_run_analysis', { analysis_type: type });
    },

    async getAnalysisRuns(type = null, limit = 10) {
      return apiCall('rec_analysis_runs', {
        analysis_type: type,
        limit,
      });
    },

    async getStashStatus() {
      return apiCall('rec_stash_status');
    },

    async getSidecarStatus() {
      const settings = await SS.getSettings();
      try {
        const health = await SS.runPluginOperation('health', {
          sidecar_url: settings.sidecarUrl,
        });
        return {
          connected: !health.error,
          url: settings.sidecarUrl,
          error: health.error || null,
          version: health.version || null,
        };
      } catch (e) {
        return {
          connected: false,
          url: settings.sidecarUrl,
          error: e.message,
        };
      }
    },

    // Actions
    async mergePerformers(destinationId, sourceIds) {
      return apiCall('rec_merge_performers', {
        destination_id: destinationId,
        source_ids: sourceIds,
      });
    },

    async deleteSceneFiles(sceneId, fileIdsToDelete, keepFileId, allFileIds) {
      return apiCall('rec_delete_files', {
        scene_id: sceneId,
        file_ids_to_delete: fileIdsToDelete,
        keep_file_id: keepFileId,
        all_file_ids: allFileIds,
      });
    },

    // Fingerprint operations
    async getFingerprintStatus() {
      return apiCall('fp_status');
    },

    async startFingerprintGeneration(options = {}) {
      // Only send params that are explicitly provided; sidecar defaults from face_config.py
      const params = { refresh_outdated: options.refreshOutdated ?? true };
      if (options.numFrames != null) params.num_frames = options.numFrames;
      if (options.minFaceSize != null) params.min_face_size = options.minFaceSize;
      if (options.maxDistance != null) params.max_distance = options.maxDistance;
      return apiCall('fp_generate', params);
    },

    async getFingerprintProgress() {
      return apiCall('fp_progress');
    },

    async stopFingerprintGeneration() {
      return apiCall('fp_stop');
    },

    // Upstream sync operations
    async updatePerformer(performerId, fields) {
      return apiCall('rec_update_performer', { performer_id: performerId, fields });
    },

    async updateTag(tagId, fields) {
      return apiCall('rec_update_tag', { tag_id: tagId, fields });
    },

    async updateStudio(studioId, fields, endpoint) {
      return apiCall('rec_update_studio', { studio_id: studioId, fields, endpoint });
    },

    async dismissUpstream(recId, reason, permanent) {
      return apiCall('rec_dismiss_upstream', { rec_id: recId, reason, permanent: !!permanent });
    },

    async getFieldConfig(endpoint) {
      return apiCall('rec_get_field_config', { endpoint });
    },

    async setFieldConfig(endpoint, fieldConfigs) {
      return apiCall('rec_set_field_config', { endpoint, field_configs: fieldConfigs });
    },

    // Database info
    async getDatabaseInfo() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('database_info', { sidecar_url: settings.sidecarUrl });
    },

    // Database update operations
    async checkUpdate() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('db_check_update', { sidecar_url: settings.sidecarUrl });
    },
    async startUpdate() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('db_update', { sidecar_url: settings.sidecarUrl });
    },
    async getUpdateStatus() {
      const settings = await SS.getSettings();
      return SS.runPluginOperation('db_update_status', { sidecar_url: settings.sidecarUrl });
    },

    // User settings
    async getUserSetting(key) {
      const result = await apiCall('user_get_setting', { key });
      return result.value;
    },

    async setUserSetting(key, value) {
      return apiCall('user_set_setting', { key, value });
    },

    async getAllUserSettings() {
      const result = await apiCall('user_get_all_settings');
      return result.settings || {};
    },
  };

  /**
   * Convert ALL_CAPS enum values to Title Case for display.
   * e.g. "BROWN" -> "Brown", "EYE_COLOR" -> "Eye Color", "NATURAL" -> "Natural"
   * Returns original value if not an ALL_CAPS string.
   */
  function normalizeEnumValue(val) {
    if (typeof val !== 'string') return val;
    // Only transform if the string is ALL_CAPS (with optional underscores)
    if (!/^[A-Z][A-Z0-9_]*$/.test(val)) return val;
    return val
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }

  // Filter out false-positive changes where values are effectively equal
  function filterRealChanges(changes) {
    return (changes || []).filter(change => {
      const local = change.local_value;
      const upstream = change.upstream_value;
      // Both null/undefined/empty string/zero
      const isEmptyVal = v => v === null || v === undefined || v === '' || v === 0;
      if (isEmptyVal(local) && isEmptyVal(upstream)) return false;
      // String case-insensitive comparison
      if (typeof local === 'string' && typeof upstream === 'string') {
        if (local.toLowerCase() === upstream.toLowerCase()) return false;
      }
      // List comparison (alias_list, urls) - case-insensitive set equality
      if (Array.isArray(local) && Array.isArray(upstream)) {
        const localSet = new Set(local.map(v => String(v).toLowerCase()));
        const upstreamSet = new Set(upstream.map(v => String(v).toLowerCase()));
        if (localSet.size === upstreamSet.size && [...localSet].every(v => upstreamSet.has(v))) return false;
      }
      // Strict equality for other types
      if (local === upstream) return false;
      return true;
    });
  }

  // ==================== State ====================

  let currentState = {
    view: 'dashboard', // dashboard, list, detail
    type: null,
    status: 'pending',
    page: 0,
    selectedRec: null,
    counts: null,
  };

  // (Polling for analysis/fingerprint progress now handled by Operations tab)

  // ==================== Dashboard Container ====================

  function createDashboardContainer() {
    const existing = document.getElementById('ss-recommendations');
    if (existing) existing.remove();

    const container = SS.createElement('div', {
      id: 'ss-recommendations',
      className: 'ss-recommendations',
    });

    // Persistent app header (stays above tabs)
    const appHeader = SS.createElement('div', {
      className: 'ss-app-header',
    });
    appHeader.innerHTML = `
      <div class="ss-app-header-left">
        <h1>Stash Sense</h1>
        <p class="ss-dashboard-subtitle">Library analysis and curation tools</p>
      </div>
      <div class="ss-app-header-right" id="ss-status-area"></div>
    `;
    container.appendChild(appHeader);

    // Content wrapper (views render inside this)
    const content = SS.createElement('div', {
      className: 'ss-dashboard-content',
    });
    container.appendChild(content);

    return container;
  }

  // ==================== Dashboard View ====================

  async function renderDashboard(mainContainer, content) {
    content.innerHTML = `
      <div class="ss-dashboard-loading">
        <div class="ss-spinner"></div>
        <p>Loading recommendations...</p>
      </div>
    `;

    try {
      const [counts, sidecarStatus, fpStatus, dbInfo, updateInfo] = await Promise.all([
        RecommendationsAPI.getCounts(),
        RecommendationsAPI.getSidecarStatus(),
        RecommendationsAPI.getFingerprintStatus(),
        RecommendationsAPI.getDatabaseInfo().catch(() => null),
        RecommendationsAPI.checkUpdate().catch(() => null),
      ]);

      currentState.counts = counts;

      // Update the persistent status area in the app header
      const statusArea = document.getElementById('ss-status-area');
      if (statusArea) {
        statusArea.className = `ss-app-header-right ${sidecarStatus.connected ? 'connected' : 'disconnected'}`;
        statusArea.innerHTML = `
          <span class="ss-status-dot"></span>
          <span class="ss-status-label">${sidecarStatus.connected ? 'Connected' : 'Disconnected'}</span>
          ${sidecarStatus.url ? `<span class="ss-status-url">${sidecarStatus.url}</span>` : ''}
          ${sidecarStatus.error ? `<span class="ss-status-error">${sidecarStatus.error}</span>` : ''}
        `;
      }

      // Build identification database stats
      const fpCoverage = fpStatus.complete_fingerprints || 0;
      const fpNeedsRefresh = fpStatus.needs_refresh_count || 0;
      const dbVersion = fpStatus.current_db_version || dbInfo?.version || 'N/A';
      const performerCount = dbInfo?.performer_count || 0;
      const faceCount = dbInfo?.face_count || 0;
      const tattooCount = dbInfo?.tattoo_embedding_count || 0;

      // Build type cards HTML
      const typeConfigs = {
        duplicate_performer: {
          title: 'Duplicate Performers',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>`,
          description: 'Performers sharing the same StashDB ID',
        },
        duplicate_scenes: {
          title: 'Duplicate Scenes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12zm-6-1l4-4-1.4-1.4-1.6 1.6V6h-2v6.2l-1.6-1.6L10 12l4 4z"/></svg>`,
          description: 'Scenes that may be duplicates based on stash-box ID, faces, or metadata',
        },
        duplicate_scene_files: {
          title: 'Duplicate Scene Files',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-8 12.5v-9l6 4.5-6 4.5z"/></svg>`,
          description: 'Scenes with multiple files attached',
        },
        upstream_performer_changes: {
          title: 'Upstream Performer Changes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 6V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/></svg>`,
          description: 'Performer fields updated on StashDB since last sync',
        },
        upstream_tag_changes: {
          title: 'Upstream Tag Changes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M21.41 11.58l-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42zM5.5 7C4.67 7 4 6.33 4 5.5S4.67 4 5.5 4 7 4.67 7 5.5 6.33 7 5.5 7z"/></svg>`,
          description: 'Tag fields updated on StashDB since last sync',
        },
        upstream_studio_changes: {
          title: 'Upstream Studio Changes',
          icon: `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z"/></svg>`,
          description: 'Studio fields updated on StashDB since last sync',
        },
      };

      content.innerHTML = `
        <div class="ss-id-database-section">
          <h2>Identification Database <span id="ss-info-fp"></span></h2>
          <p class="ss-id-database-desc">Face recognition database used for performer identification and duplicate detection.</p>

          <div class="ss-id-database-stats">
            <div class="ss-db-stat">
              <span class="ss-db-stat-value">${dbVersion}</span>
              <span class="ss-db-stat-label">Version</span>
              ${updateInfo && updateInfo.update_available ? `
                <div class="ss-update-badge" id="ss-update-badge">
                  <span class="ss-update-badge-text">v${updateInfo.latest_version} available</span>
                </div>
              ` : ''}
            </div>
            <div class="ss-db-stat">
              <span class="ss-db-stat-value">${performerCount.toLocaleString()}</span>
              <span class="ss-db-stat-label">Performers</span>
            </div>
            <div class="ss-db-stat">
              <span class="ss-db-stat-value">${faceCount.toLocaleString()}</span>
              <span class="ss-db-stat-label">Faces</span>
            </div>
            ${tattooCount > 0 ? `
            <div class="ss-db-stat">
              <span class="ss-db-stat-value">${tattooCount.toLocaleString()}</span>
              <span class="ss-db-stat-label">Tattoos</span>
            </div>
            ` : ''}
            <div class="ss-db-stat">
              <span class="ss-db-stat-value">${fpCoverage.toLocaleString()}</span>
              <span class="ss-db-stat-label">Fingerprints</span>
            </div>
            ${fpNeedsRefresh > 0 ? `
            <div class="ss-db-stat ss-db-stat-warning">
              <span class="ss-db-stat-value">${fpNeedsRefresh.toLocaleString()}</span>
              <span class="ss-db-stat-label">Need Refresh</span>
            </div>
            ` : ''}
          </div>
        </div>

        <div class="ss-dashboard-types">
          <div class="ss-section-header">
            <h2>Recommendations</h2>
            <span class="ss-count-badge">${counts.total_pending}</span>
            <span id="ss-info-types"></span>
          </div>
          <div class="ss-type-cards"></div>
        </div>
      `;

      // Add info icons
      const fpInfoSlot = content.querySelector('#ss-info-fp');
      if (fpInfoSlot) fpInfoSlot.appendChild(createInfoIcon(() => showHelpModal('Identification Database', HELP_FINGERPRINTS)));
      const typesInfoSlot = content.querySelector('#ss-info-types');
      if (typesInfoSlot) typesInfoSlot.appendChild(createInfoIcon(() => showHelpModal('Recommendations', HELP_REC_TYPES)));

      // Render type cards
      const typeCards = content.querySelector('.ss-type-cards');

      // Ensure all types are shown, even if no counts yet
      const allTypes = Object.keys(typeConfigs);
      for (const type of allTypes) {
        const config = typeConfigs[type];
        const typeCounts = counts.counts?.[type] || {};
        const pending = typeCounts.pending || 0;
        const resolved = typeCounts.resolved || 0;
        const dismissed = typeCounts.dismissed || 0;

        const card = SS.createElement('div', {
          className: 'ss-type-card',
          innerHTML: `
            <div class="ss-type-card-header">
              <span class="ss-type-icon">${config.icon}</span>
              <div class="ss-type-title-block">
                <h3>${config.title}</h3>
                <p>${config.description}</p>
              </div>
            </div>
            <div class="ss-type-card-footer">
              <div class="ss-type-counts">
                <div class="ss-count-item ss-count-pending">
                  <span class="ss-count-number">${pending}</span>
                  <span class="ss-count-label">pending</span>
                </div>
                <div class="ss-count-item ss-count-resolved">
                  <span class="ss-count-number">${resolved}</span>
                  <span class="ss-count-label">resolved</span>
                </div>
                <div class="ss-count-item ss-count-dismissed">
                  <span class="ss-count-number">${dismissed}</span>
                  <span class="ss-count-label">dismissed</span>
                </div>
              </div>
              <button class="ss-btn ss-btn-secondary ss-btn-sm" data-type="${type}">
                View All
              </button>
            </div>
          `,
        });

        card.querySelector('button').addEventListener('click', () => {
          currentState.type = type;
          currentState.view = 'list';
          renderCurrentView(mainContainer);
        });

        typeCards.appendChild(card);
      }

    } catch (e) {
      // Update status to show disconnected
      const statusArea = document.getElementById('ss-status-area');
      if (statusArea) {
        statusArea.className = 'ss-app-header-right disconnected';
        statusArea.innerHTML = `
          <span class="ss-status-dot"></span>
          <span class="ss-status-label">Disconnected</span>
        `;
      }

      content.innerHTML = `
        <div class="ss-error-state">
          <div class="ss-error-icon">
            <svg viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>
          </div>
          <h2>Connection Error</h2>
          <p>${e.message}</p>
          <p class="ss-error-hint">Make sure the Stash Sense sidecar is running and configured correctly.</p>
          <button class="ss-btn ss-btn-primary" id="ss-retry-btn">Retry</button>
        </div>
      `;
      const retryBtn = content.querySelector('#ss-retry-btn');
      if (retryBtn) retryBtn.addEventListener('click', () => location.reload());
    }
  }

  // ==================== Batch Accept All ====================

  function computeBatchChanges(recommendations) {
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = upstreamVal === null || upstreamVal === undefined || upstreamVal === '';
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    const results = [];

    for (const rec of recommendations) {
      const details = rec.details;
      const rawChanges = details.changes || [];
      const changes = filterRealChanges(rawChanges);
      if (changes.length === 0) continue;

      const isTag = rec.type === 'upstream_tag_changes';
      const isStudio = rec.type === 'upstream_studio_changes';
      const entityName = isStudio ? details.studio_name : (isTag ? details.tag_name : details.performer_name);
      const entityType = isStudio ? 'Studio' : (isTag ? 'Tag' : 'Performer');
      const entityId = isStudio ? details.studio_id : (isTag ? details.tag_id : details.performer_id);

      const fields = {};
      const fieldSummaries = [];

      for (const change of changes) {
        const mergeType = change.merge_type || 'simple';
        const fieldKey = change.field;

        if (mergeType === 'alias_list') {
          const allAliases = new Set();
          if (Array.isArray(change.local_value)) change.local_value.forEach(a => allAliases.add(a));
          if (Array.isArray(change.upstream_value)) change.upstream_value.forEach(a => allAliases.add(a));
          const merged = [...allAliases];
          fields[fieldKey] = merged;
          const localCount = (change.local_value || []).length;
          const newCount = merged.length - localCount;
          if (newCount > 0) {
            fieldSummaries.push({ field: change.field_label || fieldKey, desc: `+${newCount} aliases merged` });
          }
        } else {
          const choice = smartDefault(change.local_value, change.upstream_value);
          const resultVal = choice === 'upstream' ? change.upstream_value : change.local_value;
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');
          const resultStr = formatFieldValue(resultVal) === '(empty)' ? '' : String(resultVal || '');

          if (resultStr === localStr) {
            if (mergeType === 'name' && choice !== 'upstream') {
              if (change.upstream_value) {
                fields['_alias_add'] = fields['_alias_add'] || [];
                fields['_alias_add'].push(String(change.upstream_value));
                fieldSummaries.push({ field: 'Alias', desc: `+ "${change.upstream_value}"` });
              }
            }
            continue;
          }

          fields[fieldKey] = resultStr;

          if (mergeType === 'name' && choice === 'upstream') {
            if (change.local_value) {
              fields['_alias_add'] = fields['_alias_add'] || [];
              fields['_alias_add'].push(String(change.local_value));
            }
          }

          const fromDisplay = formatFieldValue(change.local_value);
          const toDisplay = formatFieldValue(resultVal);
          fieldSummaries.push({
            field: change.field_label || fieldKey,
            desc: `${fromDisplay} \u2192 ${toDisplay}`,
          });
        }
      }

      if (fieldSummaries.length > 0) {
        results.push({ rec, entityName, entityType, entityId, fields, changes: fieldSummaries });
      }
    }

    return results;
  }

  function showAcceptAllModal(batchChanges) {
    return new Promise((resolve) => {
      const totalChanges = batchChanges.reduce((sum, item) => sum + item.changes.length, 0);

      const overlay = document.createElement('div');
      overlay.className = 'ss-modal-overlay';
      overlay.innerHTML = `
        <div class="ss-accept-all-modal">
          <div class="ss-modal-header">
            <h3>Accept All Changes</h3>
            <button class="ss-modal-close">&times;</button>
          </div>
          <div class="ss-modal-body">
            <p>This will apply smart defaults to <strong>${batchChanges.length}</strong> ${batchChanges.length === 1 ? 'entity' : 'entities'} (${totalChanges} field ${totalChanges === 1 ? 'change' : 'changes'}). Upstream values are preferred when available; alias lists are merged.</p>
            ${batchChanges.map(item => `
              <div class="ss-batch-entity-group">
                <span class="ss-batch-entity-name">${escapeHtml(item.entityName)}</span>
                <span class="ss-batch-entity-type">${escapeHtml(item.entityType)}</span>
                <ul class="ss-batch-changes-list">
                  ${item.changes.map(c => `<li><span class="ss-batch-field-name">${escapeHtml(c.field)}:</span> ${escapeHtml(c.desc)}</li>`).join('')}
                </ul>
              </div>
            `).join('')}
          </div>
          <div class="ss-modal-footer">
            <button class="ss-btn ss-btn-secondary" id="ss-modal-cancel">Cancel</button>
            <button class="ss-accept-all-btn" id="ss-modal-confirm">Accept ${batchChanges.length} ${batchChanges.length === 1 ? 'Change' : 'Changes'}</button>
          </div>
        </div>
      `;

      document.body.appendChild(overlay);

      function close(result) {
        if (!result) overlay.remove();
        resolve(result);
      }

      overlay.querySelector('.ss-modal-close').addEventListener('click', () => close(false));
      overlay.querySelector('#ss-modal-cancel').addEventListener('click', () => close(false));
      overlay.querySelector('#ss-modal-confirm').addEventListener('click', () => close(true));
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) close(false);
      });
    });
  }

  async function processBatchChanges(batchChanges, modalOverlay) {
    const modal = modalOverlay.querySelector('.ss-accept-all-modal');
    const body = modal.querySelector('.ss-modal-body');
    const footer = modal.querySelector('.ss-modal-footer');

    footer.innerHTML = '';
    body.innerHTML = `
      <div class="ss-batch-progress">
        <div class="ss-batch-progress-bar">
          <div class="ss-batch-progress-fill" style="width: 0%"></div>
        </div>
        <div class="ss-batch-progress-text">Processing 0 / ${batchChanges.length}...</div>
      </div>
    `;

    const progressFill = body.querySelector('.ss-batch-progress-fill');
    const progressText = body.querySelector('.ss-batch-progress-text');

    let succeeded = 0;
    const failed = [];

    for (let i = 0; i < batchChanges.length; i++) {
      const item = batchChanges[i];
      const pct = Math.round(((i + 1) / batchChanges.length) * 100);
      progressText.textContent = `Processing ${i + 1} / ${batchChanges.length} — ${item.entityName}...`;
      progressFill.style.width = `${pct}%`;

      try {
        if (item.entityType === 'Tag') {
          await RecommendationsAPI.updateTag(item.entityId, item.fields);
        } else if (item.entityType === 'Studio') {
          await RecommendationsAPI.updateStudio(item.entityId, item.fields, item.rec.details.endpoint);
        } else {
          await RecommendationsAPI.updatePerformer(item.entityId, item.fields);
        }
        await RecommendationsAPI.resolve(item.rec.id, 'accepted', { batch: true });
        succeeded++;
      } catch (e) {
        failed.push({ entityName: item.entityName, error: e.message });
      }
    }

    progressFill.style.width = '100%';
    if (failed.length === 0) {
      progressText.textContent = `Done! ${succeeded} ${succeeded === 1 ? 'change' : 'changes'} applied successfully.`;
    } else {
      progressText.textContent = `${succeeded} applied, ${failed.length} failed.`;
    }

    return { succeeded, failed };
  }

  // ==================== List View ====================

  async function renderList(container) {
    const typeConfigs = {
      duplicate_performer: 'Duplicate Performers',
      duplicate_scenes: 'Duplicate Scenes',
      duplicate_scene_files: 'Duplicate Scene Files',
      upstream_performer_changes: 'Upstream Performer Changes',
      upstream_tag_changes: 'Upstream Tag Changes',
      upstream_studio_changes: 'Upstream Studio Changes',
    };

    container.innerHTML = `
      <div class="ss-list-header">
        <button class="ss-btn ss-btn-back" id="ss-back-btn">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
          </svg>
          Back
        </button>
        <h1>${typeConfigs[currentState.type] || currentState.type}</h1>
      </div>

      <div class="ss-list-filters">
        <div class="ss-filter-tabs">
          <button class="ss-filter-tab ${currentState.status === 'pending' ? 'active' : ''}" data-status="pending">
            Pending
          </button>
          <button class="ss-filter-tab ${currentState.status === 'resolved' ? 'active' : ''}" data-status="resolved">
            Resolved
          </button>
          <button class="ss-filter-tab ${currentState.status === 'dismissed' ? 'active' : ''}" data-status="dismissed">
            Dismissed
          </button>
        </div>
        ${currentState.status === 'pending' && (currentState.type === 'upstream_performer_changes' || currentState.type === 'upstream_tag_changes' || currentState.type === 'upstream_studio_changes')
          ? '<button class="ss-accept-all-btn" id="ss-accept-all-btn">Accept All Changes</button>'
          : ''
        }
      </div>

      <div class="ss-list-content">
        <div class="ss-loading-inline">
          <div class="ss-spinner"></div>
        </div>
      </div>
    `;

    // Back button
    container.querySelector('#ss-back-btn').addEventListener('click', () => {
      currentState.view = 'dashboard';
      currentState.type = null;
      renderCurrentView(container);
    });

    // Filter tabs
    container.querySelectorAll('.ss-filter-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        currentState.status = tab.dataset.status;
        currentState.page = 0;
        renderCurrentView(container);
      });
    });

    // Accept All Changes button
    const acceptAllBtn = container.querySelector('#ss-accept-all-btn');
    if (acceptAllBtn) {
      acceptAllBtn.addEventListener('click', async () => {
        acceptAllBtn.disabled = true;
        acceptAllBtn.textContent = 'Loading...';

        try {
          // Fetch ALL pending (high limit to get everything)
          const allPending = await RecommendationsAPI.getList({
            type: currentState.type,
            status: 'pending',
            limit: 10000,
            offset: 0,
          });

          const batchChanges = computeBatchChanges(allPending.recommendations);

          if (batchChanges.length === 0) {
            acceptAllBtn.textContent = 'No changes to apply';
            setTimeout(() => { acceptAllBtn.textContent = 'Accept All Changes'; acceptAllBtn.disabled = false; }, 2000);
            return;
          }

          const confirmed = await showAcceptAllModal(batchChanges);
          if (!confirmed) {
            acceptAllBtn.textContent = 'Accept All Changes';
            acceptAllBtn.disabled = false;
            return;
          }

          const overlay = document.querySelector('.ss-modal-overlay');
          const result = await processBatchChanges(batchChanges, overlay);

          overlay.remove();

          if (result.failed.length === 0) {
            acceptAllBtn.textContent = `Done! ${result.succeeded} applied`;
            acceptAllBtn.classList.add('ss-btn-success');
          } else {
            acceptAllBtn.textContent = `${result.succeeded} applied, ${result.failed.length} failed`;
            acceptAllBtn.classList.add('ss-btn-error');
          }

          setTimeout(() => {
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 2000);
        } catch (e) {
          acceptAllBtn.textContent = `Error: ${e.message}`;
          acceptAllBtn.disabled = false;
        }
      });
    }

    // Load recommendations and counts in parallel
    try {
      const PAGE_SIZE = 25;
      const [result, countsResult] = await Promise.all([
        RecommendationsAPI.getList({
          type: currentState.type,
          status: currentState.status,
          limit: PAGE_SIZE,
          offset: currentState.page * PAGE_SIZE,
        }),
        RecommendationsAPI.getCounts(),
      ]);

      // Update tab labels with counts
      const typeCounts = countsResult.counts?.[currentState.type] || {};
      container.querySelectorAll('.ss-filter-tab').forEach(tab => {
        const status = tab.dataset.status;
        const count = typeCounts[status] || 0;
        const label = status.charAt(0).toUpperCase() + status.slice(1);
        tab.textContent = `${label} (${count})`;
      });

      const listContent = container.querySelector('.ss-list-content');

      if (result.recommendations.length === 0) {
        listContent.innerHTML = `
          <div class="ss-empty-state">
            <p>No ${currentState.status} recommendations found.</p>
          </div>
        `;
        return;
      }

      listContent.innerHTML = '';

      for (const rec of result.recommendations) {
        const card = renderRecommendationCard(rec);
        card.addEventListener('click', () => {
          currentState.selectedRec = rec;
          currentState.view = 'detail';
          renderCurrentView(container);
        });
        listContent.appendChild(card);
      }

      // Pagination controls
      const totalPages = Math.ceil(result.total / PAGE_SIZE);
      if (totalPages > 1) {
        const pagination = document.createElement('div');
        pagination.style.cssText = 'display:flex;align-items:center;justify-content:center;gap:12px;padding:16px 0;';

        const prevBtn = document.createElement('button');
        prevBtn.className = 'ss-btn ss-btn-secondary';
        prevBtn.textContent = '\u2190 Prev';
        prevBtn.disabled = currentState.page === 0;
        prevBtn.style.cssText = 'min-width:80px;';
        prevBtn.addEventListener('click', () => {
          currentState.page--;
          renderCurrentView(container);
        });

        const pageText = document.createElement('span');
        pageText.style.cssText = 'color:var(--ss-text-secondary, #aaa);font-size:14px;';
        pageText.textContent = `Page ${currentState.page + 1} of ${totalPages}`;

        const nextBtn = document.createElement('button');
        nextBtn.className = 'ss-btn ss-btn-secondary';
        nextBtn.textContent = 'Next \u2192';
        nextBtn.disabled = currentState.page >= totalPages - 1;
        nextBtn.style.cssText = 'min-width:80px;';
        nextBtn.addEventListener('click', () => {
          currentState.page++;
          renderCurrentView(container);
        });

        pagination.appendChild(prevBtn);
        pagination.appendChild(pageText);
        pagination.appendChild(nextBtn);
        listContent.appendChild(pagination);
      }

    } catch (e) {
      container.querySelector('.ss-list-content').innerHTML = `
        <div class="ss-error-state">
          <p>Failed to load recommendations: ${e.message}</p>
        </div>
      `;
    }
  }

  function renderRecommendationCard(rec) {
    const details = rec.details;

    if (rec.type === 'duplicate_performer') {
      const performers = details.performers || [];
      const keeper = performers.find(p => p.is_suggested_keeper);
      const others = performers.filter(p => !p.is_suggested_keeper);

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-performer',
        innerHTML: `
          <div class="ss-rec-performers">
            ${performers.map(p => `
              <div class="ss-rec-performer-thumb ${p.is_suggested_keeper ? 'keeper' : ''}">
                ${p.image_path ? `<img src="${p.image_path}" alt="${p.name}" loading="lazy" onerror="this.style.display='none'" />` : ''}
                <span class="ss-performer-name">${p.name}</span>
                <span class="ss-performer-count">${p.scene_count} scenes</span>
              </div>
            `).join('')}
          </div>
          <div class="ss-rec-summary">
            <span class="ss-rec-type-badge">Duplicate</span>
            <span>${performers.length} performers share StashDB ID</span>
          </div>
        `,
      });
    }

    if (rec.type === 'duplicate_scene_files') {
      const files = details.files || [];

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-scene-files',
        innerHTML: `
          <div class="ss-rec-scene-info">
            <h4>${details.scene_title}</h4>
            <div class="ss-rec-scene-meta">
              ${details.studio?.name ? `<span>${details.studio.name}</span>` : ''}
              ${details.performers?.length ? `<span>${details.performers.map(p => p.name).join(', ')}</span>` : ''}
            </div>
          </div>
          <div class="ss-rec-files-summary">
            <span class="ss-rec-type-badge">${files.length} files</span>
            <span class="ss-rec-savings">Save ${details.potential_savings_formatted}</span>
          </div>
        `,
      });
    }

    if (rec.type === 'upstream_performer_changes') {
      const realChanges = filterRealChanges(details.changes);
      const changeCount = realChanges.length;
      const changedFields = realChanges.map(c => c.field_label).join(', ');

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-upstream',
        innerHTML: `
          <div class="ss-rec-card-header">
            <img src="${details.performer_image_path || ''}" class="ss-rec-thumb" onerror="this.style.display='none'"/>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">Upstream Changes: ${details.performer_name || 'Unknown'}</div>
              <div class="ss-rec-card-subtitle">
                ${changeCount} field${changeCount !== 1 ? 's' : ''} changed · ${details.endpoint_name || ''}
              </div>
              <div class="ss-rec-card-fields">${changedFields}</div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'upstream_tag_changes') {
      const realChanges = filterRealChanges(details.changes);
      const changeCount = realChanges.length;
      const changedFields = realChanges.map(c => c.field_label).join(', ');

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-upstream',
        innerHTML: `
          <div class="ss-rec-card-header">
            <div class="ss-rec-tag-icon">
              <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M21.41 11.58l-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42zM5.5 7C4.67 7 4 6.33 4 5.5S4.67 4 5.5 4 7 4.67 7 5.5 6.33 7 5.5 7z"/></svg>
            </div>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">Upstream Changes: ${details.tag_name || 'Unknown'}</div>
              <div class="ss-rec-card-subtitle">
                ${changeCount} field${changeCount !== 1 ? 's' : ''} changed · ${details.endpoint_name || ''}
              </div>
              <div class="ss-rec-card-fields">${changedFields}</div>
            </div>
          </div>
        `,
      });
    }

    if (rec.type === 'upstream_studio_changes') {
      const realChanges = filterRealChanges(details.changes);
      const changeCount = realChanges.length;
      const changedFields = realChanges.map(c => c.field_label).join(', ');

      return SS.createElement('div', {
        className: 'ss-rec-card ss-rec-upstream',
        innerHTML: `
          <div class="ss-rec-card-header">
            <div class="ss-rec-tag-icon">
              <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z"/></svg>
            </div>
            <div class="ss-rec-card-info">
              <div class="ss-rec-card-title">Upstream Changes: ${details.studio_name || 'Unknown'}</div>
              <div class="ss-rec-card-subtitle">
                ${changeCount} field${changeCount !== 1 ? 's' : ''} changed · ${details.endpoint_name || ''}
              </div>
              <div class="ss-rec-card-fields">${changedFields}</div>
            </div>
          </div>
        `,
      });
    }

    // Fallback for unknown types
    return SS.createElement('div', {
      className: 'ss-rec-card',
      innerHTML: `
        <div class="ss-rec-generic">
          <span class="ss-rec-type-badge">${rec.type}</span>
          <span>Target: ${rec.target_type} ${rec.target_id}</span>
        </div>
      `,
    });
  }

  // ==================== Detail View ====================

  async function renderDetail(container) {
    const rec = currentState.selectedRec;
    if (!rec) {
      currentState.view = 'list';
      renderCurrentView(container);
      return;
    }

    container.innerHTML = `
      <div class="ss-detail-header">
        <button class="ss-btn ss-btn-back" id="ss-back-btn">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
          </svg>
          Back
        </button>
      </div>
      <div class="ss-detail-content"></div>
    `;

    container.querySelector('#ss-back-btn').addEventListener('click', () => {
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(container);
    });

    const content = container.querySelector('.ss-detail-content');

    if (rec.type === 'duplicate_performer') {
      renderDuplicatePerformerDetail(content, rec);
    } else if (rec.type === 'duplicate_scene_files') {
      renderDuplicateSceneFilesDetail(content, rec);
    } else if (rec.type === 'upstream_performer_changes') {
      await renderUpstreamPerformerDetail(content, rec);
    } else if (rec.type === 'upstream_tag_changes') {
      await renderUpstreamTagDetail(content, rec);
    } else if (rec.type === 'upstream_studio_changes') {
      await renderUpstreamStudioDetail(content, rec);
    } else {
      content.innerHTML = `<p>Unknown recommendation type: ${rec.type}</p>`;
    }
  }

  function renderDuplicatePerformerDetail(container, rec) {
    const details = rec.details;
    const performers = details.performers || [];

    container.innerHTML = `
      <div class="ss-detail-duplicate-performer">
        <h2>Duplicate Performers <span id="ss-info-dup-perf"></span></h2>
        <p class="ss-detail-subtitle">
          These performers share StashDB ID:
          <a href="https://stashdb.org/performers/${details.stash_id}" target="_blank" rel="noopener">
            ${details.stash_id.substring(0, 8)}...
          </a>
        </p>

        <div class="ss-performer-grid">
          ${performers.map(p => `
            <div class="ss-performer-option ${p.is_suggested_keeper ? 'suggested' : ''}" data-id="${p.id}">
              <div class="ss-performer-image">
                ${p.image_path ? `<img src="${p.image_path}" alt="${p.name}" loading="lazy" onerror="this.style.display='none'" />` : '<div class="ss-no-image">No Image</div>'}
                ${p.is_suggested_keeper ? '<span class="ss-suggested-badge">Suggested Keeper</span>' : ''}
              </div>
              <div class="ss-performer-details">
                <h3>
                  <a href="/performers/${p.id}" target="_blank">${p.name}</a>
                </h3>
                <ul class="ss-performer-stats">
                  <li><strong>${p.scene_count}</strong> scenes</li>
                  <li><strong>${p.image_count}</strong> images</li>
                  <li><strong>${p.gallery_count}</strong> galleries</li>
                </ul>
                <label class="ss-radio-label">
                  <input type="radio" name="keeper" value="${p.id}" ${p.is_suggested_keeper ? 'checked' : ''} />
                  Keep this performer
                </label>
              </div>
            </div>
          `).join('')}
        </div>

        <div class="ss-detail-actions">
          <button class="ss-btn ss-btn-primary ss-btn-merge" id="ss-merge-btn">
            Merge Performers
          </button>
          <button class="ss-btn ss-btn-secondary" id="ss-dismiss-btn">
            Dismiss
          </button>
        </div>
      </div>
    `;

    // Add info icon
    const dpInfoSlot = container.querySelector('#ss-info-dup-perf');
    if (dpInfoSlot) dpInfoSlot.appendChild(createInfoIcon(() => showHelpModal('Duplicate Performers', HELP_DUP_PERFORMER)));

    // Click anywhere on card to select radio (except links)
    container.querySelectorAll('.ss-performer-option').forEach(card => {
      card.style.cursor = 'pointer';
      card.addEventListener('click', (e) => {
        if (e.target.closest('a')) return;
        const radio = card.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
      });
    });

    // Merge action
    container.querySelector('#ss-merge-btn').addEventListener('click', async () => {
      const keeperId = container.querySelector('input[name="keeper"]:checked')?.value;
      if (!keeperId) {
        showConfirmModal('Please select a performer to keep.', () => {});
        return;
      }

      const sourceIds = performers.filter(p => p.id !== keeperId).map(p => p.id);
      const btn = container.querySelector('#ss-merge-btn');

      try {
        btn.disabled = true;
        btn.textContent = 'Merging...';

        await RecommendationsAPI.mergePerformers(keeperId, sourceIds);
        await RecommendationsAPI.resolve(rec.id, 'merged', { keeper_id: keeperId });

        btn.textContent = 'Merged!';
        btn.classList.add('ss-btn-success');

        setTimeout(() => {
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        }, 1500);
      } catch (e) {
        btn.textContent = `Failed: ${e.message}`;
        btn.classList.add('ss-btn-error');
        btn.disabled = false;
      }
    });

    // Dismiss action
    container.querySelector('#ss-dismiss-btn').addEventListener('click', async () => {
      const btn = container.querySelector('#ss-dismiss-btn');
      try {
        btn.disabled = true;
        btn.textContent = 'Dismissing...';
        await RecommendationsAPI.dismiss(rec.id, 'User dismissed');
        currentState.view = 'list';
        currentState.selectedRec = null;
        renderCurrentView(document.getElementById('ss-recommendations'));
      } catch (e) {
        btn.textContent = `Failed: ${e.message}`;
        btn.disabled = false;
      }
    });
  }

  function renderDuplicateSceneFilesDetail(container, rec) {
    const details = rec.details;
    const files = details.files || [];

    container.innerHTML = `
      <div class="ss-detail-scene-files">
        <h2>${details.scene_title} <span id="ss-info-dup-files"></span></h2>
        <p class="ss-detail-subtitle">
          <a href="/scenes/${rec.target_id}" target="_blank">View Scene</a>
          ${details.studio?.name ? ` | ${details.studio.name}` : ''}
        </p>

        <div class="ss-files-summary">
          <span>Total: ${details.total_size_formatted}</span>
          <span class="ss-potential-savings">Potential savings: ${details.potential_savings_formatted}</span>
        </div>

        <div class="ss-files-list">
          ${files.map((f, i) => `
            <div class="ss-file-option ${f.is_suggested_keeper ? 'suggested' : ''}" data-id="${f.id}">
              <label class="ss-radio-label">
                <input type="radio" name="keeper" value="${f.id}" ${f.is_suggested_keeper ? 'checked' : ''} />
                <div class="ss-file-info">
                  <div class="ss-file-name">${f.basename}</div>
                  <div class="ss-file-meta">
                    <span class="ss-file-resolution">${f.resolution}</span>
                    <span class="ss-file-codec">${f.video_codec}</span>
                    <span class="ss-file-size">${f.size_formatted}</span>
                    <span class="ss-file-duration">${f.duration_formatted}</span>
                  </div>
                  <div class="ss-file-path">${f.path}</div>
                </div>
              </label>
              ${f.is_suggested_keeper ? '<span class="ss-suggested-badge">Best Quality</span>' : ''}
            </div>
          `).join('')}
        </div>

        <div class="ss-detail-actions">
          <button class="ss-btn ss-btn-danger ss-btn-delete" id="ss-delete-btn">
            Delete Other Files
          </button>
          <button class="ss-btn ss-btn-secondary" id="ss-dismiss-btn">
            Dismiss
          </button>
        </div>
      </div>
    `;

    // Add info icon
    const dfInfoSlot = container.querySelector('#ss-info-dup-files');
    if (dfInfoSlot) dfInfoSlot.appendChild(createInfoIcon(() => showHelpModal('Duplicate Scene Files', HELP_DUP_SCENE_FILES)));

    // Click anywhere on file card to select radio (except links)
    container.querySelectorAll('.ss-file-option').forEach(card => {
      card.style.cursor = 'pointer';
      card.addEventListener('click', (e) => {
        if (e.target.closest('a')) return;
        const radio = card.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
      });
    });

    // Delete action
    container.querySelector('#ss-delete-btn').addEventListener('click', async () => {
      const keeperId = container.querySelector('input[name="keeper"]:checked')?.value;
      if (!keeperId) {
        showConfirmModal('Please select a file to keep.', () => {});
        return;
      }

      const fileIdsToDelete = files.filter(f => f.id !== keeperId).map(f => f.id);
      const allFileIds = files.map(f => f.id);

      showConfirmModal(
        `Delete ${fileIdsToDelete.length} file(s)? This cannot be undone.`,
        async () => {
          const btn = container.querySelector('#ss-delete-btn');

          try {
            btn.disabled = true;
            btn.textContent = 'Deleting...';

            await RecommendationsAPI.deleteSceneFiles(
              rec.target_id,
              fileIdsToDelete,
              keeperId,
              allFileIds
            );
            await RecommendationsAPI.resolve(rec.id, 'deleted', {
              kept_file_id: keeperId,
              deleted_file_ids: fileIdsToDelete,
            });

            btn.textContent = 'Deleted!';
            btn.classList.add('ss-btn-success');

            setTimeout(() => {
              currentState.view = 'list';
              currentState.selectedRec = null;
              renderCurrentView(document.getElementById('ss-recommendations'));
            }, 1500);
          } catch (e) {
            const errMsg = e.message || '';
            // If file already deleted, resolve the recommendation anyway
            if (errMsg.includes('no rows in result set') || errMsg.includes('not found')) {
              try {
                await RecommendationsAPI.resolve(rec.id, 'deleted', {
                  kept_file_id: keeperId,
                  deleted_file_ids: fileIdsToDelete,
                  note: 'Files already deleted',
                });
                btn.textContent = 'Already deleted - resolved';
                btn.classList.add('ss-btn-success');
                setTimeout(() => {
                  currentState.view = 'list';
                  currentState.selectedRec = null;
                  renderCurrentView(document.getElementById('ss-recommendations'));
                }, 1500);
                return;
              } catch (_) { /* fall through to error display */ }
            }
            btn.textContent = `Failed: ${errMsg}`;
            btn.classList.add('ss-btn-error');
            btn.disabled = false;
          }
        },
        { showDontAsk: true, storageKey: 'delete-scene-files' }
      );
    });

    // Dismiss action
    container.querySelector('#ss-dismiss-btn').addEventListener('click', async () => {
      const btn = container.querySelector('#ss-dismiss-btn');
      try {
        btn.disabled = true;
        btn.textContent = 'Dismissing...';
        await RecommendationsAPI.dismiss(rec.id, 'User dismissed');
        currentState.view = 'list';
        currentState.selectedRec = null;
        renderCurrentView(document.getElementById('ss-recommendations'));
      } catch (e) {
        btn.textContent = `Failed: ${e.message}`;
        btn.disabled = false;
      }
    });
  }

  // ==================== Confirmation Modal ====================

  function showConfirmModal(message, onConfirm, options = {}) {
    const { showDontAsk = false, storageKey = null } = options;

    // Check "don't ask again"
    if (storageKey && localStorage.getItem(`ss-skip-confirm-${storageKey}`) === '1') {
      onConfirm();
      return;
    }

    // Use raw DOM to avoid Stash CSS interference
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:10000;';

    const modal = document.createElement('div');
    modal.style.cssText = 'background:#2a2a2a;border:1px solid #444;border-radius:10px;padding:1.5rem;max-width:420px;width:auto;min-width:300px;box-shadow:0 8px 32px rgba(0,0,0,0.4);';

    const body = document.createElement('div');
    body.style.cssText = 'font-size:0.95rem;line-height:1.5;margin-bottom:1rem;color:#fff;';
    body.textContent = message;
    modal.appendChild(body);

    let dontAskCheckbox = null;
    if (showDontAsk) {
      const label = document.createElement('label');
      label.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:0.85rem;color:#888;margin-bottom:1rem;cursor:pointer;';
      dontAskCheckbox = document.createElement('input');
      dontAskCheckbox.type = 'checkbox';
      label.appendChild(dontAskCheckbox);
      label.appendChild(document.createTextNode("Don't ask again"));
      modal.appendChild(label);
    }

    const actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:0.75rem;justify-content:flex-end;';

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'ss-btn ss-btn-danger';
    confirmBtn.style.cssText = 'padding:8px 18px;border-radius:6px;font-size:0.9rem;';
    confirmBtn.textContent = 'Confirm';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'ss-btn ss-btn-secondary';
    cancelBtn.style.cssText = 'padding:8px 18px;border-radius:6px;font-size:0.9rem;';
    cancelBtn.textContent = 'Cancel';

    actions.appendChild(confirmBtn);
    actions.appendChild(cancelBtn);
    modal.appendChild(actions);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    confirmBtn.addEventListener('click', () => {
      if (storageKey && dontAskCheckbox && dontAskCheckbox.checked) {
        localStorage.setItem(`ss-skip-confirm-${storageKey}`, '1');
      }
      overlay.remove();
      onConfirm();
    });

    cancelBtn.addEventListener('click', () => {
      overlay.remove();
    });

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });

    const escHandler = (e) => {
      if (e.key === 'Escape') {
        overlay.remove();
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);
  }

  // ==================== Help System ====================

  const HELP_FINGERPRINTS = `
    <div class="ss-help-section">
      <h4>What is the Identification Database?</h4>
      <p>The identification database contains face embeddings for known performers across multiple stash-box sources. It powers performer identification and face-based duplicate detection.</p>
    </div>
    <div class="ss-help-section">
      <h4>What are Fingerprints?</h4>
      <p>Scene fingerprints are face recognition data extracted from your local video files. Each fingerprint records which performers' faces were detected in a scene. Generate them from the Operations tab.</p>
    </div>
    <div class="ss-help-section">
      <h4>What Does "Need Refresh" Mean?</h4>
      <p>The face recognition database was updated with improved data (better face alignment, more performers). Scenes with outdated fingerprints should be regenerated for improved accuracy.</p>
    </div>
  `;

  // HELP_ACTION_RUNNER removed - actions now live on Operations tab

  const HELP_REC_TYPES = `
    <div class="ss-help-section">
      <h4>How Recommendations Work</h4>
      <p>Running an analysis creates recommendations. Each recommendation can be:</p>
      <ul class="ss-help-list">
        <li><strong>Acted on</strong> &mdash; merge performers, delete files, or apply upstream updates</li>
        <li><strong>Dismissed</strong> &mdash; hide the recommendation (view later from the Dismissed tab)</li>
        <li><strong>Left pending</strong> &mdash; come back to it later</li>
      </ul>
    </div>
    <div class="ss-help-section">
      <h4>Status Counts</h4>
      <p>Each type shows pending (needs review), resolved (action taken), and dismissed (hidden) counts. Click "View All" to browse recommendations of that type.</p>
    </div>
  `;

  const HELP_DUP_PERFORMER = `
    <div class="ss-help-section">
      <h4>Suggested Keeper</h4>
      <p>The performer with the most content (scenes, images, galleries) is suggested as the keeper. You can override this by selecting a different performer.</p>
    </div>
    <div class="ss-help-section">
      <h4>What Merging Does</h4>
      <p>Merging moves all scenes, images, and galleries from the other performer(s) to the keeper, then deletes the duplicates. This cannot be undone.</p>
    </div>
  `;

  const HELP_DUP_SCENE_FILES = `
    <div class="ss-help-section">
      <h4>Quality Scoring</h4>
      <p>Files are scored based on resolution (primary factor), bitrate, and codec. The highest-scoring file is marked as "Best Quality" and pre-selected.</p>
    </div>
    <div class="ss-help-tip">File deletion is permanent and irreversible. Make sure you've selected the right file to keep before proceeding.</div>
  `;

  const HELP_UPSTREAM_DETAIL = `
    <div class="ss-help-section">
      <h4>3-Column Comparison</h4>
      <p><strong>Local</strong> &mdash; your current value in Stash<br>
      <strong>Upstream</strong> &mdash; the current value on StashDB<br>
      <strong>Result</strong> &mdash; the value that will be written to Stash when you apply</p>
    </div>
    <div class="ss-help-section">
      <h4>Snapshots</h4>
      <p>A snapshot records what StashDB looked like when you last synced. The 3-way diff uses this to distinguish "you changed it locally" from "StashDB changed it upstream".</p>
    </div>
    <div class="ss-help-section">
      <h4>Dismiss Options</h4>
      <p><strong>Dismiss</strong> &mdash; hides this recommendation until the next analysis finds new changes.<br>
      <strong>Permanent dismiss</strong> &mdash; ignores this performer until you un-dismiss from the Dismissed tab.</p>
    </div>
  `;

  function createInfoIcon(onClick) {
    const btn = document.createElement('button');
    btn.className = 'ss-info-btn';
    btn.title = 'Help';
    btn.setAttribute('aria-label', 'Help');
    btn.textContent = 'i';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      onClick();
    });
    return btn;
  }

  function showHelpModal(title, contentHtml) {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:10001;';

    const modal = document.createElement('div');
    modal.style.cssText = 'background:#2a2a2a;border:1px solid #444;border-radius:10px;max-width:560px;width:90%;max-height:80vh;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.4);';

    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:1rem 1.25rem;border-bottom:1px solid #444;';
    const titleEl = document.createElement('h3');
    titleEl.style.cssText = 'margin:0;font-size:1.1rem;font-weight:600;color:#fff;';
    titleEl.textContent = title;
    const closeBtn = document.createElement('button');
    closeBtn.style.cssText = 'background:none;border:none;font-size:1.5rem;color:#888;cursor:pointer;padding:0;line-height:1;';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.textContent = '\u00d7';
    closeBtn.addEventListener('click', () => overlay.remove());
    header.appendChild(titleEl);
    header.appendChild(closeBtn);

    const body = document.createElement('div');
    body.style.cssText = 'padding:1.25rem;overflow-y:auto;flex:1;';
    body.innerHTML = contentHtml;

    modal.appendChild(header);
    modal.appendChild(body);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });

    const escHandler = (e) => {
      if (e.key === 'Escape') {
        overlay.remove();
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);
  }


  // ==================== Upstream Performer Validation ====================

  async function validatePerformerMerge(performerId, proposedName, proposedDisambig, proposedAliases) {
    const errors = [];

    // 1. Name uniqueness - query all performers with this name
    if (proposedName) {
      try {
        const nameCheck = await SS.stashQuery(`
          query FindPerformersByName($name: String!) {
            findPerformers(performer_filter: { name: { value: $name, modifier: EQUALS } }) {
              performers { id name disambiguation }
            }
          }
        `, { name: proposedName });
        const conflicts = (nameCheck?.findPerformers?.performers || [])
          .filter(p => p.id !== performerId);
        if (conflicts.length > 0) {
          const c = conflicts[0];
          const disambigNote = proposedDisambig ? '' : ' (add a disambiguation to make it unique)';
          errors.push(`Name "${proposedName}" already used by performer "${c.name}"${disambigNote}`);
        }
      } catch (e) {
        // Name check failed - don't block, just warn
        console.warn('[Stash Sense] Name uniqueness check failed:', e);
      }
    }

    // 2. Alias can't match performer's own name
    if (proposedName && proposedAliases) {
      const nameLower = proposedName.toLowerCase();
      for (const alias of proposedAliases) {
        if (alias.toLowerCase() === nameLower) {
          errors.push(`Alias "${alias}" matches the performer's name`);
        }
      }
    }

    // 3. No duplicate aliases
    if (proposedAliases) {
      const seen = new Set();
      for (const alias of proposedAliases) {
        const lower = alias.toLowerCase();
        if (seen.has(lower)) {
          errors.push(`Duplicate alias: "${alias}"`);
        }
        seen.add(lower);
      }
    }

    return errors;
  }

  // ==================== Upstream Performer Detail ====================

  async function renderUpstreamPerformerDetail(container, rec) {
    const details = rec.details;
    const rawChanges = details.changes || [];
    const performerId = details.performer_id;

    const changes = filterRealChanges(rawChanges);

    // If all changes were filtered out, auto-resolve and go back
    if (changes.length === 0) {
      try {
        await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', { note: 'All differences were cosmetic' });
      } catch (_) {}
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
      return;
    }

    // Load normalize setting
    let normalizeEnum = true;
    try {
      const val = await RecommendationsAPI.getUserSetting('normalize_enum_display');
      normalizeEnum = val !== false;
    } catch (_) {}

    // Display value helper - applies enum normalization if enabled
    function displayValue(val) {
      const formatted = formatFieldValue(val);
      return normalizeEnum ? normalizeEnumValue(formatted) : formatted;
    }

    // Smart default: prefer upstream (stash-box is source of truth)
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = upstreamVal === null || upstreamVal === undefined || upstreamVal === '';
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    // Build the header with gear icon
    const wrapper = document.createElement('div');
    wrapper.className = 'ss-detail-upstream-performer';

    // Header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'ss-upstream-header';
    headerDiv.innerHTML = `
      <img src="${details.performer_image_path || ''}" alt="${details.performer_name || ''}" onerror="this.style.display='none'" />
      <div style="flex:1;">
        <h2 style="margin: 0 0 4px 0;">
          <a href="/performers/${performerId}" target="_blank">${details.performer_name || 'Unknown'}</a>
        </h2>
        <span class="ss-upstream-endpoint-badge">${details.endpoint_name || 'Upstream'}</span>
      </div>
    `;
    headerDiv.appendChild(createInfoIcon(() => showHelpModal('Upstream Performer Changes', HELP_UPSTREAM_DETAIL)));
    wrapper.appendChild(headerDiv);

    // Quick select buttons
    const quickActions = document.createElement('div');
    quickActions.className = 'ss-upstream-quick-actions';
    const keepAllBtn = document.createElement('button');
    keepAllBtn.textContent = 'Keep All Local';
    const acceptAllBtn = document.createElement('button');
    acceptAllBtn.textContent = 'Accept All Upstream';
    quickActions.appendChild(keepAllBtn);
    quickActions.appendChild(acceptAllBtn);
    wrapper.appendChild(quickActions);

    // Build field rows
    changes.forEach((change, idx) => {
      const mergeType = change.merge_type || 'simple';
      const fieldRow = document.createElement('div');
      fieldRow.className = 'ss-upstream-field-row';
      fieldRow.dataset.fieldIndex = idx;
      fieldRow.dataset.fieldKey = change.field;
      fieldRow.dataset.mergeType = mergeType;

      // Field label
      const label = document.createElement('div');
      label.className = 'ss-upstream-field-label';
      label.textContent = change.field_label || change.field;
      fieldRow.appendChild(label);

      if (mergeType === 'alias_list') {
        // Alias list: 2-column sub-layout (items on left, result summary on right)
        renderAliasListField(fieldRow, change, idx);
      } else {
        // Simple, name, text: 3-column layout
        renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault);
      }

      wrapper.appendChild(fieldRow);
    });

    // Validation errors
    const errorDiv = document.createElement('div');
    errorDiv.id = 'ss-upstream-validation-errors';
    errorDiv.className = 'ss-upstream-validation-error';
    errorDiv.style.display = 'none';
    wrapper.appendChild(errorDiv);

    // Action bar
    const actionBar = document.createElement('div');
    actionBar.className = 'ss-upstream-action-bar';

    const applyBtn = document.createElement('button');
    applyBtn.className = 'ss-btn ss-upstream-apply-btn';
    applyBtn.id = 'ss-upstream-apply';
    applyBtn.textContent = 'Apply Selected Changes';

    const dismissDropdown = document.createElement('div');
    dismissDropdown.style.cssText = 'position:relative;';
    const dismissToggle = document.createElement('button');
    dismissToggle.className = 'ss-btn ss-upstream-dismiss-btn';
    dismissToggle.textContent = 'Dismiss';
    const dismissMenu = document.createElement('div');
    dismissMenu.style.cssText = 'display:none;position:absolute;bottom:100%;left:0;background:#2a2a2a;border:1px solid #444;border-radius:6px;padding:4px 0;min-width:220px;z-index:10;';

    const dismissOptions = [
      { label: 'Dismiss this update', permanent: false },
      { label: 'Never show for this performer', permanent: true },
    ];

    dismissOptions.forEach(opt => {
      const optBtn = document.createElement('button');
      optBtn.style.cssText = 'display:block;width:100%;text-align:left;padding:8px 12px;background:none;border:none;color:#fff;cursor:pointer;font-size:0.85rem;';
      optBtn.textContent = opt.label;
      optBtn.addEventListener('mouseenter', () => { optBtn.style.background = 'rgba(255,255,255,0.05)'; });
      optBtn.addEventListener('mouseleave', () => { optBtn.style.background = 'none'; });
      optBtn.addEventListener('click', async () => {
        dismissMenu.style.display = 'none';
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, 'User dismissed', opt.permanent);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          dismissToggle.textContent = `Failed: ${e.message}`;
          dismissToggle.disabled = false;
        }
      });
      dismissMenu.appendChild(optBtn);
    });

    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });

    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    dismissDropdown.appendChild(dismissToggle);
    dismissDropdown.appendChild(dismissMenu);
    actionBar.appendChild(applyBtn);
    actionBar.appendChild(dismissDropdown);
    wrapper.appendChild(actionBar);

    container.innerHTML = '';
    container.appendChild(wrapper);

    // === Quick select wiring ===
    keepAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          // Check all local/both items, uncheck upstream-only
          row.querySelectorAll('.ss-upstream-alias-item').forEach(item => {
            const cb = item.querySelector('input[type="checkbox"]');
            cb.checked = item.classList.contains('local-only') || item.classList.contains('both');
          });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = true;
            upstreamCb.checked = false;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.local_value) === '(empty)' ? '' : formatFieldValue(change.local_value);
          }
        }
      });
    });

    acceptAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          // Check all items (merge all)
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]').forEach(cb => { cb.checked = true; });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = false;
            upstreamCb.checked = true;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.upstream_value) === '(empty)' ? '' : formatFieldValue(change.upstream_value);
          }
          // For name type: also check "add old name as alias" when switching to upstream
          const aliasOpt = row.querySelector('.ss-upstream-name-alias-cb');
          if (aliasOpt) aliasOpt.checked = true;
        }
      });
    });

    // === Apply handler ===
    applyBtn.addEventListener('click', async () => {
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';
      const fields = {};

      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const fieldKey = row.dataset.fieldKey;
        const mergeType = row.dataset.mergeType;
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = changes[fieldIndex];

        if (mergeType === 'alias_list') {
          const checkedAliases = [];
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
            checkedAliases.push(cb.value);
          });
          fields[fieldKey] = checkedAliases;
        } else {
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (!resultInput) return;

          const resultVal = resultInput.value.trim();
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');

          // Skip if result equals local (no change)
          if (resultVal === localStr) {
            // But check if name type has alias add
            if (mergeType === 'name') {
              const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
              if (aliasCb && aliasCb.checked) {
                const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
                if (upstreamCb && !upstreamCb.checked) {
                  // Keeping local name, add upstream as alias
                  fields['_alias_add'] = fields['_alias_add'] || [];
                  fields['_alias_add'].push(String(change.upstream_value || ''));
                }
              }
            }
            return;
          }

          // Result differs from local -> apply change
          fields[fieldKey] = resultVal;

          // Handle name merge alias addition
          if (mergeType === 'name') {
            const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
            if (aliasCb && aliasCb.checked) {
              const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
              if (upstreamCb && upstreamCb.checked) {
                // Accepting upstream name, demote local to alias
                fields['_alias_add'] = fields['_alias_add'] || [];
                fields['_alias_add'].push(String(change.local_value || ''));
              }
            }
          }
        }
      });

      // Check if any changes were selected
      const hasChanges = Object.keys(fields).length > 0;
      if (!hasChanges) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          applyBtn.textContent = 'Done!';
          applyBtn.classList.add('ss-btn-success');
          setTimeout(() => {
            currentState.view = 'list';
            currentState.selectedRec = null;
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 1500);
        } catch (e) {
          applyBtn.textContent = `Failed: ${e.message}`;
          applyBtn.classList.add('ss-btn-error');
          applyBtn.disabled = false;
        }
        return;
      }

      // Run validation before applying
      applyBtn.disabled = true;
      applyBtn.textContent = 'Validating...';

      const proposedName = fields.name || details.performer_name;
      const proposedDisambig = fields.disambiguation || null;
      const proposedAliases = fields.aliases || fields._alias_add || [];

      const validationErrors = await validatePerformerMerge(
        performerId, proposedName, proposedDisambig, proposedAliases
      );

      if (validationErrors.length > 0) {
        errorDiv.innerHTML = validationErrors.map(e => `<div>${escapeHtml(e)}</div>`).join('');
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.disabled = false;
        return;
      }

      try {
        applyBtn.textContent = 'Applying...';
        await RecommendationsAPI.updatePerformer(performerId, fields);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        applyBtn.textContent = 'Applied!';
        applyBtn.classList.add('ss-btn-success');

        setTimeout(() => {
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        }, 1500);
      } catch (e) {
        let errorMsg = e.message;
        if (errorMsg.includes('already exists') || errorMsg.includes('name')) {
          errorMsg = `Name conflict: ${errorMsg}. Try adding a disambiguation.`;
        } else if (errorMsg.includes('duplicate') || errorMsg.includes('alias')) {
          errorMsg = `Alias conflict: ${errorMsg}. Try removing duplicate aliases.`;
        }
        errorDiv.innerHTML = `<div>${escapeHtml(errorMsg)}</div>`;
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.classList.remove('ss-btn-error');
        applyBtn.disabled = false;
      }
    });
  }

  /**
   * Upstream Tag Detail View
   * Simpler than performer — only name, description, aliases. No image, no compound fields.
   */
  async function renderUpstreamTagDetail(container, rec) {
    const details = rec.details;
    const rawChanges = details.changes || [];
    const tagId = details.tag_id;

    const changes = filterRealChanges(rawChanges);

    // If all changes were filtered out, auto-resolve and go back
    if (changes.length === 0) {
      try {
        await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', { note: 'All differences were cosmetic' });
      } catch (_) {}
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
      return;
    }

    // Display value helper
    function displayValue(val) {
      return formatFieldValue(val);
    }

    // Smart default: prefer upstream (stash-box is source of truth)
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = upstreamVal === null || upstreamVal === undefined || upstreamVal === '';
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'ss-detail-upstream-performer'; // reuse performer styles

    // Header (tag icon instead of image)
    const headerDiv = document.createElement('div');
    headerDiv.className = 'ss-upstream-header';
    headerDiv.innerHTML = `
      <div class="ss-rec-tag-icon" style="width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(255,255,255,0.05);flex-shrink:0;">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><path d="M21.41 11.58l-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42zM5.5 7C4.67 7 4 6.33 4 5.5S4.67 4 5.5 4 7 4.67 7 5.5 6.33 7 5.5 7z"/></svg>
      </div>
      <div style="flex:1;">
        <h2 style="margin: 0 0 4px 0;">
          <a href="/tags/${tagId}" target="_blank">${details.tag_name || 'Unknown'}</a>
        </h2>
        <span class="ss-upstream-endpoint-badge">${details.endpoint_name || 'Upstream'}</span>
      </div>
    `;
    wrapper.appendChild(headerDiv);

    // Quick select buttons
    const quickActions = document.createElement('div');
    quickActions.className = 'ss-upstream-quick-actions';
    const keepAllBtn = document.createElement('button');
    keepAllBtn.textContent = 'Keep All Local';
    const acceptAllBtn = document.createElement('button');
    acceptAllBtn.textContent = 'Accept All Upstream';
    quickActions.appendChild(keepAllBtn);
    quickActions.appendChild(acceptAllBtn);
    wrapper.appendChild(quickActions);

    // Build field rows
    changes.forEach((change, idx) => {
      const mergeType = change.merge_type || 'simple';
      const fieldRow = document.createElement('div');
      fieldRow.className = 'ss-upstream-field-row';
      fieldRow.dataset.fieldIndex = idx;
      fieldRow.dataset.fieldKey = change.field;
      fieldRow.dataset.mergeType = mergeType;

      const label = document.createElement('div');
      label.className = 'ss-upstream-field-label';
      label.textContent = change.field_label || change.field;
      fieldRow.appendChild(label);

      if (mergeType === 'alias_list') {
        renderAliasListField(fieldRow, change, idx);
      } else {
        renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault);
      }

      wrapper.appendChild(fieldRow);
    });

    // Validation errors
    const errorDiv = document.createElement('div');
    errorDiv.id = 'ss-upstream-validation-errors';
    errorDiv.className = 'ss-upstream-validation-error';
    errorDiv.style.display = 'none';
    wrapper.appendChild(errorDiv);

    // Action bar
    const actionBar = document.createElement('div');
    actionBar.className = 'ss-upstream-action-bar';

    const applyBtn = document.createElement('button');
    applyBtn.className = 'ss-btn ss-upstream-apply-btn';
    applyBtn.textContent = 'Apply Selected Changes';

    // Dismiss dropdown (reuse performer pattern)
    const dismissDropdown = document.createElement('div');
    dismissDropdown.style.cssText = 'position:relative;';
    const dismissToggle = document.createElement('button');
    dismissToggle.className = 'ss-btn ss-upstream-dismiss-btn';
    dismissToggle.textContent = 'Dismiss';
    const dismissMenu = document.createElement('div');
    dismissMenu.style.cssText = 'display:none;position:absolute;bottom:100%;left:0;background:#2a2a2a;border:1px solid #444;border-radius:6px;padding:4px 0;min-width:220px;z-index:10;';

    const dismissOptions = [
      { label: 'Dismiss this update', permanent: false },
      { label: 'Never show for this tag', permanent: true },
    ];

    dismissOptions.forEach(opt => {
      const optBtn = document.createElement('button');
      optBtn.style.cssText = 'display:block;width:100%;text-align:left;padding:8px 12px;background:none;border:none;color:#fff;cursor:pointer;font-size:0.85rem;';
      optBtn.textContent = opt.label;
      optBtn.addEventListener('mouseenter', () => { optBtn.style.background = 'rgba(255,255,255,0.05)'; });
      optBtn.addEventListener('mouseleave', () => { optBtn.style.background = 'none'; });
      optBtn.addEventListener('click', async () => {
        dismissMenu.style.display = 'none';
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, 'User dismissed', opt.permanent);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          dismissToggle.textContent = `Failed: ${e.message}`;
          dismissToggle.disabled = false;
        }
      });
      dismissMenu.appendChild(optBtn);
    });

    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });

    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    dismissDropdown.appendChild(dismissToggle);
    dismissDropdown.appendChild(dismissMenu);
    actionBar.appendChild(applyBtn);
    actionBar.appendChild(dismissDropdown);
    wrapper.appendChild(actionBar);

    container.innerHTML = '';
    container.appendChild(wrapper);

    // === Quick select wiring ===
    keepAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          row.querySelectorAll('.ss-upstream-alias-item').forEach(item => {
            const cb = item.querySelector('input[type="checkbox"]');
            cb.checked = item.classList.contains('local-only') || item.classList.contains('both');
          });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = true;
            upstreamCb.checked = false;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.local_value) === '(empty)' ? '' : formatFieldValue(change.local_value);
          }
        }
      });
    });

    acceptAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const mt = row.dataset.mergeType;
        if (mt === 'alias_list') {
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]').forEach(cb => { cb.checked = true; });
          updateAliasResultSummary(row);
        } else {
          const localCb = row.querySelector('.ss-upstream-cb-local');
          const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (localCb && upstreamCb && resultInput) {
            localCb.checked = false;
            upstreamCb.checked = true;
            const change = changes[parseInt(row.dataset.fieldIndex)];
            resultInput.value = formatFieldValue(change.upstream_value) === '(empty)' ? '' : formatFieldValue(change.upstream_value);
          }
          const aliasOpt = row.querySelector('.ss-upstream-name-alias-cb');
          if (aliasOpt) aliasOpt.checked = true;
        }
      });
    });

    // === Apply handler ===
    applyBtn.addEventListener('click', async () => {
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';
      const fields = {};

      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const fieldKey = row.dataset.fieldKey;
        const mergeType = row.dataset.mergeType;
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = changes[fieldIndex];

        if (mergeType === 'alias_list') {
          const checkedAliases = [];
          row.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
            checkedAliases.push(cb.value);
          });
          fields[fieldKey] = checkedAliases;
        } else {
          const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
          if (!resultInput) return;

          const resultVal = resultInput.value.trim();
          const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');

          // Skip if result equals local (no change)
          if (resultVal === localStr) {
            if (mergeType === 'name') {
              const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
              if (aliasCb && aliasCb.checked) {
                const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
                if (upstreamCb && !upstreamCb.checked) {
                  fields['_alias_add'] = fields['_alias_add'] || [];
                  fields['_alias_add'].push(String(change.upstream_value || ''));
                }
              }
            }
            return;
          }

          fields[fieldKey] = resultVal;

          if (mergeType === 'name') {
            const aliasCb = row.querySelector('.ss-upstream-name-alias-cb');
            if (aliasCb && aliasCb.checked) {
              const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
              if (upstreamCb && upstreamCb.checked) {
                fields['_alias_add'] = fields['_alias_add'] || [];
                fields['_alias_add'].push(String(change.local_value || ''));
              }
            }
          }
        }
      });

      const hasChanges = Object.keys(fields).length > 0;
      if (!hasChanges) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          applyBtn.textContent = 'Done!';
          applyBtn.classList.add('ss-btn-success');
          setTimeout(() => {
            currentState.view = 'list';
            currentState.selectedRec = null;
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 1500);
        } catch (e) {
          applyBtn.textContent = `Failed: ${e.message}`;
          applyBtn.classList.add('ss-btn-error');
          applyBtn.disabled = false;
        }
        return;
      }

      // No name uniqueness validation needed for tags — just apply
      try {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        await RecommendationsAPI.updateTag(tagId, fields);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        applyBtn.textContent = 'Applied!';
        applyBtn.classList.add('ss-btn-success');

        setTimeout(() => {
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        }, 1500);
      } catch (e) {
        errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.disabled = false;
      }
    });
  }

  /**
   * Upstream Studio Detail View
   * Similar to tag — name, url, parent_studio. No image, no compound fields.
   * Parent studio shows human-readable name with StashBox UUID stored for resolution.
   */
  async function renderUpstreamStudioDetail(container, rec) {
    const details = rec.details;
    const rawChanges = details.changes || [];
    const studioId = details.studio_id;

    const changes = filterRealChanges(rawChanges);

    // If all changes were filtered out, auto-resolve and go back
    if (changes.length === 0) {
      try {
        await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', { note: 'All differences were cosmetic' });
      } catch (_) {}
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
      return;
    }

    // Display value helper
    function displayValue(val) {
      return formatFieldValue(val);
    }

    // Smart default: prefer upstream (stash-box is source of truth)
    function smartDefault(localVal, upstreamVal) {
      const upstreamEmpty = upstreamVal === null || upstreamVal === undefined || upstreamVal === '';
      if (!upstreamEmpty) return 'upstream';
      return 'local';
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'ss-detail-upstream-performer'; // reuse performer styles

    // Header (studio icon instead of image)
    const headerDiv = document.createElement('div');
    headerDiv.className = 'ss-upstream-header';
    headerDiv.innerHTML = `
      <div class="ss-rec-tag-icon" style="width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(255,255,255,0.05);flex-shrink:0;">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><path d="M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z"/></svg>
      </div>
      <div style="flex:1;">
        <h2 style="margin: 0 0 4px 0;">
          <a href="/studios/${studioId}" target="_blank">${details.studio_name || 'Unknown'}</a>
        </h2>
        <span class="ss-upstream-endpoint-badge">${details.endpoint_name || 'Upstream'}</span>
      </div>
    `;
    wrapper.appendChild(headerDiv);

    // Quick select buttons
    const quickActions = document.createElement('div');
    quickActions.className = 'ss-upstream-quick-actions';
    const keepAllBtn = document.createElement('button');
    keepAllBtn.textContent = 'Keep All Local';
    const acceptAllBtn = document.createElement('button');
    acceptAllBtn.textContent = 'Accept All Upstream';
    quickActions.appendChild(keepAllBtn);
    quickActions.appendChild(acceptAllBtn);
    wrapper.appendChild(quickActions);

    // Build field rows
    changes.forEach((change, idx) => {
      const mergeType = change.merge_type || 'simple';
      const fieldRow = document.createElement('div');
      fieldRow.className = 'ss-upstream-field-row';
      fieldRow.dataset.fieldIndex = idx;
      fieldRow.dataset.fieldKey = change.field;
      fieldRow.dataset.mergeType = mergeType;

      const label = document.createElement('div');
      label.className = 'ss-upstream-field-label';
      label.textContent = change.field_label || change.field;
      fieldRow.appendChild(label);

      renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault);

      wrapper.appendChild(fieldRow);
    });

    // Validation errors
    const errorDiv = document.createElement('div');
    errorDiv.id = 'ss-upstream-validation-errors';
    errorDiv.className = 'ss-upstream-validation-error';
    errorDiv.style.display = 'none';
    wrapper.appendChild(errorDiv);

    // Action bar
    const actionBar = document.createElement('div');
    actionBar.className = 'ss-upstream-action-bar';

    const applyBtn = document.createElement('button');
    applyBtn.className = 'ss-btn ss-upstream-apply-btn';
    applyBtn.textContent = 'Apply Selected Changes';

    // Dismiss dropdown (reuse tag pattern)
    const dismissDropdown = document.createElement('div');
    dismissDropdown.style.cssText = 'position:relative;';
    const dismissToggle = document.createElement('button');
    dismissToggle.className = 'ss-btn ss-upstream-dismiss-btn';
    dismissToggle.textContent = 'Dismiss';
    const dismissMenu = document.createElement('div');
    dismissMenu.style.cssText = 'display:none;position:absolute;bottom:100%;left:0;background:#2a2a2a;border:1px solid #444;border-radius:6px;padding:4px 0;min-width:220px;z-index:10;';

    const dismissOptions = [
      { label: 'Dismiss this update', permanent: false },
      { label: 'Never show for this studio', permanent: true },
    ];
    dismissOptions.forEach(opt => {
      const optBtn = document.createElement('button');
      optBtn.textContent = opt.label;
      optBtn.style.cssText = 'display:block;width:100%;text-align:left;padding:8px 16px;background:none;border:none;color:#fff;cursor:pointer;font-size:13px;';
      optBtn.addEventListener('mouseenter', () => { optBtn.style.background = 'rgba(255,255,255,0.05)'; });
      optBtn.addEventListener('mouseleave', () => { optBtn.style.background = 'none'; });
      optBtn.addEventListener('click', async () => {
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, null, opt.permanent);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
          errorDiv.style.display = 'block';
          dismissToggle.disabled = false;
          dismissToggle.textContent = 'Dismiss';
        }
      });
      dismissMenu.appendChild(optBtn);
    });
    dismissDropdown.appendChild(dismissToggle);
    dismissDropdown.appendChild(dismissMenu);

    actionBar.appendChild(applyBtn);
    actionBar.appendChild(dismissDropdown);
    wrapper.appendChild(actionBar);

    container.innerHTML = '';
    container.appendChild(wrapper);

    // === Event handlers ===

    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });

    // Close dismiss menu when clicking outside (match tag pattern)
    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    keepAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const localCb = row.querySelector('.ss-upstream-cb-local');
        const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
        if (localCb) { localCb.checked = true; localCb.dispatchEvent(new Event('change')); }
        if (upstreamCb) upstreamCb.checked = false;
      });
    });

    acceptAllBtn.addEventListener('click', () => {
      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const localCb = row.querySelector('.ss-upstream-cb-local');
        const upstreamCb = row.querySelector('.ss-upstream-cb-upstream');
        if (upstreamCb) { upstreamCb.checked = true; upstreamCb.dispatchEvent(new Event('change')); }
        if (localCb) localCb.checked = false;
        // Update result input
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = changes[fieldIndex];
        const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
        if (resultInput) {
          resultInput.value = formatFieldValue(change.upstream_value) === '(empty)' ? '' : formatFieldValue(change.upstream_value);
        }
      });
    });

    // === Apply handler ===
    applyBtn.addEventListener('click', async () => {
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';
      const fields = {};

      wrapper.querySelectorAll('.ss-upstream-field-row').forEach(row => {
        const fieldKey = row.dataset.fieldKey;
        const mergeType = row.dataset.mergeType;
        const fieldIndex = parseInt(row.dataset.fieldIndex);
        const change = changes[fieldIndex];

        const resultInput = row.querySelector('.ss-upstream-result-input, .ss-upstream-textarea');
        if (!resultInput) return;

        const resultVal = resultInput.value.trim();
        const localStr = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');

        // Skip if result equals local (no change)
        if (resultVal === localStr) return;

        fields[fieldKey] = resultVal;
      });

      const hasChanges = Object.keys(fields).length > 0;
      if (!hasChanges) {
        try {
          applyBtn.disabled = true;
          applyBtn.textContent = 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          applyBtn.textContent = 'Done!';
          applyBtn.classList.add('ss-btn-success');
          setTimeout(() => {
            currentState.view = 'list';
            currentState.selectedRec = null;
            renderCurrentView(document.getElementById('ss-recommendations'));
          }, 1500);
        } catch (e) {
          applyBtn.textContent = `Failed: ${e.message}`;
          applyBtn.classList.add('ss-btn-error');
          applyBtn.disabled = false;
        }
        return;
      }

      try {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        await RecommendationsAPI.updateStudio(studioId, fields, details.endpoint);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        applyBtn.textContent = 'Applied!';
        applyBtn.classList.add('ss-btn-success');

        setTimeout(() => {
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        }, 1500);
      } catch (e) {
        errorDiv.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
        errorDiv.style.display = 'block';
        applyBtn.textContent = 'Apply Selected Changes';
        applyBtn.disabled = false;
      }
    });
  }

  /**
   * Render a 3-column compare field row (simple, name, text merge types).
   * Layout: [x] Local value | [ ] Upstream value | Result: [editable input]
   */
  function renderCompareField(fieldRow, change, idx, mergeType, displayValue, smartDefault) {
    const compareRow = document.createElement('div');
    compareRow.className = 'ss-upstream-compare-row';

    const defaultChoice = smartDefault(change.local_value, change.upstream_value);

    // Local cell
    const localCell = document.createElement('div');
    localCell.className = 'ss-upstream-value-cell local';
    const localLabel = document.createElement('div');
    localLabel.className = 'ss-upstream-value-label';
    localLabel.textContent = 'Local';
    const localCheckLabel = document.createElement('label');
    const localCb = document.createElement('input');
    localCb.type = 'checkbox';
    localCb.className = 'ss-upstream-cb-local';
    localCb.checked = defaultChoice === 'local';
    localCheckLabel.appendChild(localCb);
    localCheckLabel.appendChild(document.createTextNode(' ' + displayValue(change.local_value)));
    localCell.appendChild(localLabel);
    localCell.appendChild(localCheckLabel);

    // Upstream cell
    const upstreamCell = document.createElement('div');
    upstreamCell.className = 'ss-upstream-value-cell upstream';
    const upstreamLabel = document.createElement('div');
    upstreamLabel.className = 'ss-upstream-value-label';
    upstreamLabel.textContent = 'Upstream';
    const upstreamCheckLabel = document.createElement('label');
    const upstreamCb = document.createElement('input');
    upstreamCb.type = 'checkbox';
    upstreamCb.className = 'ss-upstream-cb-upstream';
    upstreamCb.checked = defaultChoice === 'upstream';
    upstreamCheckLabel.appendChild(upstreamCb);
    upstreamCheckLabel.appendChild(document.createTextNode(' ' + displayValue(change.upstream_value)));
    upstreamCell.appendChild(upstreamLabel);
    upstreamCell.appendChild(upstreamCheckLabel);

    // Result cell
    const resultCell = document.createElement('div');
    resultCell.className = 'ss-upstream-value-cell result';
    const resultLabel = document.createElement('div');
    resultLabel.className = 'ss-upstream-value-label';
    resultLabel.textContent = 'Result';

    let resultInput;
    if (mergeType === 'text') {
      resultInput = document.createElement('textarea');
      resultInput.className = 'ss-upstream-textarea';
    } else {
      resultInput = document.createElement('input');
      resultInput.type = 'text';
      resultInput.className = 'ss-upstream-result-input';
    }

    // Set initial result value based on smart default
    const defaultVal = defaultChoice === 'upstream' ? change.upstream_value : change.local_value;
    resultInput.value = formatFieldValue(defaultVal) === '(empty)' ? '' : String(defaultVal || '');

    resultCell.appendChild(resultLabel);
    resultCell.appendChild(resultInput);

    compareRow.appendChild(localCell);
    compareRow.appendChild(upstreamCell);
    compareRow.appendChild(resultCell);
    fieldRow.appendChild(compareRow);

    // Paired checkbox logic: clicking local unchecks upstream (and vice versa)
    localCb.addEventListener('change', () => {
      if (localCb.checked) {
        upstreamCb.checked = false;
        resultInput.value = formatFieldValue(change.local_value) === '(empty)' ? '' : String(change.local_value || '');
      }
    });

    upstreamCb.addEventListener('change', () => {
      if (upstreamCb.checked) {
        localCb.checked = false;
        resultInput.value = formatFieldValue(change.upstream_value) === '(empty)' ? '' : String(change.upstream_value || '');
      }
    });

    // Editing result directly unchecks both checkboxes
    resultInput.addEventListener('input', () => {
      localCb.checked = false;
      upstreamCb.checked = false;
    });

    // Name merge type: add "Add old name as alias" checkbox
    if (mergeType === 'name') {
      const aliasOption = document.createElement('div');
      aliasOption.className = 'ss-upstream-name-alias-option';
      const aliasLabel = document.createElement('label');
      const aliasCb = document.createElement('input');
      aliasCb.type = 'checkbox';
      aliasCb.className = 'ss-upstream-name-alias-cb';
      aliasCb.checked = true;
      aliasLabel.appendChild(aliasCb);
      aliasLabel.appendChild(document.createTextNode(' Add old name as alias when switching'));
      aliasOption.appendChild(aliasLabel);
      fieldRow.appendChild(aliasOption);
    }
  }

  /**
   * Render alias list field: vertical stacked items with per-item checkbox,
   * all items checked by default (merge behavior), with result summary.
   */
  function renderAliasListField(fieldRow, change, idx) {
    const localAliases = change.local_value || [];
    const upstreamAliases = change.upstream_value || [];
    const allAliases = buildAliasList(localAliases, upstreamAliases);

    const aliasContainer = document.createElement('div');
    aliasContainer.className = 'ss-upstream-alias-list-container';
    aliasContainer.dataset.fieldIndex = idx;
    aliasContainer.dataset.mergeType = 'alias_list';
    aliasContainer.dataset.fieldKey = change.field;

    // Two-column sub-layout
    const subLayout = document.createElement('div');
    subLayout.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;align-items:start;';

    // Left: item checkboxes
    const itemsCol = document.createElement('div');
    const aliasList = document.createElement('div');
    aliasList.className = 'ss-upstream-alias-list';

    allAliases.forEach((a, ai) => {
      const item = document.createElement('label');
      item.className = `ss-upstream-alias-item ${a.source}`;
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.name = `field_${idx}_alias_${ai}`;
      cb.value = a.value;
      cb.checked = true; // All checked by default (merge behavior)
      item.appendChild(cb);
      const span = document.createElement('span');
      span.textContent = a.value;
      item.appendChild(span);
      const tag = document.createElement('span');
      tag.className = 'ss-upstream-alias-tag';
      tag.textContent = a.source === 'both' ? 'both' : a.source === 'local-only' ? 'local' : 'upstream';
      item.appendChild(tag);
      aliasList.appendChild(item);

      cb.addEventListener('change', () => updateAliasResultSummary(fieldRow));
    });

    itemsCol.appendChild(aliasList);

    // Add custom alias button
    const addBtn = document.createElement('button');
    addBtn.className = 'ss-btn ss-btn-secondary';
    addBtn.style.cssText = 'margin-top:0.5rem;padding:4px 10px;font-size:0.8rem;';
    addBtn.textContent = '+ Add custom alias';
    addBtn.addEventListener('click', () => {
      const newAlias = prompt('Enter new alias:');
      if (!newAlias || !newAlias.trim()) return;
      const existingCount = aliasList.querySelectorAll('.ss-upstream-alias-item').length;
      const item = document.createElement('label');
      item.className = 'ss-upstream-alias-item both';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.name = `field_${idx}_alias_${existingCount}`;
      cb.value = newAlias.trim();
      cb.checked = true;
      item.appendChild(cb);
      const span = document.createElement('span');
      span.textContent = newAlias.trim();
      item.appendChild(span);
      const tag = document.createElement('span');
      tag.className = 'ss-upstream-alias-tag';
      tag.textContent = 'custom';
      item.appendChild(tag);
      aliasList.appendChild(item);
      cb.addEventListener('change', () => updateAliasResultSummary(fieldRow));
      updateAliasResultSummary(fieldRow);
    });
    itemsCol.appendChild(addBtn);

    // Right: result summary
    const resultCol = document.createElement('div');
    resultCol.className = 'ss-upstream-alias-result';
    resultCol.innerHTML = '<div class="ss-upstream-value-label">Result</div><div class="ss-upstream-alias-result-content"></div>';

    subLayout.appendChild(itemsCol);
    subLayout.appendChild(resultCol);
    aliasContainer.appendChild(subLayout);
    fieldRow.appendChild(aliasContainer);

    // Initial result summary
    updateAliasResultSummary(fieldRow);
  }

  function updateAliasResultSummary(fieldRow) {
    const resultContent = fieldRow.querySelector('.ss-upstream-alias-result-content');
    if (!resultContent) return;
    const checkedAliases = [];
    fieldRow.querySelectorAll('.ss-upstream-alias-item input[type="checkbox"]:checked').forEach(cb => {
      checkedAliases.push(cb.value);
    });
    if (checkedAliases.length === 0) {
      resultContent.innerHTML = '<span class="ss-upstream-alias-result-count">0</span> items';
    } else {
      resultContent.innerHTML = `<span class="ss-upstream-alias-result-count">${checkedAliases.length}</span> items<ul style="margin:0.25rem 0 0 1.25rem;padding:0;list-style:disc;">${checkedAliases.map(a => `<li style="font-size:0.8rem;color:#fff;margin-bottom:2px;">${escapeHtml(a)}</li>`).join('')}</ul>`;
    }
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
  }

  function formatFieldValue(val) {
    if (val === null || val === undefined) return '(empty)';
    if (Array.isArray(val)) return val.join(', ') || '(empty)';
    return String(val) || '(empty)';
  }

  function buildAliasList(localAliases, upstreamAliases) {
    const localArr = (localAliases || []).map(String);
    const upstreamArr = (upstreamAliases || []).map(String);
    // Case-insensitive lookup maps (lowercase -> original value)
    const localLower = new Map(localArr.map(a => [a.toLowerCase(), a]));
    const upstreamLower = new Map(upstreamArr.map(a => [a.toLowerCase(), a]));
    // Merge keys (deduplicated by lowercase)
    const allKeys = new Set([...localLower.keys(), ...upstreamLower.keys()]);
    const result = [];
    for (const key of allKeys) {
      const inLocal = localLower.has(key);
      const inUpstream = upstreamLower.has(key);
      // Prefer local's casing when both exist
      const value = inLocal ? localLower.get(key) : upstreamLower.get(key);
      let source;
      if (inLocal && inUpstream) {
        source = 'both';
      } else if (inLocal) {
        source = 'local-only';
      } else {
        source = 'upstream-only';
      }
      result.push({ value, source });
    }
    return result;
  }

  // ==================== View Router ====================

  function renderCurrentView(container) {
    const content = container.querySelector('.ss-dashboard-content') || container;
    switch (currentState.view) {
      case 'dashboard':
        renderDashboard(container, content);
        break;
      case 'list':
        renderList(content);
        break;
      case 'detail':
        renderDetail(content);
        break;
      default:
        renderDashboard(container, content);
    }
  }

  // ==================== Plugin Page Injection ====================

  function injectPluginPage() {
    // Check if we're on the plugin page
    const route = SS.getRoute();
    if (route.type !== 'plugin') return;

    // Check if already injected
    if (document.getElementById('ss-recommendations')) {
      console.log('[Stash Sense] Dashboard already injected');
      return;
    }

    // Try multiple selectors for Stash version compatibility
    const containerSelectors = [
      '.PluginRoutes',           // Plugin routes container
      '#root > div > div.main',  // Main content area
      '.main',                   // Fallback main
      '#root > div > div',       // Generic fallback
    ];

    let mainContainer = null;
    for (const selector of containerSelectors) {
      const el = document.querySelector(selector);
      if (el) {
        mainContainer = el;
        console.log(`[Stash Sense] Found container: ${selector}`);
        break;
      }
    }

    if (!mainContainer) {
      // Create floating container as last resort
      console.warn('[Stash Sense] No container found, creating overlay');
      mainContainer = document.createElement('div');
      mainContainer.style.cssText = 'position: fixed; top: 60px; left: 0; right: 0; bottom: 0; overflow-y: auto; background: var(--bs-body-bg, #1a1a1a); z-index: 100;';
      document.body.appendChild(mainContainer);
    }

    // Create and inject our dashboard
    const container = createDashboardContainer();

    // If the container has existing content, replace it
    // For plugin routes, we should be able to append
    if (mainContainer.classList.contains('PluginRoutes')) {
      mainContainer.innerHTML = '';
    }
    mainContainer.appendChild(container);

    // Reset state
    currentState = {
      view: 'dashboard',
      type: null,
      status: 'pending',
      page: 0,
      selectedRec: null,
      counts: null,
    };

    renderCurrentView(container);
  }

  // ==================== Initialization ====================

  function init() {
    // Try to inject if we're already on the plugin page
    setTimeout(injectPluginPage, 300);

    // Watch for navigation to plugin page
    SS.onNavigate((route) => {
      if (route.type === 'plugin') {
        setTimeout(injectPluginPage, 300);
      }
    });

    console.log(`[${SS.PLUGIN_NAME}] Recommendations module loaded`);
  }

  // Export for testing/debugging
  window.StashSenseRecommendations = {
    API: RecommendationsAPI,
    getState: () => currentState,
    init,
  };

  // Initialize
  init();
})();
