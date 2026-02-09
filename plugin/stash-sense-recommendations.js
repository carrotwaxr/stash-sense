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
      return apiCall('fp_generate', {
        refresh_outdated: options.refreshOutdated ?? true,
        num_frames: options.numFrames ?? 12,
        min_face_size: options.minFaceSize ?? 50,
        max_distance: options.maxDistance ?? 0.6,
      });
    },

    async getFingerprintProgress() {
      return apiCall('fp_progress');
    },

    async stopFingerprintGeneration() {
      return apiCall('fp_stop');
    },

    // Upstream performer sync operations
    async updatePerformer(performerId, fields) {
      return apiCall('rec_update_performer', { performer_id: performerId, fields });
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

  // ==================== Dashboard Container ====================

  function createDashboardContainer() {
    const existing = document.getElementById('ss-recommendations');
    if (existing) existing.remove();

    const container = SS.createElement('div', {
      id: 'ss-recommendations',
      className: 'ss-recommendations',
    });

    return container;
  }

  // ==================== Dashboard View ====================

  async function renderDashboard(container) {
    container.innerHTML = `
      <div class="ss-dashboard-header">
        <h1>Stash Sense Recommendations</h1>
        <p class="ss-dashboard-subtitle">Library analysis and curation tools</p>
      </div>
      <div class="ss-dashboard-loading">
        <div class="ss-spinner"></div>
        <p>Loading recommendations...</p>
      </div>
    `;

    try {
      const [counts, sidecarStatus, analysisTypes, fpStatus] = await Promise.all([
        RecommendationsAPI.getCounts(),
        RecommendationsAPI.getSidecarStatus(),
        RecommendationsAPI.getAnalysisTypes(),
        RecommendationsAPI.getFingerprintStatus(),
      ]);

      currentState.counts = counts;

      // Build fingerprint status display
      const fpRunning = fpStatus.generation_running;
      const fpProgress = fpStatus.generation_progress || {};
      const fpCoverage = fpStatus.complete_fingerprints || 0;
      const fpNeedsRefresh = fpStatus.needs_refresh_count || 0;

      container.innerHTML = `
        <div class="ss-dashboard-header" style="display:flex;align-items:center;justify-content:space-between;">
          <div>
            <h1>Stash Sense Recommendations</h1>
            <p class="ss-dashboard-subtitle">Library analysis and curation tools</p>
          </div>
          <div id="ss-dashboard-gear"></div>
        </div>

        <div class="ss-stash-status ${sidecarStatus.connected ? 'connected' : 'disconnected'}">
          <span class="ss-status-dot"></span>
          <span>Stash Sense: ${sidecarStatus.connected ? 'Connected' : 'Disconnected'}</span>
          ${sidecarStatus.url ? `<span class="ss-status-url">(${sidecarStatus.url})</span>` : ''}
          ${sidecarStatus.error ? `<span class="ss-status-error">${sidecarStatus.error}</span>` : ''}
        </div>

        <div class="ss-dashboard-summary">
          <div class="ss-summary-card ss-summary-total">
            <div class="ss-summary-number">${counts.total_pending}</div>
            <div class="ss-summary-label">Pending Recommendations</div>
          </div>
        </div>

        <div class="ss-fingerprint-section">
          <h2>Scene Fingerprints</h2>
          <p class="ss-fingerprint-desc">Fingerprints enable face-based duplicate detection. Generate them for your library to improve accuracy.</p>

          <div class="ss-fingerprint-stats">
            <div class="ss-fp-stat">
              <span class="ss-fp-stat-value">${fpCoverage}</span>
              <span class="ss-fp-stat-label">Fingerprints</span>
            </div>
            <div class="ss-fp-stat">
              <span class="ss-fp-stat-value">${fpStatus.current_db_version || 'N/A'}</span>
              <span class="ss-fp-stat-label">DB Version</span>
            </div>
            ${fpNeedsRefresh > 0 ? `
            <div class="ss-fp-stat ss-fp-stat-warning">
              <span class="ss-fp-stat-value">${fpNeedsRefresh}</span>
              <span class="ss-fp-stat-label">Need Refresh</span>
            </div>
            ` : ''}
          </div>

          <div class="ss-fingerprint-progress" id="ss-fp-progress" style="display: ${fpRunning ? 'block' : 'none'}">
            <div class="ss-progress-info">
              <span class="ss-progress-text">
                ${fpProgress.current_scene_title || 'Processing...'}
              </span>
              <span class="ss-progress-numbers">
                ${fpProgress.processed_scenes || 0} / ${fpProgress.total_scenes || 0}
              </span>
            </div>
            <div class="ss-progress-bar-container">
              <div class="ss-progress-bar" style="width: ${fpProgress.progress_pct || 0}%">
                <span class="ss-progress-pct">${Math.round(fpProgress.progress_pct || 0)}%</span>
              </div>
            </div>
            <div class="ss-progress-stats">
              <span class="ss-progress-stat ss-stat-success">${fpProgress.successful || 0} done</span>
              <span class="ss-progress-stat ss-stat-skip">${fpProgress.skipped || 0} skipped</span>
              <span class="ss-progress-stat ss-stat-fail">${fpProgress.failed || 0} failed</span>
            </div>
            <div class="ss-progress-refresh" id="ss-fp-refresh">
              <span class="ss-refresh-text">Refreshing in <span class="ss-refresh-countdown">30</span>s</span>
            </div>
          </div>

          <div class="ss-fingerprint-actions">
            <button class="ss-btn ${fpRunning ? 'ss-btn-danger' : 'ss-btn-primary'}" id="ss-fp-action-btn">
              ${fpRunning ? 'Stop Generation' : 'Generate Fingerprints'}
            </button>
          </div>
        </div>

        <div class="ss-dashboard-types">
          <h2>Recommendation Types</h2>
          <div class="ss-type-cards"></div>
        </div>

        <div class="ss-dashboard-actions">
          <h2>Action Runner</h2>
          <div class="ss-analysis-buttons"></div>
        </div>
      `

      // Add gear icon to dashboard header
      const gearSlot = container.querySelector('#ss-dashboard-gear');
      if (gearSlot) {
        gearSlot.appendChild(createSettingsGearButton(() => showSettingsModal()));
      }

      // Render type cards
      const typeCards = container.querySelector('.ss-type-cards');
      const typeConfigs = {
        duplicate_performer: {
          title: 'Duplicate Performers',
          icon: `<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>`,
          description: 'Performers sharing the same StashDB ID',
        },
        duplicate_scenes: {
          title: 'Duplicate Scenes',
          icon: `<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12zm-6-1l4-4-1.4-1.4-1.6 1.6V6h-2v6.2l-1.6-1.6L10 12l4 4z"/></svg>`,
          description: 'Scenes that may be duplicates based on stash-box ID, faces, or metadata',
        },
        duplicate_scene_files: {
          title: 'Duplicate Scene Files',
          icon: `<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-8 12.5v-9l6 4.5-6 4.5z"/></svg>`,
          description: 'Scenes with multiple files attached',
        },
        upstream_performer_changes: {
          title: 'Upstream Performer Changes',
          icon: `<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M12 6V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/></svg>`,
          description: 'Performer fields updated on StashDB since last sync',
        },
      };

      for (const [type, typeCounts] of Object.entries(counts.counts)) {
        const config = typeConfigs[type] || { title: type, icon: '', description: '' };
        const pending = typeCounts.pending || 0;
        const resolved = typeCounts.resolved || 0;
        const dismissed = typeCounts.dismissed || 0;

        const card = SS.createElement('div', {
          className: 'ss-type-card',
          innerHTML: `
            <div class="ss-type-icon">${config.icon}</div>
            <div class="ss-type-info">
              <h3>${config.title}</h3>
              <p>${config.description}</p>
              <div class="ss-type-counts">
                <span class="ss-count-pending">${pending} pending</span>
                <span class="ss-count-resolved">${resolved} resolved</span>
                <span class="ss-count-dismissed">${dismissed} dismissed</span>
              </div>
            </div>
            <button class="ss-btn ss-btn-primary" data-type="${type}">
              View All
            </button>
          `,
        });

        card.querySelector('button').addEventListener('click', () => {
          currentState.type = type;
          currentState.view = 'list';
          renderCurrentView(container);
        });

        typeCards.appendChild(card);
      }

      // Render analysis buttons with progress support
      const analysisButtons = container.querySelector('.ss-analysis-buttons');
      const buttonOrder = ['duplicate_performer', 'duplicate_scenes', 'duplicate_scene_files', 'upstream_performer_changes'];
      const sortedTypes = [...analysisTypes.types].sort((a, b) => {
        const aIdx = buttonOrder.indexOf(a.type);
        const bIdx = buttonOrder.indexOf(b.type);
        return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx);
      });

      // Track polling intervals for analysis progress
      const analysisPollers = {};

      function renderAnalysisProgress(cardEl, run) {
        const processed = run.items_processed || 0;
        const total = run.items_total || 0;
        const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
        const label = total > 0
          ? `Processing ${processed} / ${total} items...`
          : 'Starting analysis...';

        cardEl.querySelector('.ss-analysis-progress').innerHTML = `
          <div class="ss-progress-bar-container" style="margin:8px 0 4px;">
            <div class="ss-progress-bar" style="width:${pct}%;">
              <span class="ss-progress-pct">${pct}%</span>
            </div>
          </div>
          <div style="font-size:12px;color:var(--ss-text-secondary,#aaa);">${label}</div>
        `;
      }

      function startAnalysisPolling(type, runId, cardEl) {
        if (analysisPollers[type]) clearInterval(analysisPollers[type]);
        analysisPollers[type] = setInterval(async () => {
          try {
            const runs = await RecommendationsAPI.getAnalysisRuns(type, 1);
            if (!runs || !runs.length) return;
            const run = runs[0];
            if (run.status === 'running') {
              renderAnalysisProgress(cardEl, run);
            } else {
              // Completed or failed
              clearInterval(analysisPollers[type]);
              delete analysisPollers[type];
              const progressEl = cardEl.querySelector('.ss-analysis-progress');
              if (run.status === 'completed') {
                progressEl.innerHTML = `
                  <div style="font-size:13px;color:var(--ss-color-success,#4caf50);padding:4px 0;">
                    Completed \u2014 ${run.recommendations_created} recommendation${run.recommendations_created !== 1 ? 's' : ''} created
                  </div>
                `;
              } else {
                progressEl.innerHTML = `
                  <div style="font-size:13px;color:var(--ss-color-error,#f44336);padding:4px 0;">
                    Failed: ${run.error_message || 'Unknown error'}
                  </div>
                `;
              }
              // Refresh dashboard after a brief pause
              setTimeout(() => renderDashboard(container), 3000);
            }
          } catch (e) {
            console.error('[Stash Sense] Error polling analysis progress:', e);
          }
        }, 5000);
      }

      // Fetch latest run for each type to detect already-running analyses
      const latestRuns = {};
      await Promise.all(sortedTypes.map(async (analysis) => {
        try {
          const runs = await RecommendationsAPI.getAnalysisRuns(analysis.type, 1);
          if (runs && runs.length) latestRuns[analysis.type] = runs[0];
        } catch (_) { /* ignore */ }
      }));

      for (const analysis of sortedTypes) {
        const latestRun = latestRuns[analysis.type];
        const isRunning = latestRun && latestRun.status === 'running';

        const card = SS.createElement('div', {
          className: 'ss-analysis-card',
          innerHTML: `
            <button class="ss-btn ss-btn-secondary ss-analysis-btn" ${isRunning ? 'disabled' : ''}>
              <span class="ss-analysis-icon">
                ${typeConfigs[analysis.type]?.icon || ''}
              </span>
              <span>Check ${typeConfigs[analysis.type]?.title || analysis.type}</span>
            </button>
            <div class="ss-analysis-progress"></div>
          `,
        });

        const btn = card.querySelector('.ss-analysis-btn');

        // If already running, show progress and start polling
        if (isRunning) {
          renderAnalysisProgress(card, latestRun);
          startAnalysisPolling(analysis.type, latestRun.id, card);
        }

        btn.addEventListener('click', async () => {
          btn.disabled = true;
          btn.innerHTML = '<span class="ss-spinner-small"></span> Starting...';
          try {
            const result = await RecommendationsAPI.runAnalysis(analysis.type);
            btn.innerHTML = `
              <span class="ss-analysis-icon">
                ${typeConfigs[analysis.type]?.icon || ''}
              </span>
              <span>Check ${typeConfigs[analysis.type]?.title || analysis.type}</span>
            `;
            renderAnalysisProgress(card, { items_processed: 0, items_total: 0 });
            startAnalysisPolling(analysis.type, result.run_id, card);
          } catch (e) {
            btn.innerHTML = `Failed: ${e.message}`;
            btn.classList.add('ss-btn-error');
            btn.disabled = false;
          }
        });

        analysisButtons.appendChild(card);
      }

      // Fingerprint generation button handler
      const fpActionBtn = container.querySelector('#ss-fp-action-btn');
      const fpProgressEl = container.querySelector('#ss-fp-progress');
      const POLL_INTERVAL = 30; // seconds
      let fpPollTimeout = null;
      let fpCountdownInterval = null;
      let fpCountdownValue = POLL_INTERVAL;

      function startCountdown() {
        fpCountdownValue = POLL_INTERVAL;
        const countdownEl = fpProgressEl.querySelector('.ss-refresh-countdown');
        const refreshEl = fpProgressEl.querySelector('#ss-fp-refresh');
        if (refreshEl) refreshEl.style.display = 'block';

        if (fpCountdownInterval) clearInterval(fpCountdownInterval);
        fpCountdownInterval = setInterval(() => {
          fpCountdownValue--;
          if (countdownEl) countdownEl.textContent = fpCountdownValue;
          if (fpCountdownValue <= 0) {
            clearInterval(fpCountdownInterval);
          }
        }, 1000);
      }

      function stopPolling() {
        if (fpPollTimeout) {
          clearTimeout(fpPollTimeout);
          fpPollTimeout = null;
        }
        if (fpCountdownInterval) {
          clearInterval(fpCountdownInterval);
          fpCountdownInterval = null;
        }
        const refreshEl = fpProgressEl.querySelector('#ss-fp-refresh');
        if (refreshEl) refreshEl.style.display = 'none';
      }

      async function updateFingerprintProgress() {
        try {
          const progress = await RecommendationsAPI.getFingerprintProgress();
          const pct = Math.round(progress.progress_pct || 0);

          if (progress.status === 'running' || progress.status === 'stopping') {
            fpProgressEl.style.display = 'block';
            fpProgressEl.querySelector('.ss-progress-text').textContent =
              progress.current_scene_title || 'Processing...';
            fpProgressEl.querySelector('.ss-progress-numbers').textContent =
              `${progress.processed_scenes || 0} / ${progress.total_scenes || 0}`;
            fpProgressEl.querySelector('.ss-progress-bar').style.width = `${pct}%`;
            const pctEl = fpProgressEl.querySelector('.ss-progress-pct');
            if (pctEl) pctEl.textContent = `${pct}%`;
            fpProgressEl.querySelector('.ss-stat-success').textContent =
              `${progress.successful || 0} done`;
            fpProgressEl.querySelector('.ss-stat-skip').textContent =
              `${progress.skipped || 0} skipped`;
            fpProgressEl.querySelector('.ss-stat-fail').textContent =
              `${progress.failed || 0} failed`;

            if (progress.status === 'stopping') {
              fpActionBtn.textContent = 'Stopping...';
              fpActionBtn.disabled = true;
              stopPolling();
            } else {
              // Schedule next poll with countdown
              startCountdown();
              fpPollTimeout = setTimeout(updateFingerprintProgress, POLL_INTERVAL * 1000);
            }
          } else {
            // Generation finished
            stopPolling();
            fpActionBtn.textContent = 'Generate Fingerprints';
            fpActionBtn.className = 'ss-btn ss-btn-primary';
            fpActionBtn.disabled = false;

            if (progress.status === 'completed') {
              fpProgressEl.querySelector('.ss-progress-text').textContent = 'Complete!';
              fpProgressEl.querySelector('.ss-progress-bar').style.width = '100%';
              const pctEl = fpProgressEl.querySelector('.ss-progress-pct');
              if (pctEl) pctEl.textContent = '100%';
            } else if (progress.status === 'paused') {
              fpProgressEl.querySelector('.ss-progress-text').textContent = 'Paused - can resume';
            }
          }
        } catch (e) {
          console.error('[Stash Sense] Error polling fingerprint progress:', e);
          // Retry after interval even on error
          startCountdown();
          fpPollTimeout = setTimeout(updateFingerprintProgress, POLL_INTERVAL * 1000);
        }
      }

      // Start polling if already running
      if (fpStatus.generation_running) {
        updateFingerprintProgress();
      }

      fpActionBtn.addEventListener('click', async () => {
        const isRunning = fpActionBtn.textContent.includes('Stop');

        if (isRunning) {
          // Stop generation
          fpActionBtn.disabled = true;
          fpActionBtn.textContent = 'Stopping...';
          stopPolling();
          try {
            await RecommendationsAPI.stopFingerprintGeneration();
          } catch (e) {
            console.error('[Stash Sense] Error stopping generation:', e);
          }
        } else {
          // Start generation
          fpActionBtn.disabled = true;
          fpActionBtn.textContent = 'Starting...';
          try {
            await RecommendationsAPI.startFingerprintGeneration({ refreshOutdated: true });
            fpActionBtn.textContent = 'Stop Generation';
            fpActionBtn.className = 'ss-btn ss-btn-danger';
            fpActionBtn.disabled = false;
            fpProgressEl.style.display = 'block';

            // Start polling with countdown
            stopPolling();
            updateFingerprintProgress();
          } catch (e) {
            fpActionBtn.textContent = 'Generate Fingerprints';
            fpActionBtn.className = 'ss-btn ss-btn-primary';
            fpActionBtn.disabled = false;
            console.error('[Stash Sense] Error starting generation:', e);
            showConfirmModal('Failed to start fingerprint generation: ' + e.message, () => {});
          }
        }
      });

    } catch (e) {
      container.innerHTML = `
        <div class="ss-dashboard-header">
          <h1>Stash Sense Recommendations</h1>
        </div>
        <div class="ss-error-state">
          <div class="ss-error-icon">
            <svg viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>
          </div>
          <h2>Connection Error</h2>
          <p>${e.message}</p>
          <p class="ss-error-hint">Make sure the Stash Sense sidecar is running and configured correctly.</p>
          <button class="ss-btn ss-btn-primary" onclick="location.reload()">Retry</button>
        </div>
      `;
    }
  }

  // ==================== List View ====================

  async function renderList(container) {
    const typeConfigs = {
      duplicate_performer: 'Duplicate Performers',
      duplicate_scene_files: 'Duplicate Scene Files',
      upstream_performer_changes: 'Upstream Performer Changes',
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

    // Load recommendations
    try {
      const PAGE_SIZE = 25;
      const result = await RecommendationsAPI.getList({
        type: currentState.type,
        status: currentState.status,
        limit: PAGE_SIZE,
        offset: currentState.page * PAGE_SIZE,
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
    } else {
      content.innerHTML = `<p>Unknown recommendation type: ${rec.type}</p>`;
    }
  }

  function renderDuplicatePerformerDetail(container, rec) {
    const details = rec.details;
    const performers = details.performers || [];

    container.innerHTML = `
      <div class="ss-detail-duplicate-performer">
        <h2>Duplicate Performers</h2>
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
        <h2>${details.scene_title}</h2>
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
  }

  // ==================== Settings Modal ====================

  function createGearIcon() {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '20');
    svg.setAttribute('height', '20');
    svg.setAttribute('fill', 'currentColor');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.488.488 0 00-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 00-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03 1.58a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z');
    svg.appendChild(path);
    return svg;
  }

  function createSettingsGearButton(onClick) {
    const btn = document.createElement('button');
    btn.style.cssText = 'background:none;border:none;color:var(--bs-secondary-color,#888);cursor:pointer;padding:4px;display:flex;align-items:center;transition:color 0.15s;';
    btn.title = 'Settings';
    btn.appendChild(createGearIcon());
    btn.addEventListener('mouseenter', () => { btn.style.color = 'var(--bs-body-color, #fff)'; });
    btn.addEventListener('mouseleave', () => { btn.style.color = 'var(--bs-secondary-color, #888)'; });
    btn.addEventListener('click', onClick);
    return btn;
  }

  async function showSettingsModal() {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:10000;';

    const modal = document.createElement('div');
    modal.style.cssText = 'background:#2a2a2a;border:1px solid #444;border-radius:10px;max-width:600px;width:90%;max-height:80vh;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.4);';

    // Header
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:1rem 1.25rem;border-bottom:1px solid #444;';
    const title = document.createElement('h3');
    title.style.cssText = 'margin:0;font-size:1.1rem;font-weight:600;color:#fff;';
    title.textContent = 'Settings';
    const closeBtn = document.createElement('button');
    closeBtn.style.cssText = 'background:none;border:none;font-size:1.5rem;color:#888;cursor:pointer;padding:0;line-height:1;';
    closeBtn.textContent = '\u00d7';
    closeBtn.addEventListener('click', () => overlay.remove());
    header.appendChild(title);
    header.appendChild(closeBtn);

    // Tabs
    const tabBar = document.createElement('div');
    tabBar.style.cssText = 'display:flex;gap:0;border-bottom:1px solid #444;padding:0 1.25rem;';

    const tabs = ['General', 'Upstream Sync'];
    const tabBtns = [];
    const tabPanels = [];

    tabs.forEach((tabName, i) => {
      const btn = document.createElement('button');
      btn.style.cssText = 'padding:0.6rem 1rem;background:none;border:none;border-bottom:2px solid transparent;color:#888;cursor:pointer;font-size:0.9rem;transition:color 0.15s,border-color 0.15s;';
      btn.textContent = tabName;
      if (i === 0) {
        btn.style.color = '#fff';
        btn.style.borderBottomColor = '#0d6efd';
      }
      btn.addEventListener('click', () => {
        tabBtns.forEach((b, j) => {
          b.style.color = j === i ? '#fff' : '#888';
          b.style.borderBottomColor = j === i ? '#0d6efd' : 'transparent';
        });
        tabPanels.forEach((p, j) => {
          p.style.display = j === i ? 'block' : 'none';
        });
      });
      tabBar.appendChild(btn);
      tabBtns.push(btn);
    });

    // Body (scrollable)
    const body = document.createElement('div');
    body.style.cssText = 'padding:1.25rem;overflow-y:auto;flex:1;';

    // === General Tab ===
    const generalPanel = document.createElement('div');
    generalPanel.style.cssText = 'display:block;';

    const enumToggleContainer = document.createElement('div');
    enumToggleContainer.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:0.75rem;background:#333;border-radius:6px;';

    const enumLabel = document.createElement('div');
    enumLabel.style.cssText = 'flex:1;';
    const enumTitle = document.createElement('div');
    enumTitle.style.cssText = 'font-size:0.9rem;font-weight:600;color:#fff;margin-bottom:2px;';
    enumTitle.textContent = 'Normalize Enum Display';
    const enumDesc = document.createElement('div');
    enumDesc.style.cssText = 'font-size:0.8rem;color:#888;';
    enumDesc.textContent = 'Show ALL_CAPS values as Title Case (e.g. BROWN \u2192 Brown)';
    enumLabel.appendChild(enumTitle);
    enumLabel.appendChild(enumDesc);

    const enumCheckbox = document.createElement('input');
    enumCheckbox.type = 'checkbox';
    enumCheckbox.style.cssText = 'width:18px;height:18px;accent-color:#0d6efd;cursor:pointer;';

    enumToggleContainer.appendChild(enumLabel);
    enumToggleContainer.appendChild(enumCheckbox);
    generalPanel.appendChild(enumToggleContainer);

    // Load current setting
    try {
      const val = await RecommendationsAPI.getUserSetting('normalize_enum_display');
      enumCheckbox.checked = val !== false; // default to true
    } catch (_) {
      enumCheckbox.checked = true;
    }

    // Save on toggle
    enumCheckbox.addEventListener('change', async () => {
      try {
        await RecommendationsAPI.setUserSetting('normalize_enum_display', enumCheckbox.checked);
      } catch (e) {
        console.error('[Stash Sense] Failed to save setting:', e);
      }
    });

    tabPanels.push(generalPanel);

    // === Upstream Sync Tab ===
    const upstreamPanel = document.createElement('div');
    upstreamPanel.style.cssText = 'display:none;';
    tabPanels.push(upstreamPanel);

    // Render upstream sync settings into the panel
    renderUpstreamSyncSettingsTab(upstreamPanel);

    body.appendChild(generalPanel);
    body.appendChild(upstreamPanel);

    modal.appendChild(header);
    modal.appendChild(tabBar);
    modal.appendChild(body);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // Close on backdrop click
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });
  }

  async function renderUpstreamSyncSettingsTab(container) {
    const loadingDiv = document.createElement('div');
    loadingDiv.style.cssText = 'padding:1rem;color:#888;font-size:0.9rem;';
    loadingDiv.textContent = 'Loading stash-box endpoints...';
    container.appendChild(loadingDiv);

    try {
      const configResult = await SS.stashQuery(`
        query { configuration { general { stashBoxes { endpoint name } } } }
      `);
      const endpoints = configResult?.configuration?.general?.stashBoxes || [];

      if (endpoints.length === 0) {
        loadingDiv.textContent = 'No stash-box endpoints configured in Stash. Configure them in Stash Settings > Metadata Providers.';
        return;
      }

      loadingDiv.remove();

      for (const ep of endpoints) {
        const epDiv = document.createElement('div');
        epDiv.style.cssText = 'background:#333;border:1px solid #444;border-radius:8px;padding:1rem;margin-bottom:0.75rem;';
        const displayName = ep.name || new URL(ep.endpoint).hostname;

        // Header
        const epHeader = document.createElement('div');
        const epTitle = document.createElement('h4');
        epTitle.style.cssText = 'margin:0 0 4px 0;font-size:0.95rem;color:#fff;';
        epTitle.textContent = displayName;
        const epUrl = document.createElement('div');
        epUrl.style.cssText = 'font-size:0.8rem;color:#888;margin-bottom:0.5rem;';
        epUrl.textContent = ep.endpoint;
        epHeader.appendChild(epTitle);
        epHeader.appendChild(epUrl);
        epDiv.appendChild(epHeader);

        // Fields container (hidden initially)
        const fieldsWrapper = document.createElement('div');
        fieldsWrapper.style.cssText = 'display:none;';
        epDiv.appendChild(fieldsWrapper);

        // Toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.style.cssText = 'padding:4px 12px;border-radius:4px;border:1px solid #555;background:#2a2a2a;color:#fff;font-size:0.8rem;cursor:pointer;';
        toggleBtn.textContent = 'Show Monitored Fields';
        let fieldsLoaded = false;

        toggleBtn.addEventListener('click', async () => {
          const isHidden = fieldsWrapper.style.display === 'none';
          fieldsWrapper.style.display = isHidden ? 'block' : 'none';
          toggleBtn.textContent = isHidden ? 'Hide Monitored Fields' : 'Show Monitored Fields';

          if (isHidden && !fieldsLoaded) {
            fieldsLoaded = true;
            const fieldsLoading = document.createElement('div');
            fieldsLoading.style.cssText = 'padding:0.5rem;color:#888;font-size:0.85rem;';
            fieldsLoading.textContent = 'Loading field config...';
            fieldsWrapper.appendChild(fieldsLoading);

            try {
              const fieldConfig = await RecommendationsAPI.getFieldConfig(ep.endpoint);
              fieldsLoading.remove();

              const fieldsGrid = document.createElement('div');
              fieldsGrid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px;';

              const sortedFields = Object.entries(fieldConfig.fields).sort(([, a], [, b]) => a.label.localeCompare(b.label));

              for (const [fieldName, config] of sortedFields) {
                const label = document.createElement('label');
                label.style.cssText = 'display:flex;align-items:center;gap:6px;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:0.85rem;color:#fff;';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.dataset.field = fieldName;
                cb.checked = config.enabled;
                label.appendChild(cb);
                label.appendChild(document.createTextNode(config.label));
                label.addEventListener('mouseenter', () => { label.style.background = 'rgba(255,255,255,0.05)'; });
                label.addEventListener('mouseleave', () => { label.style.background = 'none'; });
                fieldsGrid.appendChild(label);
              }
              fieldsWrapper.appendChild(fieldsGrid);

              // Save button
              const actionsDiv = document.createElement('div');
              actionsDiv.style.cssText = 'display:flex;align-items:center;margin-top:0.75rem;';
              const saveBtn = document.createElement('button');
              saveBtn.style.cssText = 'padding:6px 14px;border-radius:6px;border:none;background:#0d6efd;color:white;font-size:0.85rem;cursor:pointer;';
              saveBtn.textContent = 'Save Field Config';
              const saveStatus = document.createElement('span');
              saveStatus.style.cssText = 'margin-left:8px;font-size:0.85rem;';

              saveBtn.addEventListener('click', async () => {
                saveBtn.disabled = true;
                saveStatus.textContent = 'Saving...';
                saveStatus.style.color = '#888';
                const configs = {};
                fieldsGrid.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                  configs[cb.dataset.field] = cb.checked;
                });
                try {
                  await RecommendationsAPI.setFieldConfig(ep.endpoint, configs);
                  saveStatus.textContent = 'Saved!';
                  saveStatus.style.color = '#22c55e';
                  setTimeout(() => { saveStatus.textContent = ''; }, 2000);
                } catch (e) {
                  saveStatus.textContent = `Error: ${e.message}`;
                  saveStatus.style.color = '#ef4444';
                }
                saveBtn.disabled = false;
              });

              actionsDiv.appendChild(saveBtn);
              actionsDiv.appendChild(saveStatus);
              fieldsWrapper.appendChild(actionsDiv);

            } catch (e) {
              fieldsLoading.textContent = `Error loading field config: ${e.message}`;
            }
          }
        });

        epDiv.appendChild(toggleBtn);
        container.appendChild(epDiv);
      }
    } catch (e) {
      loadingDiv.textContent = `Could not load stash-box endpoints: ${e.message}`;
    }
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
    headerDiv.appendChild(createSettingsGearButton(() => showSettingsModal()));
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
      aliasCb.checked = false;
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
    switch (currentState.view) {
      case 'dashboard':
        renderDashboard(container);
        break;
      case 'list':
        renderList(container);
        break;
      case 'detail':
        renderDetail(container);
        break;
      default:
        renderDashboard(container);
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
