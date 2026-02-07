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
  };

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
        <div class="ss-dashboard-header">
          <h1>Stash Sense Recommendations</h1>
          <p class="ss-dashboard-subtitle">Library analysis and curation tools</p>
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

        <div class="ss-dashboard-settings" id="ss-upstream-settings">
          <h2>Upstream Sync Settings</h2>
          <p class="ss-dashboard-subtitle">Configure which stash-box endpoints and fields to monitor for upstream changes.</p>
          <div id="ss-upstream-endpoints-loading">
            <div class="ss-spinner-small"></div> Loading endpoints...
          </div>
          <div id="ss-upstream-endpoints-list" style="display:none"></div>
        </div>
      `;

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

      // Render analysis buttons in desired order
      const analysisButtons = container.querySelector('.ss-analysis-buttons');
      const buttonOrder = ['duplicate_performer', 'duplicate_scenes', 'duplicate_scene_files', 'upstream_performer_changes'];
      const sortedTypes = [...analysisTypes.types].sort((a, b) => {
        const aIdx = buttonOrder.indexOf(a.type);
        const bIdx = buttonOrder.indexOf(b.type);
        return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx);
      });
      for (const analysis of sortedTypes) {
        const btn = SS.createElement('button', {
          className: 'ss-btn ss-btn-secondary ss-analysis-btn',
          innerHTML: `
            <span class="ss-analysis-icon">
              ${typeConfigs[analysis.type]?.icon || ''}
            </span>
            <span>Check ${typeConfigs[analysis.type]?.title || analysis.type}</span>
          `,
        });

        btn.addEventListener('click', async () => {
          btn.disabled = true;
          btn.innerHTML = '<span class="ss-spinner-small"></span> Running...';
          try {
            const result = await RecommendationsAPI.runAnalysis(analysis.type);
            btn.innerHTML = `Started! Run ID: ${result.run_id}`;
            btn.classList.add('ss-btn-success');
            // Refresh counts after a delay
            setTimeout(() => renderDashboard(container), 3000);
          } catch (e) {
            btn.innerHTML = `Failed: ${e.message}`;
            btn.classList.add('ss-btn-error');
            btn.disabled = false;
          }
        });

        analysisButtons.appendChild(btn);
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

      // Load upstream sync settings
      (async () => {
        const loadingEl = container.querySelector('#ss-upstream-endpoints-loading');
        const listEl = container.querySelector('#ss-upstream-endpoints-list');
        try {
          const configResult = await SS.stashQuery(`
            query { configuration { general { stashBoxes { endpoint name } } } }
          `);
          const endpoints = configResult?.configuration?.general?.stashBoxes || [];

          if (endpoints.length === 0) {
            loadingEl.innerHTML = '<p>No stash-box endpoints configured in Stash. Configure them in Stash Settings > Metadata Providers.</p>';
            return;
          }

          loadingEl.style.display = 'none';
          listEl.style.display = 'block';

          for (const ep of endpoints) {
            const epDiv = document.createElement('div');
            epDiv.className = 'ss-upstream-endpoint-settings';
            const displayName = ep.name || new URL(ep.endpoint).hostname;

            epDiv.innerHTML = `
              <div class="ss-upstream-ep-header">
                <h3>${escapeHtml(displayName)}</h3>
                <span class="ss-upstream-ep-url">${escapeHtml(ep.endpoint)}</span>
              </div>
              <div class="ss-upstream-fields-container" style="display:none">
                <div class="ss-upstream-fields-loading"><div class="ss-spinner-small"></div> Loading field config...</div>
                <div class="ss-upstream-fields-grid" style="display:none"></div>
                <div class="ss-upstream-fields-actions" style="display:none; margin-top: 0.75rem;">
                  <button class="ss-btn ss-btn-primary ss-upstream-save-fields" style="padding: 6px 14px; font-size: 0.85rem;">Save Field Config</button>
                  <span class="ss-upstream-save-status" style="margin-left: 8px; font-size: 0.85rem;"></span>
                </div>
              </div>
              <button class="ss-btn ss-btn-secondary ss-upstream-toggle-fields" style="margin-top: 0.5rem; padding: 4px 12px; font-size: 0.8rem;">Show Monitored Fields</button>
            `;

            const toggleBtn = epDiv.querySelector('.ss-upstream-toggle-fields');
            const fieldsContainer = epDiv.querySelector('.ss-upstream-fields-container');
            let fieldsLoaded = false;

            toggleBtn.addEventListener('click', async () => {
              const isHidden = fieldsContainer.style.display === 'none';
              fieldsContainer.style.display = isHidden ? 'block' : 'none';
              toggleBtn.textContent = isHidden ? 'Hide Monitored Fields' : 'Show Monitored Fields';

              if (isHidden && !fieldsLoaded) {
                fieldsLoaded = true;
                try {
                  const fieldConfig = await RecommendationsAPI.getFieldConfig(ep.endpoint);
                  const fieldsGrid = epDiv.querySelector('.ss-upstream-fields-grid');
                  const fieldsLoading = epDiv.querySelector('.ss-upstream-fields-loading');
                  const fieldsActions = epDiv.querySelector('.ss-upstream-fields-actions');

                  const sortedFields = Object.entries(fieldConfig.fields).sort(([, a], [, b]) => a.label.localeCompare(b.label));

                  fieldsGrid.innerHTML = sortedFields.map(([fieldName, config]) => `
                    <label class="ss-upstream-field-toggle">
                      <input type="checkbox" data-field="${escapeHtml(fieldName)}" ${config.enabled ? 'checked' : ''} />
                      <span>${escapeHtml(config.label)}</span>
                    </label>
                  `).join('');

                  fieldsLoading.style.display = 'none';
                  fieldsGrid.style.display = 'grid';
                  fieldsActions.style.display = 'flex';

                  // Save button
                  const saveBtn = epDiv.querySelector('.ss-upstream-save-fields');
                  const saveStatus = epDiv.querySelector('.ss-upstream-save-status');
                  saveBtn.addEventListener('click', async () => {
                    saveBtn.disabled = true;
                    saveStatus.textContent = 'Saving...';
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
                } catch (e) {
                  epDiv.querySelector('.ss-upstream-fields-loading').innerHTML = `Error loading field config: ${escapeHtml(e.message)}`;
                }
              }
            });

            listEl.appendChild(epDiv);
          }
        } catch (e) {
          loadingEl.innerHTML = `<p>Could not load stash-box endpoints: ${escapeHtml(e.message)}</p>`;
        }
      })();

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
      const result = await RecommendationsAPI.getList({
        type: currentState.type,
        status: currentState.status,
        limit: 50,
        offset: currentState.page * 50,
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
      const changeCount = (details.changes || []).length;
      const changedFields = (details.changes || []).map(c => c.field_label).join(', ');

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
      renderUpstreamPerformerDetail(content, rec);
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

  function renderUpstreamPerformerDetail(container, rec) {
    const details = rec.details;
    const changes = details.changes || [];
    const performerId = details.performer_id;

    // Build the header
    const headerHtml = `
      <div class="ss-upstream-header">
        <img src="${details.performer_image_path || ''}" alt="${details.performer_name || ''}" onerror="this.style.display='none'" />
        <div>
          <h2 style="margin: 0 0 4px 0;">
            <a href="/performers/${performerId}" target="_blank">${details.performer_name || 'Unknown'}</a>
          </h2>
          <span class="ss-upstream-endpoint-badge">${details.endpoint_name || 'Upstream'}</span>
        </div>
      </div>
    `;

    // Build field rows
    let fieldRowsHtml = '';
    changes.forEach((change, idx) => {
      const fieldName = `field_${idx}`;
      const mergeType = change.merge_type || 'simple';

      let valuesHtml = '';
      if (mergeType !== 'alias_list') {
        valuesHtml = `
          <div class="ss-upstream-values">
            <div class="ss-upstream-local-value">
              <div class="ss-upstream-value-label">Local</div>
              <div>${escapeHtml(formatFieldValue(change.local_value))}</div>
            </div>
            <div class="ss-upstream-upstream-value">
              <div class="ss-upstream-value-label">Upstream</div>
              <div>${escapeHtml(formatFieldValue(change.upstream_value))}</div>
            </div>
          </div>
        `;
      }

      let controlsHtml = '';

      if (mergeType === 'simple') {
        controlsHtml = `
          <div class="ss-upstream-radio-group" data-field-index="${idx}" data-merge-type="simple" data-field-key="${change.field}">
            <label><input type="radio" name="${fieldName}" value="keep_local" checked /> Keep local value</label>
            <label><input type="radio" name="${fieldName}" value="accept_upstream" /> Accept upstream value</label>
            <label>
              <input type="radio" name="${fieldName}" value="custom" />
              Custom edit
              <input type="text" class="ss-upstream-custom-input" data-custom-for="${fieldName}" placeholder="Enter custom value" style="display:none" />
            </label>
          </div>
        `;
      } else if (mergeType === 'name') {
        controlsHtml = `
          <div class="ss-upstream-radio-group" data-field-index="${idx}" data-merge-type="name" data-field-key="${change.field}">
            <label><input type="radio" name="${fieldName}" value="keep_local" checked /> Keep local name</label>
            <label><input type="radio" name="${fieldName}" value="accept_upstream" /> Accept upstream name</label>
            <label><input type="radio" name="${fieldName}" value="accept_upstream_alias_local" /> Accept upstream name + demote local to alias</label>
            <label><input type="radio" name="${fieldName}" value="keep_local_alias_upstream" /> Keep local name + add upstream as alias</label>
            <label>
              <input type="radio" name="${fieldName}" value="custom" />
              Custom edit
              <input type="text" class="ss-upstream-custom-input" data-custom-for="${fieldName}" placeholder="Enter custom name" style="display:none" />
            </label>
          </div>
        `;
      } else if (mergeType === 'alias_list') {
        const localAliases = change.local_value || [];
        const upstreamAliases = change.upstream_value || [];
        const allAliases = buildAliasList(localAliases, upstreamAliases);

        valuesHtml = ''; // Override - alias_list uses checkbox list instead of value comparison

        controlsHtml = `
          <div class="ss-upstream-alias-list-container" data-field-index="${idx}" data-merge-type="alias_list" data-field-key="${change.field}">
            <div class="ss-upstream-alias-list">
              ${allAliases.map((a, ai) => `
                <label class="ss-upstream-alias-item ${a.source}">
                  <input type="checkbox" name="${fieldName}_alias_${ai}" value="${escapeHtml(a.value)}" ${a.source === 'both' || a.source === 'local-only' ? 'checked' : ''} />
                  <span>${escapeHtml(a.value)}</span>
                  <span class="ss-upstream-alias-tag">${a.source === 'both' ? 'both' : a.source === 'local-only' ? 'local' : 'upstream'}</span>
                </label>
              `).join('')}
            </div>
            <button class="ss-btn ss-btn-secondary ss-upstream-add-alias-btn" type="button" style="margin-top: 0.5rem; padding: 4px 10px; font-size: 0.8rem;">+ Add custom alias</button>
          </div>
        `;
      } else if (mergeType === 'text') {
        controlsHtml = `
          <div class="ss-upstream-radio-group" data-field-index="${idx}" data-merge-type="text" data-field-key="${change.field}">
            <label><input type="radio" name="${fieldName}" value="keep_local" checked /> Keep local value</label>
            <label><input type="radio" name="${fieldName}" value="accept_upstream" /> Accept upstream value</label>
            <label><input type="radio" name="${fieldName}" value="custom" /> Custom edit</label>
            <textarea class="ss-upstream-textarea" data-custom-for="${fieldName}" style="display:none" placeholder="Enter custom text">${escapeHtml(String(change.local_value || ''))}</textarea>
          </div>
        `;
      }

      fieldRowsHtml += `
        <div class="ss-upstream-field-row">
          <div class="ss-upstream-field-label">${escapeHtml(change.field_label || change.field)}</div>
          ${valuesHtml}
          ${controlsHtml}
        </div>
      `;
    });

    // Action bar
    const actionBarHtml = `
      <div id="ss-upstream-validation-errors" class="ss-upstream-validation-error" style="display:none"></div>
      <div class="ss-upstream-action-bar">
        <button class="ss-btn ss-upstream-apply-btn" id="ss-upstream-apply">Apply Selected Changes</button>
        <div class="ss-upstream-dismiss-dropdown" style="position: relative;">
          <button class="ss-btn ss-upstream-dismiss-btn" id="ss-upstream-dismiss-toggle">Dismiss</button>
          <div class="ss-upstream-dismiss-menu" id="ss-upstream-dismiss-menu" style="display:none; position:absolute; bottom:100%; left:0; background:var(--bs-tertiary-bg, #2a2a2a); border:1px solid var(--bs-border-color, #333); border-radius:6px; padding:4px 0; min-width:220px; z-index:10;">
            <button class="ss-upstream-dismiss-option" data-permanent="false" style="display:block; width:100%; text-align:left; padding:8px 12px; background:none; border:none; color:var(--bs-body-color, #fff); cursor:pointer; font-size:0.85rem;">Dismiss this update</button>
            <button class="ss-upstream-dismiss-option" data-permanent="true" style="display:block; width:100%; text-align:left; padding:8px 12px; background:none; border:none; color:var(--bs-body-color, #fff); cursor:pointer; font-size:0.85rem;">Never show for this performer</button>
          </div>
        </div>
      </div>
    `;

    container.innerHTML = `
      <div class="ss-detail-upstream-performer">
        ${headerHtml}
        ${fieldRowsHtml}
        ${actionBarHtml}
      </div>
    `;

    // Wire up radio buttons to show/hide custom inputs
    container.querySelectorAll('.ss-upstream-radio-group').forEach(group => {
      group.addEventListener('change', (e) => {
        if (e.target.type !== 'radio') return;
        const customInput = group.querySelector('.ss-upstream-custom-input') || group.querySelector('.ss-upstream-textarea');
        if (customInput) {
          customInput.style.display = e.target.value === 'custom' ? 'block' : 'none';
          if (e.target.value === 'custom') {
            customInput.focus();
          }
        }
      });
    });

    // Wire up alias "Add custom" buttons
    container.querySelectorAll('.ss-upstream-add-alias-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const aliasContainer = btn.closest('.ss-upstream-alias-list-container');
        const aliasList = aliasContainer.querySelector('.ss-upstream-alias-list');
        const fieldIndex = aliasContainer.dataset.fieldIndex;
        const existingCount = aliasList.querySelectorAll('.ss-upstream-alias-item').length;

        const newAlias = prompt('Enter new alias:');
        if (!newAlias || !newAlias.trim()) return;

        const item = document.createElement('label');
        item.className = 'ss-upstream-alias-item both';
        item.innerHTML = `
          <input type="checkbox" name="field_${fieldIndex}_alias_${existingCount}" value="${escapeHtml(newAlias.trim())}" checked />
          <span>${escapeHtml(newAlias.trim())}</span>
          <span class="ss-upstream-alias-tag">custom</span>
        `;
        aliasList.appendChild(item);
      });
    });

    // Dismiss dropdown toggle
    const dismissToggle = container.querySelector('#ss-upstream-dismiss-toggle');
    const dismissMenu = container.querySelector('#ss-upstream-dismiss-menu');

    dismissToggle.addEventListener('click', () => {
      dismissMenu.style.display = dismissMenu.style.display === 'none' ? 'block' : 'none';
    });

    // Close dismiss menu on outside click
    document.addEventListener('click', function closeDismissMenu(e) {
      if (!dismissToggle.contains(e.target) && !dismissMenu.contains(e.target)) {
        dismissMenu.style.display = 'none';
      }
      // Clean up if container is removed from DOM
      if (!document.contains(container)) {
        document.removeEventListener('click', closeDismissMenu);
      }
    });

    // Dismiss options
    container.querySelectorAll('.ss-upstream-dismiss-option').forEach(option => {
      option.addEventListener('click', async () => {
        const permanent = option.dataset.permanent === 'true';
        dismissMenu.style.display = 'none';
        dismissToggle.disabled = true;
        dismissToggle.textContent = 'Dismissing...';
        try {
          await RecommendationsAPI.dismissUpstream(rec.id, 'User dismissed', permanent);
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        } catch (e) {
          dismissToggle.textContent = `Failed: ${e.message}`;
          dismissToggle.disabled = false;
        }
      });

      option.addEventListener('mouseenter', () => {
        option.style.background = 'rgba(255,255,255,0.05)';
      });
      option.addEventListener('mouseleave', () => {
        option.style.background = 'none';
      });
    });

    // Apply button
    container.querySelector('#ss-upstream-apply').addEventListener('click', async () => {
      const btn = container.querySelector('#ss-upstream-apply');
      const errorDiv = container.querySelector('#ss-upstream-validation-errors');
      errorDiv.style.display = 'none';
      errorDiv.innerHTML = '';
      const fields = {};

      // Collect values from simple, name, and text merge types
      container.querySelectorAll('.ss-upstream-radio-group').forEach(group => {
        const fieldKey = group.dataset.fieldKey;
        const mergeType = group.dataset.mergeType;
        const fieldIndex = group.dataset.fieldIndex;
        const selected = group.querySelector(`input[name="field_${fieldIndex}"]:checked`);
        if (!selected) return;

        const change = changes[parseInt(fieldIndex)];
        const choice = selected.value;

        if (choice === 'keep_local') {
          return;
        } else if (choice === 'accept_upstream') {
          fields[fieldKey] = change.upstream_value;
        } else if (choice === 'accept_upstream_alias_local') {
          fields[fieldKey] = change.upstream_value;
          fields['_alias_add'] = fields['_alias_add'] || [];
          fields['_alias_add'].push(String(change.local_value || ''));
        } else if (choice === 'keep_local_alias_upstream') {
          fields['_alias_add'] = fields['_alias_add'] || [];
          fields['_alias_add'].push(String(change.upstream_value || ''));
        } else if (choice === 'custom') {
          const customInput = group.querySelector('.ss-upstream-custom-input') || group.querySelector('.ss-upstream-textarea');
          if (customInput && customInput.value.trim()) {
            fields[fieldKey] = customInput.value.trim();
          }
        }
      });

      // Collect values from alias_list merge types
      container.querySelectorAll('.ss-upstream-alias-list-container').forEach(aliasContainer => {
        const fieldKey = aliasContainer.dataset.fieldKey;
        const checkedAliases = [];
        aliasContainer.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
          checkedAliases.push(cb.value);
        });
        fields[fieldKey] = checkedAliases;
      });

      // Check if any changes were selected
      const hasChanges = Object.keys(fields).length > 0;
      if (!hasChanges) {
        try {
          btn.disabled = true;
          btn.textContent = 'Resolving...';
          await RecommendationsAPI.resolve(rec.id, 'accepted_no_changes', {});
          btn.textContent = 'Done!';
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
        return;
      }

      // Run validation before applying
      btn.disabled = true;
      btn.textContent = 'Validating...';

      const proposedName = fields.name || details.performer_name;
      const proposedDisambig = fields.disambiguation || null;
      const proposedAliases = fields.aliases || fields._alias_add || [];

      const validationErrors = await validatePerformerMerge(
        performerId, proposedName, proposedDisambig, proposedAliases
      );

      if (validationErrors.length > 0) {
        errorDiv.innerHTML = validationErrors.map(e => `<div>${escapeHtml(e)}</div>`).join('');
        errorDiv.style.display = 'block';
        btn.textContent = 'Apply Selected Changes';
        btn.disabled = false;
        return;
      }

      try {
        btn.textContent = 'Applying...';
        await RecommendationsAPI.updatePerformer(performerId, fields);
        await RecommendationsAPI.resolve(rec.id, 'applied', { fields });

        btn.textContent = 'Applied!';
        btn.classList.add('ss-btn-success');

        setTimeout(() => {
          currentState.view = 'list';
          currentState.selectedRec = null;
          renderCurrentView(document.getElementById('ss-recommendations'));
        }, 1500);
      } catch (e) {
        // Server-side error handling
        let errorMsg = e.message;
        if (errorMsg.includes('already exists') || errorMsg.includes('name')) {
          errorMsg = `Name conflict: ${errorMsg}. Try adding a disambiguation.`;
        } else if (errorMsg.includes('duplicate') || errorMsg.includes('alias')) {
          errorMsg = `Alias conflict: ${errorMsg}. Try removing duplicate aliases.`;
        }
        errorDiv.innerHTML = `<div>${escapeHtml(errorMsg)}</div>`;
        errorDiv.style.display = 'block';
        btn.textContent = 'Apply Selected Changes';
        btn.classList.remove('ss-btn-error');
        btn.disabled = false;
      }
    });
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
