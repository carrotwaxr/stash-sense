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
      const [counts, stashStatus, analysisTypes, fpStatus] = await Promise.all([
        RecommendationsAPI.getCounts(),
        RecommendationsAPI.getStashStatus(),
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

        <div class="ss-stash-status ${stashStatus.connected ? 'connected' : 'disconnected'}">
          <span class="ss-status-dot"></span>
          <span>Stash: ${stashStatus.connected ? 'Connected' : 'Disconnected'}</span>
          ${stashStatus.url ? `<span class="ss-status-url">(${stashStatus.url})</span>` : ''}
          ${stashStatus.error ? `<span class="ss-status-error">${stashStatus.error}</span>` : ''}
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
              <div class="ss-progress-bar" style="width: ${fpProgress.progress_pct || 0}%"></div>
            </div>
            <div class="ss-progress-stats">
              <span class="ss-progress-stat ss-stat-success">${fpProgress.successful || 0} done</span>
              <span class="ss-progress-stat ss-stat-skip">${fpProgress.skipped || 0} skipped</span>
              <span class="ss-progress-stat ss-stat-fail">${fpProgress.failed || 0} failed</span>
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
      const buttonOrder = ['duplicate_performer', 'duplicate_scenes', 'duplicate_scene_files'];
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
      let fpPollInterval = null;

      async function updateFingerprintProgress() {
        try {
          const progress = await RecommendationsAPI.getFingerprintProgress();

          if (progress.status === 'running' || progress.status === 'stopping') {
            fpProgressEl.style.display = 'block';
            fpProgressEl.querySelector('.ss-progress-text').textContent =
              progress.current_scene_title || 'Processing...';
            fpProgressEl.querySelector('.ss-progress-numbers').textContent =
              `${progress.processed_scenes || 0} / ${progress.total_scenes || 0}`;
            fpProgressEl.querySelector('.ss-progress-bar').style.width =
              `${progress.progress_pct || 0}%`;
            fpProgressEl.querySelector('.ss-stat-success').textContent =
              `${progress.successful || 0} done`;
            fpProgressEl.querySelector('.ss-stat-skip').textContent =
              `${progress.skipped || 0} skipped`;
            fpProgressEl.querySelector('.ss-stat-fail').textContent =
              `${progress.failed || 0} failed`;

            if (progress.status === 'stopping') {
              fpActionBtn.textContent = 'Stopping...';
              fpActionBtn.disabled = true;
            }
          } else {
            // Generation finished
            clearInterval(fpPollInterval);
            fpPollInterval = null;
            fpActionBtn.textContent = 'Generate Fingerprints';
            fpActionBtn.className = 'ss-btn ss-btn-primary';
            fpActionBtn.disabled = false;

            if (progress.status === 'completed') {
              fpProgressEl.querySelector('.ss-progress-text').textContent = 'Complete!';
              fpProgressEl.querySelector('.ss-progress-bar').style.width = '100%';
            } else if (progress.status === 'paused') {
              fpProgressEl.querySelector('.ss-progress-text').textContent = 'Paused - can resume';
            }
          }
        } catch (e) {
          console.error('[Stash Sense] Error polling fingerprint progress:', e);
        }
      }

      // Start polling if already running
      if (fpStatus.generation_running) {
        fpPollInterval = setInterval(updateFingerprintProgress, 2000);
      }

      fpActionBtn.addEventListener('click', async () => {
        const isRunning = fpActionBtn.textContent.includes('Stop');

        if (isRunning) {
          // Stop generation
          fpActionBtn.disabled = true;
          fpActionBtn.textContent = 'Stopping...';
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

            // Start polling
            if (fpPollInterval) clearInterval(fpPollInterval);
            fpPollInterval = setInterval(updateFingerprintProgress, 2000);
          } catch (e) {
            fpActionBtn.textContent = 'Generate Fingerprints';
            fpActionBtn.className = 'ss-btn ss-btn-primary';
            fpActionBtn.disabled = false;
            console.error('[Stash Sense] Error starting generation:', e);
            alert('Failed to start fingerprint generation: ' + e.message);
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
                <img src="${p.image_path}" alt="${p.name}" loading="lazy" />
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
                <img src="${p.image_path}" alt="${p.name}" loading="lazy" />
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

    // Merge action
    container.querySelector('#ss-merge-btn').addEventListener('click', async () => {
      const keeperId = container.querySelector('input[name="keeper"]:checked')?.value;
      if (!keeperId) {
        alert('Please select a performer to keep');
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

    // Delete action
    container.querySelector('#ss-delete-btn').addEventListener('click', async () => {
      const keeperId = container.querySelector('input[name="keeper"]:checked')?.value;
      if (!keeperId) {
        alert('Please select a file to keep');
        return;
      }

      const fileIdsToDelete = files.filter(f => f.id !== keeperId).map(f => f.id);
      const allFileIds = files.map(f => f.id);

      const confirmMsg = `Delete ${fileIdsToDelete.length} file(s)? This cannot be undone.`;
      if (!confirm(confirmMsg)) return;

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
