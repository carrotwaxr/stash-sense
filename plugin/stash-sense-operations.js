/**
 * Stash Sense Operations Module
 * Operations tab UI for job queue visibility and controls
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  // ==================== Queue API ====================

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

  const QueueAPI = {
    async getStatus() { return apiCall('queue_status'); },
    async getJobs(status) { return apiCall('queue_list', status ? { status } : {}); },
    async getTypes() { return apiCall('queue_types'); },
    async submit(type) { return apiCall('queue_submit', { type }); },
    async cancel(jobId) { return apiCall('queue_cancel', { job_id: jobId }); },
    async stop(jobId) { return apiCall('queue_stop', { job_id: jobId }); },
    async retry(jobId) { return apiCall('queue_retry', { job_id: jobId }); },
    async clearHistory() { return apiCall('queue_clear_history'); },
  };

  // ==================== State ====================

  let pollInterval = null;
  let jobTypes = null;
  let historyExpanded = true;

  // ==================== Rendering ====================

  function createOperationsContainer() {
    return SS.createElement('div', {
      id: 'ss-operations',
      className: 'ss-operations-page',
    });
  }

  async function renderOperations(container) {
    container.innerHTML = '<div class="ss-operations-loading">Loading operations...</div>';

    try {
      const [statusResult, jobsResult, typesResult] = await Promise.all([
        QueueAPI.getStatus(),
        QueueAPI.getJobs(),
        QueueAPI.getTypes(),
      ]);
      jobTypes = typesResult.types || [];
      renderContent(container, statusResult, jobsResult.jobs || []);
      startPolling(container);
    } catch (e) {
      container.innerHTML = '';
      const errorDiv = SS.createElement('div', { className: 'ss-operations-error' });
      errorDiv.appendChild(SS.createElement('h3', { textContent: 'Failed to load operations' }));
      errorDiv.appendChild(SS.createElement('p', { textContent: e.message }));
      errorDiv.appendChild(SS.createElement('button', {
        className: 'ss-btn ss-btn-primary',
        textContent: 'Retry',
        events: { click: () => renderOperations(container) },
      }));
      container.appendChild(errorDiv);
    }
  }

  function renderContent(container, status, jobs) {
    container.innerHTML = '';

    // Header
    const header = SS.createElement('div', { className: 'ss-operations-header' });
    header.appendChild(SS.createElement('h1', { textContent: 'Operations' }));
    header.appendChild(SS.createElement('p', {
      className: 'ss-operations-subtitle',
      textContent: `${status.queued} queued \u00b7 ${status.running} running`,
    }));
    container.appendChild(header);

    // Quick Actions
    renderQuickActions(container);

    // Active Jobs (running)
    const running = jobs.filter(j => j.status === 'running' || j.status === 'stopping');
    if (running.length > 0) {
      renderSection(container, 'Active Jobs', running, true);
    }

    // Queue (pending)
    const queued = jobs.filter(j => j.status === 'queued');
    if (queued.length > 0) {
      renderSection(container, 'Queue', queued, false);
    }

    // History (completed/failed/cancelled - collapsible)
    const history = jobs.filter(j => ['completed', 'failed', 'cancelled'].includes(j.status));
    if (history.length > 0) {
      renderHistory(container, history);
    }

    if (running.length === 0 && queued.length === 0 && history.length === 0) {
      container.appendChild(SS.createElement('div', {
        className: 'ss-operations-empty',
        textContent: 'No jobs in the queue. Use Quick Actions to run an operation.',
      }));
    }
  }

  function renderQuickActions(container) {
    if (!jobTypes || jobTypes.length === 0) return;

    const section = SS.createElement('div', { className: 'ss-operations-section' });
    section.appendChild(SS.createElement('h2', { textContent: 'Quick Actions' }));

    const grid = SS.createElement('div', { className: 'ss-quick-actions' });

    for (const type of jobTypes) {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-secondary ss-quick-action-btn',
        attrs: { 'data-type': type.type_id },
        events: {
          click: async (e) => {
            const button = e.currentTarget;
            button.disabled = true;
            button.textContent = 'Submitting...';
            try {
              await QueueAPI.submit(type.type_id);
              await refreshContent(container);
            } catch (err) {
              if (err.message.includes('409') || err.message.includes('already')) {
                button.textContent = 'Already Running';
                setTimeout(() => { button.textContent = type.display_name; button.disabled = false; }, 2000);
              } else {
                button.textContent = 'Error';
                setTimeout(() => { button.textContent = type.display_name; button.disabled = false; }, 2000);
              }
              return;
            }
            button.textContent = type.display_name;
            button.disabled = false;
          },
        },
      });

      // Resource badge
      const badge = SS.createElement('span', {
        className: `ss-resource-badge ss-resource-${type.resource}`,
        textContent: type.resource.toUpperCase(),
      });
      btn.appendChild(document.createTextNode(type.display_name + ' '));
      btn.appendChild(badge);
      grid.appendChild(btn);
    }

    section.appendChild(grid);
    container.appendChild(section);
  }

  function renderSection(container, title, jobs, isActive) {
    const section = SS.createElement('div', { className: 'ss-operations-section' });
    section.appendChild(SS.createElement('h2', { textContent: title }));

    const list = SS.createElement('div', { className: 'ss-job-list' });
    for (const job of jobs) {
      list.appendChild(renderJobCard(job, isActive));
    }
    section.appendChild(list);
    container.appendChild(section);
  }

  function renderJobCard(job, isActive) {
    const card = SS.createElement('div', {
      className: `ss-job-card ss-job-${job.status}`,
    });

    // Job header row
    const headerRow = SS.createElement('div', { className: 'ss-job-header' });

    const typeInfo = jobTypes ? jobTypes.find(t => t.type_id === job.type) : null;
    const displayName = typeInfo ? typeInfo.display_name : job.type;

    headerRow.appendChild(SS.createElement('span', {
      className: 'ss-job-name',
      textContent: displayName,
    }));

    // Resource badge
    if (typeInfo) {
      headerRow.appendChild(SS.createElement('span', {
        className: `ss-resource-badge ss-resource-${typeInfo.resource}`,
        textContent: typeInfo.resource.toUpperCase(),
      }));
    }

    // Status badge
    headerRow.appendChild(SS.createElement('span', {
      className: `ss-status-badge ss-status-${job.status}`,
      textContent: job.status,
    }));

    card.appendChild(headerRow);

    // Progress bar for active jobs
    if (isActive && job.items_total && job.items_total > 0) {
      const pct = Math.round((job.items_processed / job.items_total) * 100);
      const progressWrap = SS.createElement('div', { className: 'ss-progress-wrap' });
      const progressBar = SS.createElement('div', { className: 'ss-progress-bar ss-ops-progress-bar' });
      const progressFill = SS.createElement('div', {
        className: 'ss-progress-fill',
        styles: { width: `${pct}%` },
      });
      progressBar.appendChild(progressFill);
      progressWrap.appendChild(progressBar);
      progressWrap.appendChild(SS.createElement('span', {
        className: 'ss-ops-progress-text',
        textContent: `${job.items_processed} / ${job.items_total} (${pct}%)`,
      }));
      card.appendChild(progressWrap);
    }

    // Meta row
    const metaRow = SS.createElement('div', { className: 'ss-job-meta' });
    const triggeredByDisplay = job.triggered_by
      ? job.triggered_by.charAt(0).toUpperCase() + job.triggered_by.slice(1)
      : 'Unknown';
    metaRow.appendChild(SS.createElement('span', {
      textContent: `Triggered by ${triggeredByDisplay}`,
    }));
    if (job.started_at) {
      const elapsed = getElapsed(job.started_at);
      metaRow.appendChild(SS.createElement('span', { textContent: elapsed }));
    }
    card.appendChild(metaRow);

    // Error message
    if (job.error_message) {
      card.appendChild(SS.createElement('div', {
        className: 'ss-job-error',
        textContent: job.error_message,
      }));
    }

    // Action buttons
    const actions = SS.createElement('div', { className: 'ss-job-actions' });

    function actionHandler(button, action, label, pendingLabel) {
      return async () => {
        button.disabled = true;
        button.textContent = pendingLabel;
        try {
          await action();
          const container = button.closest('#ss-operations');
          if (container) await refreshContent(container);
        } catch (err) {
          button.textContent = 'Error';
          setTimeout(() => { button.textContent = label; button.disabled = false; }, 2000);
        }
      };
    }

    if (job.status === 'running') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-danger ss-btn-sm',
        textContent: 'Stop',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.stop(job.id), 'Stop', 'Stopping\u2026'));
      actions.appendChild(btn);
    } else if (job.status === 'stopping') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-danger ss-btn-sm',
        textContent: 'Force Cancel',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.cancel(job.id), 'Force Cancel', 'Cancelling\u2026'));
      actions.appendChild(btn);
    } else if (job.status === 'queued') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-secondary ss-btn-sm',
        textContent: 'Cancel',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.cancel(job.id), 'Cancel', 'Cancelling\u2026'));
      actions.appendChild(btn);
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      const btn = SS.createElement('button', {
        className: 'ss-btn ss-btn-primary ss-btn-sm',
        textContent: 'Retry',
      });
      btn.addEventListener('click', actionHandler(btn, () => QueueAPI.retry(job.id), 'Retry', 'Retrying\u2026'));
      actions.appendChild(btn);
    }
    if (actions.children.length > 0) {
      card.appendChild(actions);
    }

    return card;
  }

  function renderHistory(container, jobs) {
    const section = SS.createElement('div', { className: 'ss-operations-section' });

    const headerRow = SS.createElement('div', { className: 'ss-setting-row-header' });

    const toggle = SS.createElement('h2', {
      className: 'ss-collapsible-header',
      events: {
        click: () => {
          historyExpanded = !historyExpanded;
          list.style.display = historyExpanded ? '' : 'none';
          toggle.classList.toggle('ss-collapsed');
        },
      },
    });
    toggle.textContent = `History (${jobs.length})`;
    if (!historyExpanded) {
      toggle.classList.add('ss-collapsed');
    }
    headerRow.appendChild(toggle);

    const clearBtn = SS.createElement('button', {
      className: 'ss-btn ss-btn-secondary ss-btn-sm',
      textContent: 'Clear History',
      events: {
        click: async (e) => {
          e.stopPropagation();
          const button = e.currentTarget;
          button.disabled = true;
          button.textContent = 'Clearing...';
          try {
            await QueueAPI.clearHistory();
            await refreshContent(container);
          } catch (err) {
            button.textContent = 'Error';
            setTimeout(() => { button.textContent = 'Clear History'; button.disabled = false; }, 2000);
          }
        },
      },
    });
    headerRow.appendChild(clearBtn);

    section.appendChild(headerRow);

    const list = SS.createElement('div', {
      className: 'ss-job-list',
    });
    if (!historyExpanded) {
      list.style.display = 'none';
    }
    for (const job of jobs.slice(0, 20)) {
      list.appendChild(renderJobCard(job, false));
    }
    section.appendChild(list);
    container.appendChild(section);
  }

  function getElapsed(startedAt) {
    if (!startedAt) return '';
    const start = new Date(startedAt + 'Z');
    const now = new Date();
    const secs = Math.floor((now - start) / 1000);
    if (secs < 60) return `${secs}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  }

  // ==================== Polling ====================

  function startPolling(container) {
    stopPolling();
    pollInterval = setInterval(() => refreshContent(container), 3000);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  async function refreshContent(container) {
    // Bail out if container has been removed from DOM (navigation away)
    if (!document.contains(container)) {
      stopPolling();
      return;
    }
    try {
      const [statusResult, jobsResult] = await Promise.all([
        QueueAPI.getStatus(),
        QueueAPI.getJobs(),
      ]);
      renderContent(container, statusResult, jobsResult.jobs || []);
    } catch (e) {
      console.error('[Stash Sense] Operations poll error:', e);
    }
  }

  // ==================== Tab Injection ====================

  function injectOperationsTab() {
    const route = SS.getRoute();
    if (route.type !== 'plugin') return;

    // Wait for tab bar to exist (created by settings module)
    const dashboard = document.getElementById('ss-recommendations');
    if (!dashboard) return;

    const tabBar = dashboard.querySelector('.ss-page-tabs');
    if (!tabBar) return;

    // Already injected?
    if (document.getElementById('ss-operations')) return;
    if (tabBar.querySelector('[data-tab="operations"]')) return;

    // Check if we should start on this tab
    const initialTab = SS.getTabFromUrl();

    // Create Operations tab button -- insert BEFORE Settings tab
    const operationsTab = SS.createElement('button', {
      className: `ss-page-tab ${initialTab === 'operations' ? 'active' : ''}`,
      textContent: 'Operations',
      attrs: { 'data-tab': 'operations' },
    });

    const settingsTabBtn = tabBar.querySelector('[data-tab="settings"]');
    if (settingsTabBtn) {
      tabBar.insertBefore(operationsTab, settingsTabBtn);
    } else {
      tabBar.appendChild(operationsTab);
    }

    // Create operations panel
    const operationsPanel = createOperationsContainer();
    operationsPanel.style.display = initialTab === 'operations' ? '' : 'none';
    operationsPanel.setAttribute('data-panel', 'operations');

    // Insert before settings panel
    const settingsPanel = document.getElementById('ss-settings');
    if (settingsPanel) {
      settingsPanel.parentElement.insertBefore(operationsPanel, settingsPanel);
    } else {
      dashboard.appendChild(operationsPanel);
    }

    // If starting on operations tab, hide other panels and lazy load
    if (initialTab === 'operations') {
      // Deactivate other tabs
      tabBar.querySelectorAll('.ss-page-tab').forEach(t => {
        if (t !== operationsTab) t.classList.remove('active');
      });
      // Hide other panels
      const recPanel = dashboard.querySelector('.ss-page-panel[data-panel="recommendations"]');
      if (recPanel) recPanel.style.display = 'none';
      if (settingsPanel) settingsPanel.style.display = 'none';
      // Load content
      operationsPanel.dataset.loaded = 'true';
      renderOperations(operationsPanel);
    }

    // Patch the existing tab click handler to include operations
    tabBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.ss-page-tab');
      if (!btn) return;
      const tabName = btn.dataset.tab;

      // Update URL
      SS.setTabInUrl(tabName);

      // Show/hide operations panel
      operationsPanel.style.display = tabName === 'operations' ? '' : 'none';

      // Lazy load on first view
      if (tabName === 'operations' && !operationsPanel.dataset.loaded) {
        operationsPanel.dataset.loaded = 'true';
        renderOperations(operationsPanel);
      }

      // Stop polling when leaving operations tab
      if (tabName !== 'operations') {
        stopPolling();
      }
    });
  }

  // ==================== Initialization ====================

  function cleanup() {
    stopPolling();
    const operations = document.getElementById('ss-operations');
    if (operations) operations.remove();
    jobTypes = null;
  }

  function init() {
    const tryInject = () => {
      if (SS.getRoute().type === 'plugin') {
        setTimeout(injectOperationsTab, 800);
      }
    };

    tryInject();

    SS.onNavigate((route) => {
      if (route.type === 'plugin') {
        setTimeout(injectOperationsTab, 800);
      } else {
        stopPolling();
      }
    });

    // Clean up when leaving plugin page
    SS.onLeavePlugin(cleanup);

    console.log(`[${SS.PLUGIN_NAME}] Operations module loaded`);
  }

  window.StashSenseOperations = {
    refresh: () => {
      const container = document.getElementById('ss-operations');
      if (container) renderOperations(container);
    },
    init,
  };

  init();
})();
