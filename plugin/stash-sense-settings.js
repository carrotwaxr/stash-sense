/**
 * Stash Sense Settings Module
 * Settings tab UI for the plugin page — hardware info, sidecar configuration
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  // ==================== Settings API ====================

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

  const SettingsAPI = {
    async getAll() {
      return apiCall('settings_get_all');
    },
    async update(key, value) {
      return apiCall('settings_update', { key, value });
    },
    async reset(key) {
      return apiCall('settings_reset', { key });
    },
    async getSystemInfo() {
      return apiCall('system_info');
    },
  };

  // ==================== State ====================

  let settingsData = null;
  let systemInfo = null;
  let saveTimeouts = {};

  // ==================== Rendering ====================

  function createSettingsContainer() {
    const container = SS.createElement('div', {
      id: 'ss-settings',
      className: 'ss-settings-page',
    });
    return container;
  }

  async function renderSettings(container) {
    container.innerHTML = '<div class="ss-settings-loading">Loading settings...</div>';

    try {
      const [settingsResult, infoResult] = await Promise.all([
        SettingsAPI.getAll(),
        SettingsAPI.getSystemInfo(),
      ]);
      settingsData = settingsResult;
      systemInfo = infoResult;
    } catch (e) {
      const errorDiv = SS.createElement('div', { className: 'ss-settings-error' });
      errorDiv.appendChild(SS.createElement('h3', { textContent: 'Failed to load settings' }));
      errorDiv.appendChild(SS.createElement('p', { textContent: e.message }));
      errorDiv.appendChild(SS.createElement('button', {
        className: 'ss-btn ss-btn-primary',
        textContent: 'Retry',
        events: { click: () => window.StashSenseSettings.refresh() },
      }));
      container.innerHTML = '';
      container.appendChild(errorDiv);
      return;
    }

    container.innerHTML = '';

    // Header
    const header = SS.createElement('div', { className: 'ss-settings-header' });
    header.innerHTML = `
      <h1>Settings</h1>
      <p class="ss-settings-subtitle">Sidecar configuration and hardware info</p>
    `;
    container.appendChild(header);

    // Hardware banner
    if (systemInfo?.hardware) {
      container.appendChild(renderHardwareBanner(systemInfo));
    }

    // Settings categories
    if (settingsData?.categories) {
      const cats = Object.entries(settingsData.categories)
        .sort((a, b) => (a[1].order || 0) - (b[1].order || 0));

      for (const [catKey, cat] of cats) {
        container.appendChild(renderCategory(catKey, cat));
      }
    }

    // Display settings (from plugin-local user settings)
    container.appendChild(await renderDisplayCategory());

    // Upstream sync field monitoring
    container.appendChild(await renderUpstreamSyncCategory());

    // Job Schedules section
    await renderSchedulesCategory(container);
  }

  function renderHardwareBanner(info) {
    const hw = info.hardware;
    const banner = SS.createElement('div', { className: 'ss-hw-banner' });

    const tierClass = hw.tier === 'gpu-high' ? 'ss-tier-high' :
                      hw.tier === 'gpu-low' ? 'ss-tier-low' : 'ss-tier-cpu';

    const gpuText = hw.gpu_available
      ? `${hw.gpu_name || 'GPU'} (${hw.gpu_vram_mb ? Math.round(hw.gpu_vram_mb / 1024) + 'GB VRAM' : 'unknown VRAM'})`
      : 'No GPU detected';

    const uptimeMin = Math.floor((info.uptime_seconds || 0) / 60);
    const uptimeStr = uptimeMin < 60
      ? `${uptimeMin}m`
      : `${Math.floor(uptimeMin / 60)}h ${uptimeMin % 60}m`;

    banner.innerHTML = `
      <div class="ss-hw-banner-content">
        <div class="ss-hw-main">
          <span class="ss-hw-tier ${tierClass}">${hw.tier}</span>
          <span class="ss-hw-gpu">${gpuText}</span>
        </div>
        <div class="ss-hw-details">
          <span>${hw.memory_total_mb ? Math.round(hw.memory_total_mb / 1024) + 'GB RAM' : ''}</span>
          <span>${hw.cpu_cores || '?'} cores</span>
          <span>v${info.version || '?'}</span>
          <span>up ${uptimeStr}</span>
        </div>
      </div>
    `;
    return banner;
  }

  function renderCategory(catKey, cat) {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: `<h2>${cat.label}</h2>`,
    });

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    for (const [key, setting] of Object.entries(cat.settings)) {
      body.appendChild(renderSetting(key, setting));
    }

    section.appendChild(header);
    section.appendChild(body);
    return section;
  }

  function renderSetting(key, setting) {
    const row = SS.createElement('div', { className: 'ss-setting-row' });

    const info = SS.createElement('div', { className: 'ss-setting-info' });
    info.innerHTML = `
      <label class="ss-setting-label">${setting.label}</label>
      <span class="ss-setting-desc">${setting.description}</span>
    `;

    const control = SS.createElement('div', { className: 'ss-setting-control' });

    if (setting.type === 'bool') {
      control.appendChild(renderToggle(key, setting));
    } else if (setting.type === 'int' || setting.type === 'float') {
      control.appendChild(renderNumberInput(key, setting));
    } else {
      control.appendChild(renderTextInput(key, setting));
    }

    // Reset button (only when overridden)
    if (setting.is_override) {
      const resetBtn = SS.createElement('button', {
        className: 'ss-setting-reset',
        textContent: 'Reset',
        events: {
          click: async () => {
            try {
              const result = await SettingsAPI.reset(key);
              setting.value = result.value;
              setting.is_override = false;
              // Re-render this row
              const parent = row.parentNode;
              const newRow = renderSetting(key, setting);
              parent.replaceChild(newRow, row);
              showSaveIndicator(newRow, 'Reset');
            } catch (e) {
              console.error(`Failed to reset ${key}:`, e);
            }
          },
        },
      });
      control.appendChild(resetBtn);
    }

    row.appendChild(info);
    row.appendChild(control);
    return row;
  }

  function renderToggle(key, setting) {
    const toggle = SS.createElement('label', { className: 'ss-toggle' });
    const input = SS.createElement('input', {
      attrs: { type: 'checkbox' },
    });
    input.checked = !!setting.value;
    input.addEventListener('change', () => {
      debouncedSave(key, input.checked, toggle.closest('.ss-setting-row'));
    });

    const slider = SS.createElement('span', { className: 'ss-toggle-slider' });
    toggle.appendChild(input);
    toggle.appendChild(slider);
    return toggle;
  }

  function renderNumberInput(key, setting) {
    const wrapper = SS.createElement('div', { className: 'ss-number-input' });
    const input = SS.createElement('input', {
      attrs: {
        type: 'number',
        value: String(setting.value),
        step: setting.type === 'float' ? '0.5' : '1',
      },
    });
    if (setting.min !== undefined) input.setAttribute('min', setting.min);
    if (setting.max !== undefined) input.setAttribute('max', setting.max);

    input.addEventListener('input', () => {
      const val = setting.type === 'float' ? parseFloat(input.value) : parseInt(input.value, 10);
      if (!isNaN(val)) {
        debouncedSave(key, val, wrapper.closest('.ss-setting-row'));
      }
    });

    wrapper.appendChild(input);

    if (setting.min !== undefined || setting.max !== undefined) {
      const range = SS.createElement('span', {
        className: 'ss-setting-range',
        textContent: `${setting.min ?? ''}–${setting.max ?? ''}`,
      });
      wrapper.appendChild(range);
    }

    return wrapper;
  }

  function renderTextInput(key, setting) {
    const input = SS.createElement('input', {
      attrs: { type: 'text', value: String(setting.value || '') },
    });
    input.addEventListener('input', () => {
      debouncedSave(key, input.value, input.closest('.ss-setting-row'));
    });
    return input;
  }

  // ==================== Display Settings ====================

  async function renderDisplayCategory() {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Display</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    // Normalize Enum Display
    const row = SS.createElement('div', { className: 'ss-setting-row' });
    const info = SS.createElement('div', { className: 'ss-setting-info' });
    info.innerHTML = `
      <label class="ss-setting-label">Normalize Enum Display</label>
      <span class="ss-setting-desc">Show ALL_CAPS values as Title Case (e.g. BROWN → Brown)</span>
    `;

    const control = SS.createElement('div', { className: 'ss-setting-control' });
    const toggle = SS.createElement('label', { className: 'ss-toggle' });
    const input = SS.createElement('input', { attrs: { type: 'checkbox' } });

    try {
      const val = await apiCall('user_get_setting', { key: 'normalize_enum_display' });
      input.checked = val.value !== false;
    } catch (e) {
      console.error('[Stash Sense] Failed to load normalize_enum_display setting:', e);
      input.checked = true;
    }

    input.addEventListener('change', async () => {
      try {
        await apiCall('user_set_setting', { key: 'normalize_enum_display', value: input.checked });
        showSaveIndicator(row, 'Saved');
      } catch (e) {
        showSaveIndicator(row, 'Error', true);
        console.error('[Stash Sense] Failed to save display setting:', e);
      }
    });

    const slider = SS.createElement('span', { className: 'ss-toggle-slider' });
    toggle.appendChild(input);
    toggle.appendChild(slider);
    control.appendChild(toggle);
    row.appendChild(info);
    row.appendChild(control);
    body.appendChild(row);

    section.appendChild(body);
    return section;
  }

  // ==================== Upstream Sync Settings ====================

  async function renderUpstreamSyncCategory() {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Upstream Sync</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    const loadingRow = SS.createElement('div', {
      className: 'ss-setting-row ss-setting-hint',
      textContent: 'Loading stash-box endpoints...',
    });
    body.appendChild(loadingRow);
    section.appendChild(body);

    // Load endpoints asynchronously
    try {
      const configResult = await SS.stashQuery(`
        query { configuration { general { stashBoxes { endpoint name } } } }
      `);
      const endpoints = configResult?.configuration?.general?.stashBoxes || [];

      body.removeChild(loadingRow);

      if (endpoints.length === 0) {
        const emptyRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'No stash-box endpoints configured. Configure them in Stash Settings \u2192 Metadata Providers.',
        });
        body.appendChild(emptyRow);
      } else {
        const helpRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'Configure which fields are monitored for upstream changes per endpoint.',
        });
        body.appendChild(helpRow);

        for (const ep of endpoints) {
          body.appendChild(renderEndpointFieldConfig(ep));
        }
      }
    } catch (e) {
      loadingRow.textContent = `Could not load stash-box endpoints: ${e.message}`;
    }

    return section;
  }

  function renderEndpointFieldConfig(ep) {
    const displayName = ep.name || (() => { try { return new URL(ep.endpoint).hostname; } catch (_) { return ep.endpoint; } })();
    const row = SS.createElement('div', { className: 'ss-setting-row-vertical' });

    // Header row with name and toggle
    const headerRow = SS.createElement('div', { className: 'ss-setting-row-header' });

    const info = SS.createElement('div', { className: 'ss-setting-info' });
    info.innerHTML = `
      <label class="ss-setting-label">${displayName}</label>
      <span class="ss-setting-desc">${ep.endpoint}</span>
    `;
    headerRow.appendChild(info);

    const toggleBtn = SS.createElement('button', {
      className: 'ss-setting-reset',
      textContent: 'Show Fields',
    });
    headerRow.appendChild(toggleBtn);
    row.appendChild(headerRow);

    // Fields wrapper (hidden initially)
    const fieldsWrapper = SS.createElement('div', { className: 'ss-upstream-fields-wrapper' });
    fieldsWrapper.style.display = 'none';
    row.appendChild(fieldsWrapper);

    let fieldsLoaded = false;

    toggleBtn.addEventListener('click', async () => {
      const isHidden = fieldsWrapper.style.display === 'none';
      fieldsWrapper.style.display = isHidden ? 'block' : 'none';
      toggleBtn.textContent = isHidden ? 'Hide Fields' : 'Show Fields';

      if (isHidden && !fieldsLoaded) {
        fieldsLoaded = true;
        const loading = SS.createElement('div', {
          className: 'ss-setting-hint',
          textContent: 'Loading field config...',
        });
        fieldsWrapper.appendChild(loading);

        try {
          const fieldConfig = await apiCall('rec_get_field_config', { endpoint: ep.endpoint });
          loading.remove();

          const fieldsGrid = SS.createElement('div', {
            className: 'ss-upstream-fields-grid',
          });

          const sortedFields = Object.entries(fieldConfig.fields).sort(([, a], [, b]) => a.label.localeCompare(b.label));

          for (const [fieldName, config] of sortedFields) {
            const label = SS.createElement('label', { className: 'ss-upstream-field-label' });
            const cb = SS.createElement('input', { attrs: { type: 'checkbox' } });
            cb.dataset.field = fieldName;
            cb.checked = config.enabled;
            label.appendChild(cb);
            label.appendChild(document.createTextNode(' ' + config.label));
            fieldsGrid.appendChild(label);
          }
          fieldsWrapper.appendChild(fieldsGrid);

          // Save button
          const actionsDiv = SS.createElement('div', { className: 'ss-setting-control' });
          const saveBtn = SS.createElement('button', {
            className: 'ss-btn ss-btn-primary ss-btn-sm',
            textContent: 'Save Field Config',
          });
          const saveStatus = SS.createElement('span', { className: 'ss-setting-hint' });

          saveBtn.addEventListener('click', async () => {
            saveBtn.disabled = true;
            saveStatus.textContent = 'Saving...';
            const configs = {};
            fieldsGrid.querySelectorAll('input[type="checkbox"]').forEach(cb => {
              configs[cb.dataset.field] = cb.checked;
            });
            try {
              await apiCall('rec_set_field_config', { endpoint: ep.endpoint, field_configs: configs });
              saveStatus.textContent = 'Saved!';
              saveStatus.style.color = '#22c55e';
              setTimeout(() => { saveStatus.textContent = ''; saveStatus.style.color = ''; }, 2000);
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
          loading.textContent = `Error loading field config: ${e.message}`;
        }
      }
    });

    return row;
  }

  // ==================== Job Schedules ====================

  let scheduleTimeouts = {};

  function saveSchedule(type, enabled, intervalHours, row) {
    if (scheduleTimeouts[type]) {
      clearTimeout(scheduleTimeouts[type]);
    }
    scheduleTimeouts[type] = setTimeout(async () => {
      try {
        await apiCall('queue_update_schedule', {
          type,
          enabled,
          interval_hours: intervalHours,
        });
        showSaveIndicator(row, 'Saved');
      } catch (e) {
        showSaveIndicator(row, 'Error', true);
        console.error(`Failed to save schedule ${type}:`, e);
      }
    }, 500);
  }

  async function renderSchedulesCategory(container) {
    const section = SS.createElement('div', { className: 'ss-settings-category' });

    const header = SS.createElement('div', {
      className: 'ss-settings-cat-header',
      innerHTML: '<h2>Job Schedules</h2>',
    });
    section.appendChild(header);

    const body = SS.createElement('div', { className: 'ss-settings-cat-body' });

    const desc = SS.createElement('div', {
      className: 'ss-setting-row ss-setting-hint',
      textContent: 'Configure automatic job schedules. Jobs will run at the specified intervals.',
    });
    body.appendChild(desc);

    try {
      const [schedulesResult, typesResult] = await Promise.all([
        apiCall('queue_schedules'),
        apiCall('queue_types'),
      ]);

      const schedules = schedulesResult.schedules || [];
      const types = typesResult.types || [];

      if (schedules.length === 0) {
        const emptyRow = SS.createElement('div', {
          className: 'ss-setting-row ss-setting-hint',
          textContent: 'No schedules configured. They will be created automatically on next sidecar restart.',
        });
        body.appendChild(emptyRow);
        section.appendChild(body);
        container.appendChild(section);
        return;
      }

      for (const schedule of schedules) {
        const typeInfo = types.find(t => t.type_id === schedule.type);
        const displayName = typeInfo ? typeInfo.display_name : schedule.type;
        const description = typeInfo?.description || '';
        const allowedIntervals = typeInfo?.allowed_intervals || [];

        const row = SS.createElement('div', { className: 'ss-setting-row' });

        const info = SS.createElement('div', { className: 'ss-setting-info' });
        info.innerHTML = `
          <label class="ss-setting-label">${displayName}</label>
          <span class="ss-setting-desc">${description}</span>
        `;
        row.appendChild(info);

        const control = SS.createElement('div', { className: 'ss-setting-control' });

        // Enable toggle
        const toggle = SS.createElement('label', { className: 'ss-toggle' });
        const checkbox = SS.createElement('input', {
          attrs: { type: 'checkbox' },
        });
        checkbox.checked = !!schedule.enabled;
        const slider = SS.createElement('span', { className: 'ss-toggle-slider' });
        toggle.appendChild(checkbox);
        toggle.appendChild(slider);
        control.appendChild(toggle);

        // Interval select dropdown
        const select = SS.createElement('select', { className: 'ss-select' });
        const currentHours = schedule.interval_hours || 24;

        for (const interval of allowedIntervals) {
          const option = SS.createElement('option', {
            attrs: { value: String(interval.hours) },
            textContent: interval.label,
          });
          if (interval.hours === currentHours) {
            option.selected = true;
          }
          select.appendChild(option);
        }

        // If current value isn't in the allowed list, add it as a fallback
        if (allowedIntervals.length > 0 && !allowedIntervals.some(i => i.hours === currentHours)) {
          const fallback = SS.createElement('option', {
            attrs: { value: String(currentHours), selected: 'selected' },
            textContent: `Every ${currentHours} hours`,
          });
          select.insertBefore(fallback, select.firstChild);
        }

        select.disabled = !schedule.enabled;
        control.appendChild(select);

        // Auto-save on toggle change
        checkbox.addEventListener('change', () => {
          select.disabled = !checkbox.checked;
          saveSchedule(schedule.type, checkbox.checked, parseFloat(select.value), row);
        });

        // Auto-save on interval change
        select.addEventListener('change', () => {
          saveSchedule(schedule.type, checkbox.checked, parseFloat(select.value), row);
        });

        row.appendChild(control);
        body.appendChild(row);
      }
    } catch (e) {
      const errorRow = SS.createElement('div', {
        className: 'ss-setting-row ss-setting-hint',
        textContent: `Failed to load schedules: ${e.message}`,
      });
      errorRow.style.color = '#ef4444';
      body.appendChild(errorRow);
    }

    section.appendChild(body);
    container.appendChild(section);
  }

  // ==================== Save Logic ====================

  function debouncedSave(key, value, row) {
    if (saveTimeouts[key]) {
      clearTimeout(saveTimeouts[key]);
    }
    saveTimeouts[key] = setTimeout(async () => {
      try {
        await SettingsAPI.update(key, value);
        showSaveIndicator(row, 'Saved');
      } catch (e) {
        showSaveIndicator(row, 'Error', true);
        console.error(`Failed to save ${key}:`, e);
      }
    }, 500);
  }

  function showSaveIndicator(row, text, isError = false) {
    if (!row) return;
    // Remove any existing indicator
    const existing = row.querySelector('.ss-save-indicator');
    if (existing) existing.remove();

    const indicator = SS.createElement('span', {
      className: `ss-save-indicator ${isError ? 'ss-save-error' : 'ss-save-ok'}`,
      textContent: text,
    });
    row.appendChild(indicator);
    setTimeout(() => indicator.remove(), 2000);
  }

  // ==================== Plugin Page Integration ====================

  function injectSettingsTab() {
    const route = SS.getRoute();
    if (route.type !== 'plugin') return;

    // Wait for the recommendations module to create the dashboard
    const existing = document.getElementById('ss-recommendations');
    if (!existing) {
      // Recommendations not yet injected, wait
      return;
    }

    // Check if settings tab already exists
    if (document.getElementById('ss-settings')) return;

    // Find the tab bar (created by recommendations module)
    // The recommendations module creates tabs at the top of the dashboard
    // We need to add a "Settings" tab to it
    const dashboard = existing;

    // Look for or create a top-level tab bar
    let tabBar = dashboard.querySelector('.ss-page-tabs');
    if (!tabBar) {
      // Create page-level tabs (Recommendations | Settings)
      tabBar = SS.createElement('div', { className: 'ss-page-tabs' });

      const recTab = SS.createElement('button', {
        className: 'ss-page-tab active',
        textContent: 'Recommendations',
        attrs: { 'data-tab': 'recommendations' },
      });

      const settingsTab = SS.createElement('button', {
        className: 'ss-page-tab',
        textContent: 'Settings',
        attrs: { 'data-tab': 'settings' },
      });

      tabBar.appendChild(recTab);
      tabBar.appendChild(settingsTab);

      // Insert tab bar at the top of the dashboard
      dashboard.insertBefore(tabBar, dashboard.firstChild);

      // Create settings panel (hidden by default)
      const settingsPanel = createSettingsContainer();
      settingsPanel.style.display = 'none';
      dashboard.appendChild(settingsPanel);

      // Wrap existing content as the "recommendations" panel
      const recContent = SS.createElement('div', {
        className: 'ss-page-panel',
        attrs: { 'data-panel': 'recommendations' },
      });
      // Move all children except tab bar and settings panel into rec panel
      const children = Array.from(dashboard.children);
      for (const child of children) {
        if (child !== tabBar && child !== settingsPanel) {
          recContent.appendChild(child);
        }
      }
      dashboard.insertBefore(recContent, settingsPanel);

      // Tab switching
      tabBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.ss-page-tab');
        if (!btn) return;

        const tabName = btn.dataset.tab;

        // Update active state
        tabBar.querySelectorAll('.ss-page-tab').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');

        // Show/hide panels
        recContent.style.display = tabName === 'recommendations' ? '' : 'none';
        settingsPanel.style.display = tabName === 'settings' ? '' : 'none';

        // Lazy load settings on first view
        if (tabName === 'settings' && !settingsPanel.dataset.loaded) {
          settingsPanel.dataset.loaded = 'true';
          renderSettings(settingsPanel);
        }
      });
    }
  }

  // ==================== Initialization ====================

  function init() {
    // Try to inject after recommendations module
    const tryInject = () => {
      if (SS.getRoute().type === 'plugin') {
        setTimeout(injectSettingsTab, 600);
      }
    };

    tryInject();

    SS.onNavigate((route) => {
      if (route.type === 'plugin') {
        setTimeout(injectSettingsTab, 600);
      }
    });

    console.log(`[${SS.PLUGIN_NAME}] Settings module loaded`);
  }

  // Export
  window.StashSenseSettings = {
    API: SettingsAPI,
    refresh: () => {
      const container = document.getElementById('ss-settings');
      if (container) renderSettings(container);
    },
    init,
  };

  init();
})();
