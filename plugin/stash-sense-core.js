/**
 * Stash Sense Core Module
 * Shared utilities, settings, and API client
 */
(function() {
  'use strict';

  // Plugin configuration
  const PLUGIN_ID = 'stash-sense';
  const PLUGIN_NAME = 'Stash Sense';

  // Default settings
  const DEFAULTS = {
    sidecarUrl: 'http://localhost:5000',
    minConfidence: 50,
    maxResults: 5,
  };

  // Cached state
  let cachedSettings = null;
  let sidecarStatus = null; // null = unknown, true = connected, false = disconnected

  // ==================== Settings ====================

  async function getSettings() {
    if (cachedSettings) return cachedSettings;

    try {
      const response = await fetch('/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `query Configuration { configuration { plugins } }`,
        }),
      });
      const result = await response.json();
      const pluginConfig = result?.data?.configuration?.plugins?.[PLUGIN_ID];

      cachedSettings = {
        sidecarUrl: (pluginConfig?.sidecarUrl || DEFAULTS.sidecarUrl).replace(/\/$/, ''),
        minConfidence: parseInt(pluginConfig?.minConfidence || DEFAULTS.minConfidence, 10),
        maxResults: parseInt(pluginConfig?.maxResults || DEFAULTS.maxResults, 10),
      };

      console.log(`[${PLUGIN_NAME}] Settings loaded:`, cachedSettings);
      return cachedSettings;
    } catch (e) {
      console.error(`[${PLUGIN_NAME}] Failed to load settings:`, e);
      return DEFAULTS;
    }
  }

  function clearSettingsCache() {
    cachedSettings = null;
  }

  // ==================== Sidecar API Client ====================

  /**
   * Make a request to the sidecar API
   */
  async function sidecarFetch(endpoint, options = {}) {
    const settings = await getSettings();
    const url = `${settings.sidecarUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Sidecar API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  /**
   * Run a plugin operation via Stash's GraphQL API (for operations that need Stash access)
   */
  async function runPluginOperation(mode, args = {}) {
    const query = `
      mutation RunPluginOperation($plugin_id: ID!, $args: Map!) {
        runPluginOperation(plugin_id: $plugin_id, args: $args)
      }
    `;

    const response = await fetch('/graphql', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        variables: {
          plugin_id: PLUGIN_ID,
          args: { mode, ...args },
        },
      }),
    });

    const result = await response.json();

    if (result.errors) {
      throw new Error(result.errors[0]?.message || 'GraphQL error');
    }

    const output = result?.data?.runPluginOperation;
    if (typeof output === 'string') {
      try {
        const parsed = JSON.parse(output);
        return parsed.output || parsed;
      } catch {
        return { error: output };
      }
    }

    return output?.output || output || {};
  }

  /**
   * Check sidecar health (via Python backend to bypass CSP)
   */
  async function checkHealth() {
    try {
      const settings = await getSettings();
      const result = await runPluginOperation('health', {
        sidecar_url: settings.sidecarUrl,
      });
      if (result.error) {
        sidecarStatus = false;
        return null;
      }
      sidecarStatus = true;
      return result;
    } catch (e) {
      sidecarStatus = false;
      return null;
    }
  }

  function getSidecarStatus() {
    return sidecarStatus;
  }

  function setSidecarStatus(status) {
    sidecarStatus = status;
  }

  // ==================== Stash GraphQL Helpers ====================

  /**
   * Execute a GraphQL query against Stash
   */
  async function stashQuery(query, variables = {}) {
    const response = await fetch('/graphql', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, variables }),
    });
    const result = await response.json();
    if (result.errors) {
      throw new Error(result.errors[0]?.message || 'GraphQL error');
    }
    return result.data;
  }

  /**
   * Look up a performer by StashDB ID
   */
  async function findPerformerByStashDBId(stashdbId, endpoint = 'https://stashdb.org/graphql') {
    const query = `
      query FindByStashDBId($stashdb_id: String!, $endpoint: String!) {
        findPerformers(performer_filter: {
          stash_id_endpoint: {
            endpoint: $endpoint
            stash_id: $stashdb_id
            modifier: EQUALS
          }
        }) {
          performers {
            id
            name
            image_path
          }
        }
      }
    `;

    try {
      const data = await stashQuery(query, { stashdb_id: stashdbId, endpoint });
      const performers = data?.findPerformers?.performers || [];
      return performers.length > 0 ? performers[0] : null;
    } catch (e) {
      console.error('Failed to lookup performer:', e);
      return null;
    }
  }

  /**
   * Get performer details by ID
   */
  async function getPerformer(id) {
    const query = `
      query GetPerformer($id: ID!) {
        findPerformer(id: $id) {
          id
          name
          image_path
          scene_count
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findPerformer;
  }

  /**
   * Get scene details by ID
   */
  async function getScene(id) {
    const query = `
      query GetScene($id: ID!) {
        findScene(id: $id) {
          id
          title
          date
          paths {
            screenshot
          }
          files {
            id
            path
            basename
            size
            duration
            video_codec
            width
            height
          }
          performers {
            id
            name
            image_path
          }
          studio {
            id
            name
          }
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findScene;
  }

  // ==================== URL Routing ====================

  /**
   * Get the current route info
   */
  function getRoute() {
    const path = window.location.pathname;

    // Scene page
    const sceneMatch = path.match(/\/scenes\/(\d+)/);
    if (sceneMatch) {
      return { type: 'scene', id: sceneMatch[1] };
    }

    // Performer page
    const performerMatch = path.match(/\/performers\/(\d+)/);
    if (performerMatch) {
      return { type: 'performer', id: performerMatch[1] };
    }

    // Plugin page
    if (path.startsWith('/plugins/stash-sense')) {
      const subpath = path.replace('/plugins/stash-sense', '') || '/';
      return { type: 'plugin', subpath };
    }

    return { type: 'other', path };
  }

  // ==================== UI Utilities ====================

  /**
   * Format bytes to human readable
   */
  function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
  }

  /**
   * Format duration in seconds to HH:MM:SS
   */
  function formatDuration(seconds) {
    if (!seconds) return 'Unknown';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  /**
   * Format a date string
   */
  function formatDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  /**
   * Create an element with classes and attributes
   */
  function createElement(tag, options = {}) {
    const el = document.createElement(tag);
    if (options.className) el.className = options.className;
    if (options.id) el.id = options.id;
    if (options.innerHTML) el.innerHTML = options.innerHTML;
    if (options.textContent) el.textContent = options.textContent;
    if (options.attrs) {
      for (const [key, value] of Object.entries(options.attrs)) {
        el.setAttribute(key, value);
      }
    }
    if (options.events) {
      for (const [event, handler] of Object.entries(options.events)) {
        el.addEventListener(event, handler);
      }
    }
    if (options.children) {
      for (const child of options.children) {
        el.appendChild(child);
      }
    }
    return el;
  }

  /**
   * Convert distance (0-2) to confidence percentage (0-100)
   */
  function distanceToConfidence(distance) {
    const clamped = Math.max(0, Math.min(1, distance));
    return Math.round((1 - clamped) * 100);
  }

  /**
   * Get confidence level class
   */
  function getConfidenceClass(confidence) {
    if (confidence >= 70) return 'high';
    if (confidence >= 50) return 'medium';
    return 'low';
  }

  // ==================== SPA Navigation ====================

  const navigationCallbacks = [];

  function onNavigate(callback) {
    navigationCallbacks.push(callback);
  }

  function initNavigationWatcher() {
    let lastUrl = window.location.href;

    const observer = new MutationObserver(() => {
      if (window.location.href !== lastUrl) {
        lastUrl = window.location.href;
        const route = getRoute();
        for (const callback of navigationCallbacks) {
          try {
            callback(route);
          } catch (e) {
            console.error(`[${PLUGIN_NAME}] Navigation callback error:`, e);
          }
        }
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  // ==================== Export ====================

  window.StashSense = {
    // Constants
    PLUGIN_ID,
    PLUGIN_NAME,
    DEFAULTS,

    // Settings
    getSettings,
    clearSettingsCache,

    // Sidecar API
    sidecarFetch,
    runPluginOperation,
    checkHealth,
    getSidecarStatus,
    setSidecarStatus,

    // Stash GraphQL
    stashQuery,
    findPerformerByStashDBId,
    getPerformer,
    getScene,

    // Routing
    getRoute,
    onNavigate,
    initNavigationWatcher,

    // Utilities
    formatSize,
    formatDuration,
    formatDate,
    createElement,
    distanceToConfidence,
    getConfidenceClass,
  };

  console.log(`[${PLUGIN_NAME}] Core module loaded`);
})();
