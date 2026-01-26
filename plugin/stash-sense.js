(function() {
  'use strict';

  // Plugin configuration
  const PLUGIN_ID = 'stash-sense';
  const PLUGIN_NAME = 'Stash Sense';

  // Default settings
  const DEFAULTS = {
    sidecarUrl: 'http://localhost:5000',
    minConfidence: 50,  // Now a percentage (0-100)
    maxResults: 5,
  };

  // State
  let cachedSettings = null;
  let sidecarStatus = null;  // null = unknown, true = connected, false = disconnected

  // Get plugin settings via GraphQL
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

  // Convert distance (0-2, lower=better) to confidence percentage (0-100, higher=better)
  function distanceToConfidence(distance) {
    // Distance 0 = 100% confidence, Distance 1 = 0% confidence
    // Clamp between 0 and 1 for reasonable display
    const clamped = Math.max(0, Math.min(1, distance));
    return Math.round((1 - clamped) * 100);
  }

  // Get confidence level class
  function getConfidenceClass(confidence) {
    if (confidence >= 70) return 'high';
    if (confidence >= 50) return 'medium';
    return 'low';
  }

  // Extract scene ID from URL
  function getSceneIdFromUrl() {
    const match = window.location.pathname.match(/\/scenes\/(\d+)/);
    return match ? match[1] : null;
  }

  // Call the face recognition API via Python backend (bypasses CSP)
  async function identifyScene(sceneId, onProgress) {
    const settings = await getSettings();

    onProgress?.('Connecting to Stash Sense...');

    // Convert confidence percentage to max_distance
    // minConfidence 50% means max_distance 0.5
    const maxDistance = 1 - (settings.minConfidence / 100);

    const result = await runPluginOperation('identify_scene', {
      scene_id: sceneId,
      sidecar_url: settings.sidecarUrl,
      top_k: settings.maxResults,
      max_distance: maxDistance,
    });

    if (result.error) {
      throw new Error(result.error);
    }

    return result;
  }

  // Run a plugin operation via Stash's GraphQL API
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

    // Parse the output from the plugin
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

  // Check if sidecar is healthy via Python backend
  async function checkHealth() {
    try {
      const settings = await getSettings();
      const result = await runPluginOperation('health', {
        sidecar_url: settings.sidecarUrl,
      });
      return result.error ? null : result;
    } catch (e) {
      return null;
    }
  }

  // Look up performer in local Stash by StashDB ID
  async function findLocalPerformer(stashdbId) {
    const query = `
      query FindByStashDBId($stashdb_id: String!) {
        findPerformers(performer_filter: {
          stash_id_endpoint: {
            endpoint: "https://stashdb.org/graphql"
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
      const response = await fetch('/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          variables: { stashdb_id: stashdbId },
        }),
      });
      const result = await response.json();
      const performers = result?.data?.findPerformers?.performers || [];
      return performers.length > 0 ? performers[0] : null;
    } catch (e) {
      console.error('Failed to lookup performer:', e);
      return null;
    }
  }

  // Add performer to scene
  async function addPerformerToScene(sceneId, performerId) {
    // First, get current performers
    const getQuery = `
      query GetScene($id: ID!) {
        findScene(id: $id) {
          performers { id }
        }
      }
    `;

    const updateQuery = `
      mutation UpdateScene($id: ID!, $performer_ids: [ID!]) {
        sceneUpdate(input: { id: $id, performer_ids: $performer_ids }) {
          id
        }
      }
    `;

    try {
      const getResponse = await fetch('/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: getQuery,
          variables: { id: sceneId },
        }),
      });
      const getResult = await getResponse.json();
      const currentPerformers = getResult?.data?.findScene?.performers || [];
      const currentIds = currentPerformers.map(p => p.id);

      // Add new performer if not already present
      if (!currentIds.includes(performerId)) {
        currentIds.push(performerId);
      }

      const updateResponse = await fetch('/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: updateQuery,
          variables: { id: sceneId, performer_ids: currentIds },
        }),
      });
      return updateResponse.ok;
    } catch (e) {
      console.error('Failed to add performer:', e);
      return false;
    }
  }

  // Create the results modal
  function createModal() {
    // Remove existing modal if present
    const existing = document.getElementById('ss-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'ss-modal';
    modal.className = 'ss-modal';
    modal.innerHTML = `
      <div class="ss-modal-backdrop"></div>
      <div class="ss-modal-content">
        <div class="ss-modal-header">
          <h3>Stash Sense Results</h3>
          <button class="ss-modal-close" aria-label="Close">&times;</button>
        </div>
        <div class="ss-modal-body">
          <div class="ss-loading">
            <div class="ss-spinner"></div>
            <p class="ss-loading-text">Connecting to Stash Sense...</p>
            <p class="ss-loading-detail"></p>
          </div>
          <div class="ss-results" style="display: none;"></div>
          <div class="ss-error" style="display: none;"></div>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Close handlers
    const closeModal = () => modal.remove();
    modal.querySelector('.ss-modal-close').addEventListener('click', closeModal);
    modal.querySelector('.ss-modal-backdrop').addEventListener('click', closeModal);

    // ESC key handler
    const escHandler = (e) => {
      if (e.key === 'Escape') {
        closeModal();
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);

    return modal;
  }

  // Update loading state
  function updateLoading(modal, message, detail = '') {
    const loadingText = modal.querySelector('.ss-loading-text');
    const loadingDetail = modal.querySelector('.ss-loading-detail');
    if (loadingText) loadingText.textContent = message;
    if (loadingDetail) loadingDetail.textContent = detail;
  }

  // Render results in the modal
  async function renderResults(modal, results, sceneId) {
    const loading = modal.querySelector('.ss-loading');
    const resultsDiv = modal.querySelector('.ss-results');
    const errorDiv = modal.querySelector('.ss-error');

    loading.style.display = 'none';

    if (!results.persons || results.persons.length === 0) {
      errorDiv.innerHTML = `
        <div class="ss-error-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
          </svg>
        </div>
        <p class="ss-error-title">No faces detected</p>
        <p class="ss-error-hint">
          This could mean the scene doesn't have clear face shots, or the sprite sheet quality is too low.
          Try regenerating sprites with higher quality settings.
        </p>
      `;
      errorDiv.style.display = 'block';
      return;
    }

    resultsDiv.innerHTML = `
      <p class="ss-summary">
        Analyzed <strong>${results.frames_analyzed}</strong> frames,
        detected <strong>${results.faces_detected}</strong> faces,
        identified <strong>${results.persons.length}</strong> unique person(s).
      </p>
      <div class="ss-persons"></div>
    `;

    const personsDiv = resultsDiv.querySelector('.ss-persons');

    for (const person of results.persons) {
      const personDiv = document.createElement('div');
      personDiv.className = 'ss-person';

      if (!person.best_match) {
        personDiv.innerHTML = `
          <div class="ss-person-header">
            <span class="ss-person-label">Unknown Person</span>
            <span class="ss-person-frames">${person.frame_count} appearances</span>
          </div>
          <p class="ss-no-match">No match found in database</p>
        `;
      } else {
        const match = person.best_match;
        // Convert distance to confidence percentage
        const confidence = distanceToConfidence(match.distance || (1 - match.confidence) || 0.5);
        const confidenceClass = getConfidenceClass(confidence);

        // Check if performer exists in local Stash
        const localPerformer = await findLocalPerformer(match.stashdb_id);

        personDiv.innerHTML = `
          <div class="ss-person-header">
            <span class="ss-person-label">Person ${person.person_id + 1}</span>
            <span class="ss-person-frames">${person.frame_count} appearances</span>
          </div>
          <div class="ss-match">
            <div class="ss-match-image">
              ${match.image_url ? `<img src="${match.image_url}" alt="${match.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
            </div>
            <div class="ss-match-info">
              <h4>${match.name}</h4>
              <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
              ${match.country ? `<div class="ss-country">${match.country}</div>` : ''}
              <div class="ss-links">
                <a href="https://stashdb.org/performers/${match.stashdb_id}" target="_blank" rel="noopener" class="ss-link">
                  View on StashDB
                </a>
              </div>
              <div class="ss-actions">
                ${localPerformer
                  ? `<button class="ss-btn ss-btn-add" data-performer-id="${localPerformer.id}" data-scene-id="${sceneId}">
                       Add to Scene
                     </button>
                     <span class="ss-local-status">In library as: ${localPerformer.name}</span>`
                  : `<span class="ss-local-status ss-not-in-library">Not in library</span>`
                }
              </div>
            </div>
          </div>
          ${person.all_matches && person.all_matches.length > 1 ? `
            <details class="ss-other-matches">
              <summary>Other possible matches (${person.all_matches.length - 1})</summary>
              <ul>
                ${person.all_matches.slice(1).map(m => {
                  const altConf = distanceToConfidence(m.distance || (1 - m.confidence) || 0.5);
                  return `
                    <li>
                      <a href="https://stashdb.org/performers/${m.stashdb_id}" target="_blank" rel="noopener">
                        ${m.name}
                      </a>
                      <span class="ss-alt-confidence">${altConf}%</span>
                    </li>
                  `;
                }).join('')}
              </ul>
            </details>
          ` : ''}
        `;
      }

      personsDiv.appendChild(personDiv);
    }

    // Add click handlers for "Add to Scene" buttons
    resultsDiv.querySelectorAll('.ss-btn-add').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const performerId = e.target.dataset.performerId;
        const targetSceneId = e.target.dataset.sceneId;
        btn.disabled = true;
        btn.textContent = 'Adding...';

        const success = await addPerformerToScene(targetSceneId, performerId);
        if (success) {
          btn.textContent = 'Added!';
          btn.classList.add('ss-btn-success');
        } else {
          btn.textContent = 'Failed';
          btn.classList.add('ss-btn-error');
          btn.disabled = false;
        }
      });
    });

    resultsDiv.style.display = 'block';
  }

  // Show error in modal
  function showError(modal, message) {
    const loading = modal.querySelector('.ss-loading');
    const errorDiv = modal.querySelector('.ss-error');

    loading.style.display = 'none';

    // Categorize error for better messaging
    let title = 'Analysis Failed';
    let hint = 'Check plugin settings and ensure Stash Sense is running.';

    if (message.includes('Connection') || message.includes('connect')) {
      title = 'Connection Failed';
      hint = 'Could not connect to Stash Sense. Make sure the sidecar container is running and the URL in plugin settings is correct.';
    } else if (message.includes('timeout') || message.includes('Timeout')) {
      title = 'Request Timed Out';
      hint = 'Scene analysis took too long. The scene may have too many frames or the sidecar is overloaded.';
    } else if (message.includes('sprite') || message.includes('Sprite')) {
      title = 'Sprite Sheet Error';
      hint = 'Could not fetch the scene sprite sheet. Make sure sprites are generated for this scene.';
    }

    errorDiv.innerHTML = `
      <div class="ss-error-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
      </div>
      <p class="ss-error-title">${title}</p>
      <p class="ss-error-message">${message}</p>
      <p class="ss-error-hint">${hint}</p>
    `;
    errorDiv.style.display = 'block';
  }

  // Main identify handler for scenes
  async function handleIdentifyScene() {
    const sceneId = getSceneIdFromUrl();
    if (!sceneId) {
      alert('Could not determine scene ID');
      return;
    }

    const modal = createModal();

    try {
      updateLoading(modal, 'Fetching scene sprites...', 'This may take a moment for long scenes');

      const results = await identifyScene(sceneId, (stage) => {
        updateLoading(modal, stage);
      });

      updateLoading(modal, 'Processing results...');
      await renderResults(modal, results, sceneId);
    } catch (error) {
      console.error(`[${PLUGIN_NAME}] Analysis failed:`, error);
      showError(modal, error.message);
    }
  }

  // Create the identify button
  function createIdentifyButton() {
    const btn = document.createElement('button');
    btn.className = 'ss-identify-btn btn btn-secondary';
    btn.title = sidecarStatus === false ? 'Stash Sense: Not connected' : 'Identify performers using face recognition';
    btn.innerHTML = `
      <span class="ss-btn-icon ${sidecarStatus === true ? 'ss-connected' : sidecarStatus === false ? 'ss-disconnected' : ''}">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
        </svg>
      </span>
      <span class="ss-btn-text">Identify Performers</span>
    `;
    btn.addEventListener('click', handleIdentifyScene);
    return btn;
  }

  // Update all buttons with connection status
  function updateButtonStatus(connected) {
    document.querySelectorAll('.ss-identify-btn .ss-btn-icon').forEach(icon => {
      icon.classList.remove('ss-connected', 'ss-disconnected');
      if (connected === true) {
        icon.classList.add('ss-connected');
      } else if (connected === false) {
        icon.classList.add('ss-disconnected');
      }
    });

    document.querySelectorAll('.ss-identify-btn').forEach(btn => {
      btn.title = connected === false
        ? 'Stash Sense: Not connected'
        : 'Identify performers using face recognition';
    });
  }

  // Inject button into scene page
  function injectSceneButton() {
    // Don't inject if not on a scene page
    if (!window.location.pathname.match(/\/scenes\/\d+/)) return;

    // Don't inject if button already exists
    if (document.querySelector('.ss-identify-btn')) return;

    // Find the button container - try multiple selectors for Stash version compatibility
    const buttonContainers = [
      // Stash v0.24+
      '.scene-toolbar .btn-group',
      '.detail-header .ml-auto .btn-group',
      // Stash v0.20-0.23
      '.scene-header .btn-group',
      '.detail-header-buttons',
      // Generic fallbacks
      '.scene-operations',
      '.scene-tabs + div .ml-auto',
      '.ml-auto.btn-group',
    ];

    for (const selector of buttonContainers) {
      const container = document.querySelector(selector);
      if (container) {
        container.appendChild(createIdentifyButton());
        console.log(`[${PLUGIN_NAME}] Button injected into ${selector}`);
        return;
      }
    }

    // Fallback: create floating button
    const floatingBtn = createIdentifyButton();
    floatingBtn.classList.add('ss-floating-btn');
    document.body.appendChild(floatingBtn);
    console.log(`[${PLUGIN_NAME}] Floating button added (no container found)`);
  }

  // Watch for page changes (Stash is a SPA)
  function watchForPageChanges() {
    let lastUrl = window.location.href;

    const observer = new MutationObserver(() => {
      if (window.location.href !== lastUrl) {
        lastUrl = window.location.href;
        // Small delay to let page render
        setTimeout(injectSceneButton, 300);
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  // Initialize plugin
  async function init() {
    console.log(`[${PLUGIN_NAME}] Plugin initializing...`);

    // Check if sidecar is available
    const health = await checkHealth();
    if (health) {
      sidecarStatus = true;
      console.log(`[${PLUGIN_NAME}] Sidecar connected: ${health.performer_count} performers, ${health.face_count} faces`);
    } else {
      sidecarStatus = false;
      console.warn(`[${PLUGIN_NAME}] Sidecar not available. Check settings.`);
    }

    // Initial injection
    setTimeout(injectSceneButton, 500);

    // Watch for navigation
    watchForPageChanges();

    // Periodically check sidecar status (every 60s)
    setInterval(async () => {
      const health = await checkHealth();
      const newStatus = health ? true : false;
      if (newStatus !== sidecarStatus) {
        sidecarStatus = newStatus;
        updateButtonStatus(sidecarStatus);
        console.log(`[${PLUGIN_NAME}] Sidecar status changed: ${sidecarStatus ? 'connected' : 'disconnected'}`);
      }
    }, 60000);

    console.log(`[${PLUGIN_NAME}] Plugin initialized`);
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
