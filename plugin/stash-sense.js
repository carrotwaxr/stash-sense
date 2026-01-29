/**
 * Stash Sense Main Entry Point
 *
 * Loads core module and feature modules:
 * - Face recognition (scene page integration)
 * - Recommendations dashboard (plugin page)
 */
(function() {
  'use strict';

  // Wait for core module
  function waitForCore(callback, attempts = 0) {
    if (window.StashSense) {
      callback();
    } else if (attempts < 50) {
      setTimeout(() => waitForCore(callback, attempts + 1), 100);
    } else {
      console.error('[Stash Sense] Core module failed to load');
    }
  }

  waitForCore(() => {
    const SS = window.StashSense;

    // ==================== Face Recognition Module ====================

    const FaceRecognition = {
      // Convert distance to confidence percentage
      distanceToConfidence(distance) {
        const clamped = Math.max(0, Math.min(1, distance));
        return Math.round((1 - clamped) * 100);
      },

      // Call the face recognition API via Python backend
      async identifyScene(sceneId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.('Connecting to Stash Sense...');

        const maxDistance = 1 - (settings.minConfidence / 100);

        const result = await SS.runPluginOperation('identify_scene', {
          scene_id: sceneId,
          sidecar_url: settings.sidecarUrl,
          top_k: settings.maxResults,
          max_distance: maxDistance,
        });

        if (result.error) {
          throw new Error(result.error);
        }

        return result;
      },

      // Add performer to scene
      async addPerformerToScene(sceneId, performerId) {
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
          const getResult = await SS.stashQuery(getQuery, { id: sceneId });
          const currentPerformers = getResult?.findScene?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: sceneId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer:', e);
          return false;
        }
      },

      // Create the results modal
      createModal() {
        const existing = document.getElementById('ss-modal');
        if (existing) existing.remove();

        const modal = SS.createElement('div', {
          id: 'ss-modal',
          className: 'ss-modal',
          innerHTML: `
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
          `,
        });

        document.body.appendChild(modal);

        const closeModal = () => modal.remove();
        modal.querySelector('.ss-modal-close').addEventListener('click', closeModal);
        modal.querySelector('.ss-modal-backdrop').addEventListener('click', closeModal);

        const escHandler = (e) => {
          if (e.key === 'Escape') {
            closeModal();
            document.removeEventListener('keydown', escHandler);
          }
        };
        document.addEventListener('keydown', escHandler);

        return modal;
      },

      updateLoading(modal, message, detail = '') {
        const loadingText = modal.querySelector('.ss-loading-text');
        const loadingDetail = modal.querySelector('.ss-loading-detail');
        if (loadingText) loadingText.textContent = message;
        if (loadingDetail) loadingDetail.textContent = detail;
      },

      async renderResults(modal, results, sceneId) {
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
            const confidence = this.distanceToConfidence(match.distance || (1 - match.confidence) || 0.5);
            const confidenceClass = SS.getConfidenceClass(confidence);

            const localPerformer = await SS.findPerformerByStashDBId(match.stashdb_id);

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
                      const altConf = this.distanceToConfidence(m.distance || (1 - m.confidence) || 0.5);
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

            const success = await this.addPerformerToScene(targetSceneId, performerId);
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
      },

      showError(modal, message) {
        const loading = modal.querySelector('.ss-loading');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        let title = 'Analysis Failed';
        let hint = 'Check plugin settings and ensure Stash Sense is running.';

        if (message.includes('Connection') || message.includes('connect')) {
          title = 'Connection Failed';
          hint = 'Could not connect to Stash Sense. Make sure the sidecar container is running.';
        } else if (message.includes('timeout') || message.includes('Timeout')) {
          title = 'Request Timed Out';
          hint = 'Scene analysis took too long.';
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
      },

      async handleIdentify() {
        const route = SS.getRoute();
        if (route.type !== 'scene') {
          alert('Could not determine scene ID');
          return;
        }

        const sceneId = route.id;
        const modal = this.createModal();

        try {
          this.updateLoading(modal, 'Fetching scene sprites...', 'This may take a moment');

          const results = await this.identifyScene(sceneId, (stage) => {
            this.updateLoading(modal, stage);
          });

          this.updateLoading(modal, 'Processing results...');
          await this.renderResults(modal, results, sceneId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      createButton() {
        const status = SS.getSidecarStatus();
        const btn = SS.createElement('button', {
          className: 'ss-identify-btn btn btn-secondary',
          attrs: {
            title: status === false ? 'Stash Sense: Not connected' : 'Identify performers using face recognition',
          },
          innerHTML: `
            <span class="ss-btn-icon ${status === true ? 'ss-connected' : status === false ? 'ss-disconnected' : ''}">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
              </svg>
            </span>
            <span class="ss-btn-text">Identify Performers</span>
          `,
        });
        btn.addEventListener('click', () => this.handleIdentify());
        return btn;
      },

      updateButtonStatus(connected) {
        document.querySelectorAll('.ss-identify-btn .ss-btn-icon').forEach(icon => {
          icon.classList.remove('ss-connected', 'ss-disconnected');
          if (connected === true) {
            icon.classList.add('ss-connected');
          } else if (connected === false) {
            icon.classList.add('ss-disconnected');
          }
        });
      },

      injectSceneButton() {
        const route = SS.getRoute();
        if (route.type !== 'scene') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.scene-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.scene-header .btn-group',
          '.detail-header-buttons',
          '.scene-operations',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createButton());
            console.log(`[${SS.PLUGIN_NAME}] Button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },
    };

    // ==================== Initialization ====================

    async function init() {
      console.log(`[${SS.PLUGIN_NAME}] Initializing...`);

      // Check sidecar health
      const health = await SS.checkHealth();
      if (health) {
        console.log(`[${SS.PLUGIN_NAME}] Sidecar connected: ${health.performer_count} performers`);
      } else {
        console.warn(`[${SS.PLUGIN_NAME}] Sidecar not available`);
      }

      // Initialize navigation watcher
      SS.initNavigationWatcher();

      // Inject scene button
      setTimeout(() => FaceRecognition.injectSceneButton(), 500);

      // Watch for navigation
      SS.onNavigate((route) => {
        if (route.type === 'scene') {
          setTimeout(() => FaceRecognition.injectSceneButton(), 300);
        }
      });

      // Periodic health check
      setInterval(async () => {
        const health = await SS.checkHealth();
        const newStatus = health ? true : false;
        if (newStatus !== SS.getSidecarStatus()) {
          SS.setSidecarStatus(newStatus);
          FaceRecognition.updateButtonStatus(newStatus);
        }
      }, 60000);

      console.log(`[${SS.PLUGIN_NAME}] Initialized`);
    }

    // Start
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  });
})();
