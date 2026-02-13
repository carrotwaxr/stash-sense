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

      // Get scene's existing performer StashDB IDs
      async getScenePerformerStashDBIds(sceneId) {
        const query = `
          query GetScenePerformers($id: ID!) {
            findScene(id: $id) {
              performers {
                id
                name
                stash_ids { endpoint stash_id }
              }
            }
          }
        `;
        try {
          const data = await SS.stashQuery(query, { id: sceneId });
          const performers = data?.findScene?.performers || [];
          return performers;
        } catch (e) {
          console.error('Failed to get scene performers:', e);
          return [];
        }
      },

      // Call the face recognition API via Python backend
      async identifyScene(sceneId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.('Connecting to Stash Sense...');

        // Get existing performer StashDB IDs for tagged-performer awareness
        const scenePerformers = await this.getScenePerformerStashDBIds(sceneId);
        const stashdbIds = [];
        for (const p of scenePerformers) {
          for (const sid of (p.stash_ids || [])) {
            if (sid.endpoint === 'https://stashdb.org/graphql') {
              stashdbIds.push(sid.stash_id);
            }
          }
        }

        const result = await SS.runPluginOperation('identify_scene', {
          scene_id: sceneId,
          sidecar_url: settings.sidecarUrl,
          top_k: settings.maxResults,
          scene_performer_stashdb_ids: stashdbIds,
          // max_distance and min_face_size use tuned defaults in backend (0.5, 60px)
        });

        if (result.error) {
          throw new Error(result.error);
        }

        // Attach scene performers for UI rendering
        result._scenePerformers = scenePerformers;

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

      // Call the face recognition API for a single image
      async identifyImage(imageId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.('Connecting to Stash Sense...');

        const result = await SS.runPluginOperation('identify_image', {
          image_id: imageId,
          sidecar_url: settings.sidecarUrl,
        });

        if (result.error) {
          throw new Error(result.error);
        }

        return result;
      },

      // Add performer to image
      async addPerformerToImage(imageId, performerId) {
        const getQuery = `
          query GetImage($id: ID!) {
            findImage(id: $id) {
              performers { id }
            }
          }
        `;

        const updateQuery = `
          mutation UpdateImage($id: ID!, $performer_ids: [ID!]) {
            imageUpdate(input: { id: $id, performer_ids: $performer_ids }) {
              id
            }
          }
        `;

        try {
          const getResult = await SS.stashQuery(getQuery, { id: imageId });
          const currentPerformers = getResult?.findImage?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: imageId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer to image:', e);
          return false;
        }
      },

      // Create the results modal
      createModal() {
        const existing = document.getElementById('ss-modal');
        if (existing) existing.remove();

        // Use inline styles to prevent Stash's Bootstrap CSS from
        // turning this into a drawer/sheet layout
        const modal = document.createElement('div');
        modal.id = 'ss-modal';
        modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:10000;display:flex;align-items:center;justify-content:center;';

        const backdrop = document.createElement('div');
        backdrop.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);';
        modal.appendChild(backdrop);

        const content = document.createElement('div');
        content.className = 'ss-modal-content';
        content.style.cssText = 'position:relative;background:var(--bs-body-bg, #1a1a1a);border-radius:8px;width:90%;max-width:700px;max-height:85vh;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,0.5);';

        const header = document.createElement('div');
        header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--bs-border-color, #333);';
        const title = document.createElement('h3');
        title.style.cssText = 'margin:0;font-size:18px;font-weight:600;color:var(--bs-body-color, #fff);';
        title.textContent = 'Stash Sense Results';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'ss-modal-close';
        closeBtn.style.cssText = 'background:none;border:none;font-size:24px;color:var(--bs-secondary-color, #888);cursor:pointer;padding:0;line-height:1;';
        closeBtn.setAttribute('aria-label', 'Close');
        closeBtn.innerHTML = '&times;';
        header.appendChild(title);
        header.appendChild(closeBtn);
        content.appendChild(header);

        const body = document.createElement('div');
        body.className = 'ss-modal-body';
        body.style.cssText = 'padding:20px;overflow-y:auto;flex:1;';
        body.innerHTML = `
          <div class="ss-loading">
            <div class="ss-spinner"></div>
            <p class="ss-loading-text">Connecting to Stash Sense...</p>
            <p class="ss-loading-detail"></p>
          </div>
          <div class="ss-results" style="display: none;"></div>
          <div class="ss-error" style="display: none;"></div>
        `;
        content.appendChild(body);
        modal.appendChild(content);

        document.body.appendChild(modal);

        const closeModal = () => modal.remove();
        closeBtn.addEventListener('click', closeModal);
        backdrop.addEventListener('click', closeModal);

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

        // Build set of StashDB IDs already tagged on this scene
        const taggedStashDBIds = new Set();
        const scenePerformers = results._scenePerformers || [];
        const scenePerformerLocalIds = new Set();
        for (const p of scenePerformers) {
          scenePerformerLocalIds.add(p.id);
          for (const sid of (p.stash_ids || [])) {
            if (sid.endpoint === 'https://stashdb.org/graphql') {
              taggedStashDBIds.add(sid.stash_id);
            }
          }
        }

        // Separate persons with multi-frame clusters from singletons
        const multiFrame = results.persons.filter(p => p.frame_count > 1 && p.best_match);
        const singleFrame = results.persons.filter(p => p.frame_count <= 1 && p.best_match);
        const unknown = results.persons.filter(p => !p.best_match);

        const clusterCount = multiFrame.length;
        const totalPersons = results.persons.filter(p => p.best_match).length;

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Analyzed <strong>${results.frames_analyzed}</strong> frames,
            detected <strong>${results.faces_detected}</strong> faces,
            found <strong>${clusterCount}</strong> distinct person(s)${singleFrame.length ? ` + ${singleFrame.length} single-frame detection(s)` : ''}.
          </p>
          <div class="ss-persons"></div>
          ${singleFrame.length ? '<div class="ss-singletons"></div>' : ''}
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        // Render multi-frame persons (high confidence clusters)
        for (const person of multiFrame) {
          const personDiv = await this._renderPerson(person, sceneId, taggedStashDBIds, scenePerformerLocalIds);
          personsDiv.appendChild(personDiv);
        }

        // Render single-frame detections collapsed by default
        if (singleFrame.length) {
          const singletonsDiv = resultsDiv.querySelector('.ss-singletons');
          const details = document.createElement('details');
          details.className = 'ss-singleton-section';
          details.innerHTML = `
            <summary class="ss-singleton-header">Single-frame detections (${singleFrame.length})</summary>
          `;
          const innerDiv = document.createElement('div');
          innerDiv.className = 'ss-singleton-list';
          for (const person of singleFrame) {
            const personDiv = await this._renderPerson(person, sceneId, taggedStashDBIds, scenePerformerLocalIds);
            innerDiv.appendChild(personDiv);
          }
          details.appendChild(innerDiv);
          singletonsDiv.appendChild(details);
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

      async _renderPerson(person, sceneId, taggedStashDBIds, scenePerformerLocalIds) {
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
          return personDiv;
        }

        const match = person.best_match;
        const confidence = this.distanceToConfidence(match.distance || (1 - match.confidence) || 0.5);
        const confidenceClass = SS.getConfidenceClass(confidence);

        // Check if already tagged (from API flag or local cross-reference)
        const isAlreadyTagged = match.already_tagged || taggedStashDBIds.has(match.stashdb_id);

        const localPerformer = await SS.findPerformerByStashDBId(match.stashdb_id);
        const isLocallyTagged = localPerformer && scenePerformerLocalIds.has(localPerformer.id);
        const showAlreadyTagged = isAlreadyTagged || isLocallyTagged;

        let actionsHtml;
        if (showAlreadyTagged) {
          actionsHtml = `<span class="ss-local-status ss-already-tagged">Already tagged on scene</span>`;
        } else if (localPerformer) {
          actionsHtml = `
            <button class="ss-btn ss-btn-add" data-performer-id="${localPerformer.id}" data-scene-id="${sceneId}">
              Add to Scene
            </button>
            <span class="ss-local-status">In library as: ${localPerformer.name}</span>`;
        } else {
          actionsHtml = `<span class="ss-local-status ss-not-in-library">Not in library</span>`;
        }

        personDiv.innerHTML = `
          <div class="ss-person-header">
            <span class="ss-person-label">Person ${person.person_id + 1}</span>
            <span class="ss-person-frames">${person.frame_count} appearances</span>
            ${showAlreadyTagged ? '<span class="ss-tagged-badge">Tagged</span>' : ''}
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
                ${actionsHtml}
              </div>
            </div>
          </div>
          ${person.all_matches && person.all_matches.length > 1 ? `
            <details class="ss-other-matches">
              <summary>Other possible matches (${person.all_matches.length - 1})</summary>
              <ul>
                ${person.all_matches.slice(1).map(m => {
                  const altConf = this.distanceToConfidence(m.distance || (1 - m.confidence) || 0.5);
                  const altTagged = m.already_tagged || taggedStashDBIds.has(m.stashdb_id);
                  return `
                    <li>
                      <a href="https://stashdb.org/performers/${m.stashdb_id}" target="_blank" rel="noopener">
                        ${m.name}
                      </a>
                      <span class="ss-alt-confidence">${altConf}%</span>
                      ${altTagged ? '<span class="ss-tagged-badge ss-tagged-badge-sm">Tagged</span>' : ''}
                    </li>
                  `;
                }).join('')}
              </ul>
            </details>
          ` : ''}
        `;

        return personDiv;
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

      async handleIdentifyImage() {
        const route = SS.getRoute();
        if (route.type !== 'image') return;

        const imageId = route.id;
        const modal = this.createModal();

        try {
          this.updateLoading(modal, 'Analyzing image...', 'Detecting faces');

          const results = await this.identifyImage(imageId, (stage) => {
            this.updateLoading(modal, stage);
          });

          this.updateLoading(modal, 'Processing results...');
          await this.renderImageResults(modal, results, imageId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Image analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      async renderImageResults(modal, results, imageId) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        if (!results.faces || results.faces.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No faces detected</p>
            <p class="ss-error-hint">The image may not contain clear face shots.</p>
          `;
          errorDiv.style.display = 'block';
          return;
        }

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Detected <strong>${results.face_count}</strong> face(s) in image.
          </p>
          <div class="ss-persons"></div>
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        for (let i = 0; i < results.faces.length; i++) {
          const face = results.faces[i];
          const personDiv = document.createElement('div');
          personDiv.className = 'ss-person';

          if (!face.matches || face.matches.length === 0) {
            personDiv.innerHTML = `
              <div class="ss-person-header">
                <span class="ss-person-label">Face ${i + 1}</span>
              </div>
              <p class="ss-no-match">No match found in database</p>
            `;
          } else {
            const match = face.matches[0];
            const confidence = this.distanceToConfidence(match.distance);
            const confidenceClass = SS.getConfidenceClass(confidence);
            const localPerformer = await SS.findPerformerByStashDBId(match.stashdb_id);

            personDiv.innerHTML = `
              <div class="ss-person-header">
                <span class="ss-person-label">Face ${i + 1}</span>
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
                      ? `<button class="ss-btn ss-btn-add" data-performer-id="${localPerformer.id}" data-image-id="${imageId}">
                           Add to Image
                         </button>
                         <span class="ss-local-status">In library as: ${localPerformer.name}</span>`
                      : `<span class="ss-local-status ss-not-in-library">Not in library</span>`
                    }
                  </div>
                </div>
              </div>
              ${face.matches.length > 1 ? `
                <details class="ss-other-matches">
                  <summary>Other possible matches (${face.matches.length - 1})</summary>
                  <ul>
                    ${face.matches.slice(1).map(m => {
                      const altConf = this.distanceToConfidence(m.distance);
                      return `<li>
                        <a href="https://stashdb.org/performers/${m.stashdb_id}" target="_blank" rel="noopener">${m.name}</a>
                        <span class="ss-alt-confidence">${altConf}%</span>
                      </li>`;
                    }).join('')}
                  </ul>
                </details>
              ` : ''}
            `;
          }

          personsDiv.appendChild(personDiv);
        }

        // Add click handlers for "Add to Image" buttons
        resultsDiv.querySelectorAll('.ss-btn-add').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const performerId = e.target.dataset.performerId;
            const targetImageId = e.target.dataset.imageId;
            btn.disabled = true;
            btn.textContent = 'Adding...';

            const success = await this.addPerformerToImage(targetImageId, performerId);
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

      createImageButton() {
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
        btn.addEventListener('click', () => this.handleIdentifyImage());
        return btn;
      },

      injectImageButton() {
        const route = SS.getRoute();
        if (route.type !== 'image') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.image-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.image-header .btn-group',
          '.detail-header-buttons',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createImageButton());
            console.log(`[${SS.PLUGIN_NAME}] Image button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createImageButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },

      // Call the gallery identification API
      async identifyGallery(galleryId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.('Connecting to Stash Sense...');

        const result = await SS.runPluginOperation('identify_gallery', {
          gallery_id: galleryId,
          sidecar_url: settings.sidecarUrl,
        });

        if (result.error) {
          throw new Error(result.error);
        }

        return result;
      },

      // Add performer to gallery
      async addPerformerToGallery(galleryId, performerId) {
        const getQuery = `
          query GetGallery($id: ID!) {
            findGallery(id: $id) {
              performers { id }
            }
          }
        `;

        const updateQuery = `
          mutation UpdateGallery($id: ID!, $performer_ids: [ID!]) {
            galleryUpdate(input: { id: $id, performer_ids: $performer_ids }) {
              id
            }
          }
        `;

        try {
          const getResult = await SS.stashQuery(getQuery, { id: galleryId });
          const currentPerformers = getResult?.findGallery?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: galleryId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer to gallery:', e);
          return false;
        }
      },

      async handleIdentifyGallery() {
        const route = SS.getRoute();
        if (route.type !== 'gallery') return;

        const galleryId = route.id;
        const modal = this.createModal();

        try {
          this.updateLoading(modal, 'Identifying performers in gallery...', 'This may take a while for large galleries');

          const results = await this.identifyGallery(galleryId, (stage) => {
            this.updateLoading(modal, stage);
          });

          this.updateLoading(modal, 'Processing results...');
          await this.renderGalleryResults(modal, results, galleryId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Gallery analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      async renderGalleryResults(modal, results, galleryId) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        if (!results.performers || results.performers.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No performers identified</p>
            <p class="ss-error-hint">
              Processed ${results.images_processed || 0}/${results.total_images || 0} images
              but no confident matches were found.
            </p>
          `;
          errorDiv.style.display = 'block';
          return;
        }

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Processed <strong>${results.images_processed}</strong>/${results.total_images} images,
            detected <strong>${results.faces_detected}</strong> faces,
            identified <strong>${results.performers.length}</strong> performer(s).
          </p>
          <div class="ss-gallery-actions-bar">
            <button class="ss-btn ss-btn-primary ss-accept-all-btn">Accept All</button>
          </div>
          <div class="ss-persons"></div>
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        for (const performer of results.performers) {
          const personDiv = document.createElement('div');
          personDiv.className = 'ss-person';

          const confidence = this.distanceToConfidence(performer.best_distance);
          const confidenceClass = SS.getConfidenceClass(confidence);

          const localPerformer = await SS.findPerformerByStashDBId(performer.performer_id);

          personDiv.innerHTML = `
            <div class="ss-person-header">
              <span class="ss-person-label">${performer.name}</span>
              <span class="ss-person-frames">Found in ${performer.image_count}/${results.total_images} images</span>
            </div>
            <div class="ss-match">
              <div class="ss-match-image">
                ${performer.image_url ? `<img src="${performer.image_url}" alt="${performer.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
              </div>
              <div class="ss-match-info">
                <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
                ${performer.country ? `<div class="ss-country">${performer.country}</div>` : ''}
                <div class="ss-links">
                  <a href="https://stashdb.org/performers/${performer.performer_id}" target="_blank" rel="noopener" class="ss-link">
                    View on StashDB
                  </a>
                </div>
                ${localPerformer ? `
                  <div class="ss-gallery-performer-actions" data-performer-id="${localPerformer.id}" data-stashdb-id="${performer.performer_id}">
                    <div class="ss-gallery-tag-toggle">
                      <label class="ss-toggle-label">
                        <input type="checkbox" class="ss-tag-images-toggle" />
                        <span>Also tag individual images</span>
                      </label>
                    </div>
                    <div class="ss-actions">
                      <button class="ss-btn ss-btn-add ss-gallery-accept-btn"
                              data-performer-id="${localPerformer.id}"
                              data-gallery-id="${galleryId}"
                              data-image-ids='${JSON.stringify(performer.image_ids)}'>
                        Add to Gallery
                      </button>
                      <span class="ss-local-status">In library as: ${localPerformer.name}</span>
                    </div>
                  </div>
                ` : `
                  <div class="ss-actions">
                    <span class="ss-local-status ss-not-in-library">Not in library</span>
                  </div>
                `}
              </div>
            </div>
          `;

          personsDiv.appendChild(personDiv);
        }

        // Click handlers for individual accept buttons
        resultsDiv.querySelectorAll('.ss-gallery-accept-btn').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const performerId = btn.dataset.performerId;
            const targetGalleryId = btn.dataset.galleryId;
            const imageIds = JSON.parse(btn.dataset.imageIds);
            const tagImages = btn.closest('.ss-gallery-performer-actions')
              ?.querySelector('.ss-tag-images-toggle')?.checked || false;

            btn.disabled = true;
            btn.textContent = 'Adding...';

            let success = await this.addPerformerToGallery(targetGalleryId, performerId);

            if (success && tagImages) {
              btn.textContent = `Tagging images...`;
              for (const imgId of imageIds) {
                await this.addPerformerToImage(imgId, performerId);
              }
            }

            if (success) {
              btn.textContent = tagImages ? `Added to gallery + ${imageIds.length} images` : 'Added to gallery!';
              btn.classList.add('ss-btn-success');
            } else {
              btn.textContent = 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
            }
          });
        });

        // Accept All handler
        resultsDiv.querySelector('.ss-accept-all-btn')?.addEventListener('click', async (e) => {
          const acceptAllBtn = e.target;
          acceptAllBtn.disabled = true;
          acceptAllBtn.textContent = 'Accepting...';

          const buttons = resultsDiv.querySelectorAll('.ss-gallery-accept-btn:not(:disabled)');
          for (const btn of buttons) {
            btn.click();
            // Small delay between operations
            await new Promise(r => setTimeout(r, 200));
          }

          acceptAllBtn.textContent = 'All accepted!';
          acceptAllBtn.classList.add('ss-btn-success');
        });

        resultsDiv.style.display = 'block';
      },

      createGalleryButton() {
        const status = SS.getSidecarStatus();
        const btn = SS.createElement('button', {
          className: 'ss-identify-btn btn btn-secondary',
          attrs: {
            title: status === false ? 'Stash Sense: Not connected' : 'Identify all performers in this gallery',
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
        btn.addEventListener('click', () => this.handleIdentifyGallery());
        return btn;
      },

      injectGalleryButton() {
        const route = SS.getRoute();
        if (route.type !== 'gallery') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.gallery-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.gallery-header .btn-group',
          '.detail-header-buttons',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createGalleryButton());
            console.log(`[${SS.PLUGIN_NAME}] Gallery button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createGalleryButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },
    };

    // ==================== Initialization ====================

    // Wait for a DOM element to appear, retrying with increasing delays
    function waitForElement(selector, callback, maxAttempts = 20) {
      let attempts = 0;
      function check() {
        const el = document.querySelector(selector);
        if (el) {
          callback(el);
        } else if (attempts < maxAttempts) {
          attempts++;
          setTimeout(check, 250);
        }
      }
      setTimeout(check, 300);
    }

    // Inject button into the appropriate toolbar for the current page type
    function injectButton(route) {
      if (document.querySelector('.ss-identify-btn')) return;

      const toolbarMap = {
        scene:   { selector: '.scene-toolbar-group:last-child',   create: () => FaceRecognition.createButton() },
        image:   { selector: '.image-toolbar-group:last-child',   create: () => FaceRecognition.createImageButton() },
        gallery: { selector: '.gallery-toolbar-group:last-child', create: () => FaceRecognition.createGalleryButton() },
      };

      const config = toolbarMap[route.type];
      if (!config) return;

      waitForElement(config.selector, (container) => {
        if (document.querySelector('.ss-identify-btn')) return; // re-check after wait
        container.appendChild(config.create());
        console.log(`[${SS.PLUGIN_NAME}] Button injected into ${config.selector}`);
      });
    }

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

      // Inject button for current page
      injectButton(SS.getRoute());

      // Watch for navigation
      SS.onNavigate((route) => {
        injectButton(route);
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
