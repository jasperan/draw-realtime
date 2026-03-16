<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import OutputGallery from './OutputGallery.svelte';
  import MultiStyleTab from './MultiStyleTab.svelte';

  // Tab state
  let activeTab: 'processing' | 'multistyle' = 'processing';
  let processingTabButton: HTMLButtonElement;
  let multistyleTabButton: HTMLButtonElement;

  // State
  let settings: any = null;
  let selectedModel = '';
  let selectedPreset = '';
  let prompt = '';

  // MonarchRT generate mode
  let numFrames = 21;

  // Check if current model is a MonarchRT text-to-video model
  $: isMonarchRT = settings?.models?.[selectedModel]?.includes?.('MonarchRT') || false;
  $: isGenerateMode = selectedModel.startsWith('monarchrt-');

  // Video sources
  let serverVideos: any[] = [];
  let selectedServerVideo = '';
  let uploadedFile: any = null;

  // Processing state
  let currentJob: any = null;
  let isProcessing = false;
  let pollInterval: any = null;

  // Video URLs for playback
  let inputVideoUrl = '';
  let outputVideoUrl = '';

  // Video elements for sync playback
  let inputVideoEl: HTMLVideoElement;
  let outputVideoEl: HTMLVideoElement;
  let syncPlayback = true;

  // Job history
  let jobHistory: any[] = [];

  // Real-time preview state
  let previewInputUrl = '';
  let previewOutputUrl = '';
  let previewTimestamp = 0;
  let lastPreviewFrame = -1;

  const blankCaptionTrack = 'data:text/vtt;charset=utf-8,WEBVTT%0A%0A';

  function focusTab(tab: 'processing' | 'multistyle') {
    if (tab === 'processing') {
      processingTabButton?.focus();
      return;
    }
    multistyleTabButton?.focus();
  }

  function activateTab(tab: 'processing' | 'multistyle', shouldFocus = true) {
    activeTab = tab;
    if (shouldFocus) {
      focusTab(tab);
    }
  }

  function handleTabKeydown(event: KeyboardEvent, currentTab: 'processing' | 'multistyle') {
    switch (event.key) {
      case 'ArrowRight':
        event.preventDefault();
        activateTab(currentTab === 'processing' ? 'multistyle' : 'processing');
        break;
      case 'ArrowLeft':
        event.preventDefault();
        activateTab(currentTab === 'processing' ? 'multistyle' : 'processing');
        break;
      case 'Home':
        event.preventDefault();
        activateTab('processing');
        break;
      case 'End':
        event.preventDefault();
        activateTab('multistyle');
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        activateTab(currentTab);
        break;
    }
  }

  // Format seconds to MM:SS
  function formatTime(seconds: number): string {
    if (!seconds || seconds <= 0) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  // Handle preset selection
  function selectPreset(presetKey: string) {
    selectedPreset = presetKey;
    if (settings.presets && settings.presets[presetKey]) {
      prompt = settings.presets[presetKey].prompt;
    }
  }

  let loadError = '';

  onMount(async () => {
    try {
      // Load settings
      const res = await fetch('/api/settings');
      if (!res.ok) throw new Error(`Settings API returned ${res.status}`);
      settings = await res.json();
      selectedModel = settings.default_model;
      selectedPreset = settings.default_preset;
      if (settings.presets && settings.presets[selectedPreset]) {
        prompt = settings.presets[selectedPreset].prompt;
      } else {
        prompt = settings.default_prompt;
      }

      // Load server videos
      const videosRes = await fetch('/api/videos');
      if (videosRes.ok) {
        const videosData = await videosRes.json();
        serverVideos = videosData.videos || [];
      }

      // Load existing jobs
      await refreshJobs();
    } catch (e) {
      console.error('Failed to initialize:', e);
      loadError = e instanceof Error ? e.message : 'Failed to connect to server';
    }
  });

  async function refreshJobs() {
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      jobHistory = data.jobs || [];
    } catch (e) {
      console.error('Failed to load jobs:', e);
    }
  }

  async function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const error = await res.json();
        alert(`Upload failed: ${error.detail}`);
        return;
      }

      uploadedFile = await res.json();
      selectedServerVideo = '';
      inputVideoUrl = `/api/input/${uploadedFile.filename}`;
      outputVideoUrl = '';
    } catch (e) {
      console.error('Upload error:', e);
      alert('Failed to upload file');
    }
  }

  function selectServerVideo(video: any) {
    selectedServerVideo = video.name;
    uploadedFile = null;
    inputVideoUrl = `/api/input/${video.name}`;
    outputVideoUrl = '';
  }

  async function startProcessing() {
    if (isGenerateMode) {
      return startGenerating();
    }

    if (!inputVideoUrl) {
      alert('Please select or upload a video first');
      return;
    }

    isProcessing = true;
    currentJob = null;
    outputVideoUrl = '';
    lastPreviewFrame = -1;

    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('model', selectedModel);

    if (uploadedFile) {
      formData.append('uploaded_file', uploadedFile.filename);
    } else if (selectedServerVideo) {
      formData.append('video_name', selectedServerVideo);
    }

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const error = await res.json();
        alert(`Processing failed: ${error.detail}`);
        isProcessing = false;
        return;
      }

      currentJob = await res.json();
      startPolling();
    } catch (e) {
      console.error('Processing error:', e);
      alert('Failed to start processing');
      isProcessing = false;
    }
  }

  async function startGenerating() {
    if (!prompt.trim()) {
      alert('Please enter a text prompt for video generation');
      return;
    }

    isProcessing = true;
    currentJob = null;
    outputVideoUrl = '';
    inputVideoUrl = '';
    lastPreviewFrame = -1;

    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('model', selectedModel);
    formData.append('num_frames', numFrames.toString());

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const error = await res.json();
        alert(`Generation failed: ${error.detail}`);
        isProcessing = false;
        return;
      }

      currentJob = await res.json();
      startPolling();
    } catch (e) {
      console.error('Generation error:', e);
      alert('Failed to start generation');
      isProcessing = false;
    }
  }

  function startPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
    }

    pollInterval = setInterval(async () => {
      if (!currentJob) return;

      try {
        const res = await fetch(`/api/job/${currentJob.job_id}`);
        if (!res.ok) {
          stopPolling();
          return;
        }

        currentJob = await res.json();

        if (currentJob.status === 'processing' && currentJob.preview_frame && currentJob.input_frame) {
          const currentFrame = Math.floor(currentJob.current_frame / 100) * 100;
          if (currentFrame !== lastPreviewFrame) {
            lastPreviewFrame = currentFrame;
            previewTimestamp = Date.now();
            previewInputUrl = `/api/preview/${currentJob.input_frame}?t=${previewTimestamp}`;
            previewOutputUrl = `/api/preview/${currentJob.preview_frame}?t=${previewTimestamp}`;
          }
        }

        if (currentJob.status === 'completed') {
          stopPolling();
          outputVideoUrl = `/api/output/${currentJob.output_filename}`;
          previewInputUrl = '';
          previewOutputUrl = '';
          isProcessing = false;
          await refreshJobs();
        } else if (currentJob.status === 'failed') {
          stopPolling();
          previewInputUrl = '';
          previewOutputUrl = '';
          isProcessing = false;
          alert(`Processing failed: ${currentJob.error}`);
          await refreshJobs();
        }
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 500);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  function loadJob(job: any) {
    if (job.status !== 'completed') return;

    inputVideoUrl = `/api/input/${job.input_path}`;
    outputVideoUrl = `/api/output/${job.output_filename}`;
    prompt = job.prompt;
    selectedModel = job.model;
    currentJob = job;
  }

  async function deleteJob(job: any, event: Event) {
    event.stopPropagation();

    if (!confirm('Delete this job and its output?')) return;

    try {
      await fetch(`/api/job/${job.job_id}`, { method: 'DELETE' });
      await refreshJobs();

      if (currentJob?.job_id === job.job_id) {
        outputVideoUrl = '';
        currentJob = null;
      }
    } catch (e) {
      console.error('Delete error:', e);
    }
  }

  // Sync video playback
  function handleInputPlay() {
    if (syncPlayback && outputVideoEl && outputVideoUrl) {
      outputVideoEl.currentTime = inputVideoEl.currentTime;
      outputVideoEl.play();
    }
  }

  function handleInputPause() {
    if (syncPlayback && outputVideoEl) {
      outputVideoEl.pause();
    }
  }

  function handleInputSeek() {
    if (syncPlayback && outputVideoEl && outputVideoUrl) {
      outputVideoEl.currentTime = inputVideoEl.currentTime;
    }
  }

  function handleOutputPlay() {
    if (syncPlayback && inputVideoEl && inputVideoUrl) {
      inputVideoEl.currentTime = outputVideoEl.currentTime;
      inputVideoEl.play();
    }
  }

  function handleOutputPause() {
    if (syncPlayback && inputVideoEl) {
      inputVideoEl.pause();
    }
  }

  function handleOutputSeek() {
    if (syncPlayback && inputVideoEl && inputVideoUrl) {
      inputVideoEl.currentTime = outputVideoEl.currentTime;
    }
  }

  function playBoth() {
    if (inputVideoEl) inputVideoEl.play();
    if (outputVideoEl && outputVideoUrl) outputVideoEl.play();
  }

  function pauseBoth() {
    if (inputVideoEl) inputVideoEl.pause();
    if (outputVideoEl) outputVideoEl.pause();
  }

  function restartBoth() {
    if (inputVideoEl) {
      inputVideoEl.currentTime = 0;
      inputVideoEl.play();
    }
    if (outputVideoEl && outputVideoUrl) {
      outputVideoEl.currentTime = 0;
      outputVideoEl.play();
    }
  }

  onDestroy(() => {
    stopPolling();
  });
</script>

<main class="studio-container">
  <!-- Header -->
  <header class="studio-header">
    <div class="header-brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"/>
          <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
          <line x1="9" y1="9" x2="9.01" y2="9"/>
          <line x1="15" y1="9" x2="15.01" y2="9"/>
        </svg>
      </div>
      <div class="brand-text">
        <h1>Draw Realtime</h1>
        <span class="tagline">AI Video Studio</span>
      </div>
    </div>
    <p class="header-description">
      Transform videos with real-time AI-powered diffusion
    </p>
  </header>

  <!-- Navigation Tabs -->
  <nav aria-label="Workspace modes">
    <div class="tab-navigation" role="tablist" aria-label="Workspace modes">
      <button
        bind:this={processingTabButton}
        id="processing-tab"
        class="tab-button"
        class:active={activeTab === 'processing'}
        on:click={() => activateTab('processing', false)}
        on:keydown={(event) => handleTabKeydown(event, 'processing')}
        role="tab"
        type="button"
        aria-controls="processing-panel"
        aria-selected={activeTab === 'processing'}
        tabindex={activeTab === 'processing' ? 0 : -1}
      >
        <span class="tab-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
        </span>
        <span class="tab-label">Video Processing</span>
      </button>
      <button
        bind:this={multistyleTabButton}
        id="multistyle-tab"
        class="tab-button"
        class:active={activeTab === 'multistyle'}
        on:click={() => activateTab('multistyle', false)}
        on:keydown={(event) => handleTabKeydown(event, 'multistyle')}
        role="tab"
        type="button"
        aria-controls="multistyle-panel"
        aria-selected={activeTab === 'multistyle'}
        tabindex={activeTab === 'multistyle' ? 0 : -1}
      >
        <span class="tab-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <rect x="3" y="3" width="7" height="7"/>
            <rect x="14" y="3" width="7" height="7"/>
            <rect x="14" y="14" width="7" height="7"/>
            <rect x="3" y="14" width="7" height="7"/>
          </svg>
        </span>
        <span class="tab-label">Multi-Style</span>
      </button>
    </div>
  </nav>

  {#if loadError}
    <div class="error-toast" role="alert">
      <span class="error-icon">!</span>
      <span>Failed to connect to server: {loadError}</span>
    </div>
  {/if}

  {#if activeTab === 'multistyle'}
    <div id="multistyle-panel" role="tabpanel" aria-labelledby="multistyle-tab">
      <MultiStyleTab />
    </div>
  {:else}
    <div class="workspace" id="processing-panel" role="tabpanel" aria-labelledby="processing-tab">
      <!-- Real-time Preview Panel -->
      {#if isProcessing && previewInputUrl && previewOutputUrl}
        <section class="preview-panel" aria-label="Real-time processing preview">
          <div class="panel-header">
            <h2>Live Preview</h2>
            <div class="processing-badge">
              <span class="pulse-dot"></span>
              Processing
            </div>
          </div>

          <div class="comparison-view">
            <div class="frame-card">
              <span class="frame-label">Original</span>
              <div class="frame-image">
                <img src={previewInputUrl} alt="Original frame" loading="lazy"/>
              </div>
            </div>

            <div class="transform-arrow">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </div>

            <div class="frame-card">
              <span class="frame-label">Generated</span>
              <div class="frame-image">
                <img src={previewOutputUrl} alt="Generated frame" loading="lazy"/>
              </div>
            </div>
          </div>

          <div class="progress-metrics">
            <div class="metric">
              <span class="metric-value">{currentJob?.current_frame || 0}</span>
              <span class="metric-label">/ {currentJob?.total_frames || 0} frames</span>
            </div>
            <div class="metric-divider"></div>
            <div class="metric">
              <span class="metric-value">{currentJob?.processing_fps || 0}</span>
              <span class="metric-label">fps</span>
            </div>
            <div class="metric-divider"></div>
            <div class="metric">
              <span class="metric-value">{formatTime(currentJob?.eta_seconds || 0)}</span>
              <span class="metric-label">remaining</span>
            </div>
            <div class="metric-spacer"></div>
            <div class="metric-progress">{(currentJob?.progress || 0).toFixed(1)}%</div>
          </div>

          <div class="progress-track">
            <div class="progress-fill" style="width: {currentJob?.progress || 0}%"></div>
          </div>
        </section>
      {/if}

      <!-- Video Player Section -->
      <section class="player-section" aria-label="Video players">
        {#if isGenerateMode}
          <!-- MonarchRT Single View -->
          <div class="player-grid single">
            <div class="player-card wide">
              <div class="card-header">
                <h3>Generated Output</h3>
              </div>
              <div class="player-container">
                {#if outputVideoUrl}
                  <video
                    bind:this={outputVideoEl}
                    src={outputVideoUrl}
                    controls
                    loop
                    playsinline
                  >
                    <track kind="captions" srclang="en" label="Captions" src={blankCaptionTrack} />
                  </video>
                {:else if isProcessing}
                  <div class="loading-state">
                    <div class="spinner-artistic">
                      <div class="spinner-ring"></div>
                      <div class="spinner-ring"></div>
                      <div class="spinner-ring"></div>
                    </div>
                    <p class="loading-text">Generating with MonarchRT…</p>
                    <p class="loading-subtext">{(currentJob?.progress || 0).toFixed(1)}% complete</p>
                    {#if currentJob}
                      <div class="mini-progress">
                        <div class="mini-progress-fill" style="width: {currentJob.progress || 0}%"></div>
                      </div>
                    {/if}
                  </div>
                {:else}
                  <div class="empty-state">
                    <div class="empty-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M12 19l7-7 3 3-7 7-3-3z"/>
                        <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
                        <path d="M2 2l7.586 7.586"/>
                        <circle cx="11" cy="11" r="2"/>
                      </svg>
                    </div>
                    <p>Enter a prompt to generate video</p>
                  </div>
                {/if}
              </div>
            </div>
          </div>
        {:else}
          <!-- Side-by-side Comparison -->
          <div class="player-grid">
            <div class="player-card">
              <div class="card-header">
                <h3>Input</h3>
                {#if inputVideoUrl}
                  <span class="file-badge">Ready</span>
                {/if}
              </div>
              <div class="player-container">
                {#if inputVideoUrl}
                  <video
                    bind:this={inputVideoEl}
                    src={inputVideoUrl}
                    controls
                    loop
                    playsinline
                    on:play={handleInputPlay}
                    on:pause={handleInputPause}
                    on:seeked={handleInputSeek}
                  >
                    <track kind="captions" srclang="en" label="Captions" src={blankCaptionTrack} />
                  </video>
                {:else}
                  <div class="empty-state">
                    <div class="empty-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                      </svg>
                    </div>
                    <p>Select or upload a video</p>
                  </div>
                {/if}
              </div>
            </div>

            <div class="player-card">
              <div class="card-header">
                <h3>Output</h3>
                {#if outputVideoUrl}
                  <span class="file-badge success">Complete</span>
                {:else if isProcessing}
                  <span class="file-badge processing">Processing</span>
                {/if}
              </div>
              <div class="player-container">
                {#if outputVideoUrl}
                  <video
                    bind:this={outputVideoEl}
                    src={outputVideoUrl}
                    controls
                    loop
                    playsinline
                    on:play={handleOutputPlay}
                    on:pause={handleOutputPause}
                    on:seeked={handleOutputSeek}
                  >
                    <track kind="captions" srclang="en" label="Captions" src={blankCaptionTrack} />
                  </video>
                {:else if isProcessing}
                  <div class="loading-state">
                    <div class="spinner-artistic">
                      <div class="spinner-ring"></div>
                      <div class="spinner-ring"></div>
                    </div>
                    <p class="loading-text">Transforming…</p>
                    <p class="loading-subtext">See live preview above</p>
                  </div>
                {:else}
                  <div class="empty-state">
                    <div class="empty-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                        <polyline points="2 17 12 22 22 17"/>
                        <polyline points="2 12 12 17 22 12"/>
                      </svg>
                    </div>
                    <p>Output appears here</p>
                  </div>
                {/if}
              </div>
            </div>
          </div>
        {/if}

        <!-- Playback Controls -->
        {#if inputVideoUrl && outputVideoUrl}
          <div class="playback-bar">
            <button class="control-btn" on:click={playBoth} title="Play both">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
              <span>Play</span>
            </button>
            <button class="control-btn" on:click={pauseBoth} title="Pause both">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
              </svg>
              <span>Pause</span>
            </button>
            <button class="control-btn" on:click={restartBoth} title="Restart">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"/>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
              <span>Restart</span>
            </button>
            <div class="sync-toggle">
              <label class="toggle-switch">
                <input type="checkbox" bind:checked={syncPlayback} />
                <span class="toggle-slider"></span>
              </label>
              <span class="toggle-label">Sync playback</span>
            </div>
          </div>
        {/if}
      </section>

      <!-- Control Panel -->
      <section class="control-panel" aria-label="Processing controls">
        {#if isGenerateMode}
          <div class="mode-indicator generate">
            <span class="mode-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 19l7-7 3 3-7 7-3-3z"/>
                <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
              </svg>
            </span>
            <span class="mode-text">MonarchRT Text-to-Video Mode: Generate video from text prompts</span>
          </div>
        {/if}

        <div class="controls-grid">
          {#if !isGenerateMode}
            <fieldset class="control-group wide">
              <legend class="control-label">Video Source</legend>
              <div class="source-controls">
                <label class="upload-button">
                  <input id="video-upload-input" name="videoUpload" type="file" accept="video/*" on:change={handleFileUpload} />
                  <span class="upload-content">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    Upload MP4
                  </span>
                </label>

                {#if serverVideos.length > 0}
                  <div class="select-wrapper">
                    <select
                      id="video-library-select"
                      name="videoLibrary"
                      value={selectedServerVideo}
                      aria-label="Select video from library"
                      on:change={(e) => {
                        const video = serverVideos.find(v => v.name === e.currentTarget.value);
                        if (video) selectServerVideo(video);
                      }}
                    >
                      <option value="">Select from library…</option>
                      {#each serverVideos as video}
                        <option value={video.name}>{video.name} ({video.duration}s)</option>
                      {/each}
                    </select>
                    <span class="select-arrow">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <polyline points="6 9 12 15 18 9"/>
                      </svg>
                    </span>
                  </div>
                {/if}
              </div>

              {#if uploadedFile}
                <span class="selected-indicator">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  {uploadedFile.original_name}
                </span>
              {:else if selectedServerVideo}
                <span class="selected-indicator">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  {selectedServerVideo}
                </span>
              {/if}
            </fieldset>
          {/if}

          <div class="control-group">
            <label class="control-label" for="model-select">AI Model</label>
            <div class="select-wrapper">
              {#if settings}
                <select id="model-select" name="model" bind:value={selectedModel}>
                  {#each Object.entries(settings.models) as [key, desc]}
                    <option value={key}>{desc}</option>
                  {/each}
                </select>
              {:else}
                <select id="model-select" name="model" disabled>
                  <option>Loading models…</option>
                </select>
              {/if}
              <span class="select-arrow">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                  <polyline points="6 9 12 15 18 9"/>
                </svg>
              </span>
            </div>
          </div>

          {#if isGenerateMode}
            <div class="control-group">
              <label class="control-label" for="duration-select">Duration</label>
              <div class="select-wrapper">
                <select id="duration-select" name="duration" bind:value={numFrames}>
                  <option value={21}>21 frames (~1.3s)</option>
                  <option value={41}>41 frames (~2.6s)</option>
                  <option value={61}>61 frames (~3.8s)</option>
                  <option value={81}>81 frames (~5s)</option>
                </select>
                <span class="select-arrow">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                    <polyline points="6 9 12 15 18 9"/>
                  </svg>
                </span>
              </div>
            </div>
          {/if}
        </div>

        {#if !isGenerateMode}
          <fieldset class="control-group full">
            <legend class="control-label">Style Preset</legend>
            <div class="preset-grid">
              {#if settings?.presets}
                {#each Object.entries(settings.presets) as [key, preset]}
                  <button
                    class="preset-chip"
                    class:active={selectedPreset === key}
                    on:click={() => selectPreset(key)}
                    type="button"
                  >
                    {preset.description}
                  </button>
                {/each}
              {/if}
            </div>
          </fieldset>
        {/if}

        <div class="control-group full">
          <label class="control-label" for="prompt-input">
            {isGenerateMode ? 'Describe your video' : 'Transformation prompt'}
          </label>
          <input
            id="prompt-input"
            name="prompt"
            type="text"
            class="prompt-input"
            bind:value={prompt}
            autocomplete="off"
            placeholder={isGenerateMode
              ? 'A golden retriever running through wildflowers, cinematic lighting…'
              : 'oil painting, vibrant colors, masterpiece quality…'}
          />
        </div>

        <button
          class="action-button"
          class:generate={isGenerateMode}
          on:click={startProcessing}
          disabled={isProcessing || (!isGenerateMode && !inputVideoUrl)}
          type="button"
        >
          {#if isProcessing}
            <span class="btn-spinner"></span>
            <span>{isGenerateMode ? 'Generating…' : 'Processing…'}</span>
          {:else}
            <span>{isGenerateMode ? 'Generate Video' : 'Process Video'}</span>
          {/if}
        </button>
      </section>

      <!-- Job History -->
      {#if jobHistory.length > 0}
        <section class="history-panel" aria-label="Job history">
          <h3 class="panel-title">Recent Jobs</h3>
          <div class="job-list">
            {#each jobHistory as job}
              <button
                class="job-item"
                class:completed={job.status === 'completed'}
                class:failed={job.status === 'failed'}
                class:processing={job.status === 'processing'}
                on:click={() => loadJob(job)}
                disabled={job.status !== 'completed'}
                type="button"
              >
                <div class="job-content">
                  <span class="job-name">{job.input_path}</span>
                  <span class="job-prompt">{job.prompt.substring(0, 45)}{job.prompt.length > 45 ? '...' : ''}</span>
                </div>
                <div class="job-meta">
                  {#if job.status === 'processing'}
                    <span class="status-badge processing">{(job.progress || 0).toFixed(0)}%</span>
                  {:else if job.status === 'completed'}
                    <span class="status-badge completed">Done</span>
                  {:else if job.status === 'failed'}
                    <span class="status-badge failed">Failed</span>
                  {:else}
                    <span class="status-badge pending">Queued</span>
                  {/if}
                  {#if job.status === 'completed' || job.status === 'failed'}
                    <button
                      class="delete-action"
                      on:click={(e) => deleteJob(job, e)}
                      title="Delete job"
                      type="button"
                    >
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                      </svg>
                    </button>
                  {/if}
                </div>
              </button>
            {/each}
          </div>
        </section>
      {/if}

      <!-- Output Gallery -->
      <OutputGallery />
    </div>
  {/if}
</main>

<style>
  /* Container */
  .studio-container {
    max-width: 1440px;
    margin: 0 auto;
    padding: var(--space-xl);
    position: relative;
    z-index: 2;
  }

  @media (max-width: 768px) {
    .studio-container {
      padding: var(--space-md);
    }
  }

  /* Header */
  .studio-header {
    text-align: center;
    margin-bottom: var(--space-3xl);
    padding: var(--space-2xl) 0;
  }

  .header-brand {
    display: inline-flex;
    align-items: center;
    gap: var(--space-md);
    margin-bottom: var(--space-md);
  }

  .brand-icon {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, var(--color-accent-primary), #f97316);
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.3);
  }

  .brand-icon svg {
    width: 32px;
    height: 32px;
    color: white;
  }

  .brand-text {
    text-align: left;
  }

  .brand-text h1 {
    font-family: var(--font-display);
    font-size: var(--text-3xl);
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
    letter-spacing: -0.02em;
  }

  .tagline {
    font-size: var(--text-sm);
    color: var(--color-accent-secondary);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 500;
  }

  .header-description {
    color: var(--color-text-secondary);
    font-size: var(--text-lg);
    margin: 0;
  }

  /* Tab Navigation */
  .tab-navigation {
    display: flex;
    gap: var(--space-xs);
    margin-bottom: var(--space-2xl);
    padding: var(--space-xs);
    background: var(--color-bg-secondary);
    border-radius: var(--radius-xl);
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
    border: 1px solid var(--color-border);
  }

  .tab-button {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-md) var(--space-xl);
    background: transparent;
    border: none;
    border-radius: var(--radius-lg);
    color: var(--color-text-secondary);
    font-family: var(--font-body);
    font-size: var(--text-base);
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-base);
  }

  .tab-button:hover {
    color: var(--color-text-primary);
    background: var(--color-bg-hover);
  }

  .tab-button:focus-visible,
  .control-btn:focus-visible,
  .preset-chip:focus-visible,
  .action-button:focus-visible,
  .select-wrapper select:focus-visible,
  .prompt-input:focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.9);
    outline-offset: 2px;
  }

  .tab-button.active {
    background: linear-gradient(135deg, var(--color-accent-primary), #f97316);
    color: white;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.35);
  }

  .tab-icon svg {
    width: 18px;
    height: 18px;
  }

  @media (max-width: 640px) {
    .tab-button {
      padding: var(--space-sm) var(--space-lg);
    }

    .tab-label {
      display: none;
    }
  }

  /* Error Toast */
  .error-toast {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    padding: var(--space-md) var(--space-lg);
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: var(--radius-md);
    color: #fca5a5;
    margin-bottom: var(--space-xl);
  }

  .error-icon {
    width: 24px;
    height: 24px;
    background: var(--color-error);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
  }

  /* Workspace */
  .workspace {
    display: flex;
    flex-direction: column;
    gap: var(--space-xl);
  }

  /* Preview Panel */
  .preview-panel {
    background: linear-gradient(145deg, var(--color-bg-secondary), var(--color-bg-tertiary));
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
    padding: var(--space-xl);
    box-shadow: var(--shadow-lg);
    animation: slideDown 0.4s ease-out;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-lg);
  }

  .panel-header h2 {
    font-family: var(--font-display);
    font-size: var(--text-xl);
    font-weight: 500;
    color: var(--color-text-primary);
    margin: 0;
  }

  .processing-badge {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-xs) var(--space-md);
    background: var(--color-accent-subtle);
    border: 1px solid var(--color-accent-glow);
    border-radius: 100px;
    color: var(--color-accent-secondary);
    font-size: var(--text-sm);
    font-weight: 500;
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    background: var(--color-accent-primary);
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  .comparison-view {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: var(--space-lg);
    align-items: center;
    margin-bottom: var(--space-lg);
  }

  @media (max-width: 900px) {
    .comparison-view {
      grid-template-columns: 1fr;
    }

    .transform-arrow {
      transform: rotate(90deg);
    }
  }

  .frame-card {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
  }

  .frame-label {
    font-size: var(--text-sm);
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .frame-image {
    aspect-ratio: 1;
    background: var(--color-bg-primary);
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--color-border);
  }

  .frame-image img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .transform-arrow {
    color: var(--color-accent-primary);
  }

  .transform-arrow svg {
    width: 32px;
    height: 32px;
  }

  .progress-metrics {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    margin-bottom: var(--space-md);
    flex-wrap: wrap;
  }

  .metric {
    display: flex;
    align-items: baseline;
    gap: var(--space-xs);
  }

  .metric-value {
    font-family: var(--font-display);
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .metric-label {
    font-size: var(--text-sm);
    color: var(--color-text-muted);
  }

  .metric-divider {
    width: 1px;
    height: 20px;
    background: var(--color-border);
  }

  .metric-spacer {
    flex: 1;
  }

  .metric-progress {
    font-family: var(--font-display);
    font-size: var(--text-2xl);
    font-weight: 600;
    color: var(--color-accent-primary);
  }

  .progress-track {
    height: 6px;
    background: var(--color-bg-primary);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-accent-primary), #f97316);
    border-radius: 3px;
    transition: width 0.3s ease;
    box-shadow: 0 0 20px var(--color-accent-glow);
  }

  /* Player Section */
  .player-section {
    display: flex;
    flex-direction: column;
    gap: var(--space-lg);
  }

  .player-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--space-lg);
  }

  .player-grid.single {
    grid-template-columns: 1fr;
    max-width: 900px;
    margin: 0 auto;
    width: 100%;
  }

  @media (max-width: 900px) {
    .player-grid {
      grid-template-columns: 1fr;
    }
  }

  .player-card {
    background: var(--color-bg-card);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
    overflow: hidden;
    backdrop-filter: blur(10px);
    transition: all var(--transition-base);
  }

  .player-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: var(--shadow-lg);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-md) var(--space-lg);
    background: var(--color-bg-secondary);
    border-bottom: 1px solid var(--color-border);
  }

  .card-header h3 {
    font-family: var(--font-display);
    font-size: var(--text-base);
    font-weight: 500;
    color: var(--color-text-primary);
    margin: 0;
  }

  .file-badge {
    padding: var(--space-xs) var(--space-sm);
    background: var(--color-bg-tertiary);
    border-radius: 100px;
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .file-badge.success {
    background: rgba(16, 185, 129, 0.15);
    color: var(--color-success);
  }

  .file-badge.processing {
    background: var(--color-accent-subtle);
    color: var(--color-accent-secondary);
  }

  .player-container {
    aspect-ratio: 1;
    background: var(--color-bg-primary);
    position: relative;
  }

  .player-card.wide .player-container {
    aspect-ratio: 16/9;
  }

  .player-container video {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  /* Empty & Loading States */
  .empty-state,
  .loading-state {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-md);
    color: var(--color-text-muted);
    padding: var(--space-xl);
    text-align: center;
  }

  .empty-icon {
    width: 64px;
    height: 64px;
    color: var(--color-text-muted);
    opacity: 0.5;
  }

  .empty-icon svg {
    width: 100%;
    height: 100%;
  }

  .empty-state p {
    margin: 0;
    font-size: var(--text-base);
  }

  /* Artistic Spinner */
  .spinner-artistic {
    position: relative;
    width: 60px;
    height: 60px;
  }

  .spinner-ring {
    position: absolute;
    inset: 0;
    border: 2px solid transparent;
    border-top-color: var(--color-accent-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .spinner-ring:nth-child(2) {
    inset: 8px;
    border-top-color: var(--color-accent-secondary);
    animation-duration: 1.5s;
    animation-direction: reverse;
  }

  .spinner-ring:nth-child(3) {
    inset: 16px;
    border-top-color: #f97316;
    animation-duration: 2s;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .loading-text {
    font-family: var(--font-display);
    font-size: var(--text-lg);
    color: var(--color-text-primary);
    margin: 0;
  }

  .loading-subtext {
    font-size: var(--text-sm);
    color: var(--color-text-muted);
    margin: 0;
  }

  .mini-progress {
    width: 200px;
    height: 4px;
    background: var(--color-bg-secondary);
    border-radius: 2px;
    overflow: hidden;
    margin-top: var(--space-sm);
  }

  .mini-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-accent-primary), #f97316);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  /* Playback Bar */
  .playback-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-md);
    padding: var(--space-md);
    background: var(--color-bg-secondary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
  }

  .control-btn {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-lg);
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-text-primary);
    font-family: var(--font-body);
    font-size: var(--text-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .control-btn:hover {
    background: var(--color-bg-hover);
    border-color: rgba(255, 255, 255, 0.2);
  }

  .control-btn svg {
    width: 16px;
    height: 16px;
  }

  .sync-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    margin-left: var(--space-md);
    padding-left: var(--space-md);
    border-left: 1px solid var(--color-border);
  }

  .toggle-switch {
    position: relative;
    width: 44px;
    height: 24px;
    cursor: pointer;
  }

  .toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .toggle-slider {
    position: absolute;
    inset: 0;
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: 24px;
    transition: all var(--transition-base);
  }

  .toggle-slider::before {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 18px;
    height: 18px;
    background: var(--color-text-secondary);
    border-radius: 50%;
    transition: all var(--transition-base);
  }

  .toggle-switch input:checked + .toggle-slider {
    background: var(--color-accent-subtle);
    border-color: var(--color-accent-glow);
  }

  .toggle-switch input:checked + .toggle-slider::before {
    transform: translateX(20px);
    background: var(--color-accent-primary);
  }

  .toggle-label {
    font-size: var(--text-sm);
    color: var(--color-text-secondary);
  }

  @media (max-width: 640px) {
    .playback-bar {
      flex-wrap: wrap;
    }

    .sync-toggle {
      margin-left: 0;
      padding-left: 0;
      border-left: none;
      width: 100%;
      justify-content: center;
    }
  }

  /* Control Panel */
  .control-panel {
    background: var(--color-bg-card);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
    padding: var(--space-xl);
    backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    gap: var(--space-xl);
  }

  .mode-indicator {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    padding: var(--space-md) var(--space-lg);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
  }

  .mode-indicator.generate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(249, 115, 22, 0.1));
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: var(--color-accent-secondary);
  }

  .mode-icon svg {
    width: 20px;
    height: 20px;
  }

  .controls-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: var(--space-lg);
  }

  @media (max-width: 768px) {
    .controls-grid {
      grid-template-columns: 1fr;
    }
  }

  .control-group {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
  }

  fieldset.control-group {
    border: none;
    padding: 0;
    margin: 0;
    min-width: 0;
  }

  .control-group.wide {
    grid-column: span 1;
  }

  .control-group.full {
    grid-column: 1 / -1;
  }

  .control-label {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0;
  }

  .source-controls {
    display: flex;
    gap: var(--space-md);
    flex-wrap: wrap;
  }

  .upload-button {
    position: relative;
    cursor: pointer;
    flex-shrink: 0;
  }

  .upload-button input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }

  .upload-content {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-md) var(--space-lg);
    background: var(--color-bg-tertiary);
    border: 1px dashed var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-text-primary);
    font-size: var(--text-sm);
    font-weight: 500;
    transition: all var(--transition-fast);
  }

  .upload-button:hover .upload-content,
  .upload-button:focus-within .upload-content {
    background: var(--color-bg-hover);
    border-color: var(--color-accent-primary);
    color: var(--color-accent-secondary);
  }

  .upload-content svg {
    width: 18px;
    height: 18px;
  }

  .select-wrapper {
    position: relative;
    flex: 1;
    min-width: 200px;
  }

  .select-wrapper select {
    width: 100%;
    padding: var(--space-md) var(--space-xl) var(--space-md) var(--space-md);
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-text-primary);
    font-family: var(--font-body);
    font-size: var(--text-sm);
    cursor: pointer;
    appearance: none;
    transition: all var(--transition-fast);
  }

  .select-wrapper select:hover,
  .select-wrapper select:focus {
    border-color: var(--color-border-focus);
  }

  .select-arrow {
    position: absolute;
    right: var(--space-md);
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
    color: var(--color-text-muted);
  }

  .select-arrow svg {
    width: 16px;
    height: 16px;
  }

  .selected-indicator {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) 0;
    color: var(--color-accent-secondary);
    font-size: var(--text-sm);
  }

  .selected-indicator svg {
    width: 16px;
    height: 16px;
  }

  /* Preset Grid */
  .preset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: var(--space-sm);
  }

  .preset-chip {
    padding: var(--space-sm) var(--space-md);
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-text-secondary);
    font-family: var(--font-body);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: all var(--transition-fast);
    text-align: center;
  }

  .preset-chip:hover {
    background: var(--color-bg-hover);
    border-color: rgba(255, 255, 255, 0.15);
    color: var(--color-text-primary);
  }

  .preset-chip.active {
    background: linear-gradient(135deg, var(--color-accent-primary), #f97316);
    border-color: transparent;
    color: white;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.35);
  }

  /* Prompt Input */
  .prompt-input {
    width: 100%;
    padding: var(--space-md) var(--space-lg);
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-text-primary);
    font-family: var(--font-body);
    font-size: var(--text-base);
    transition: all var(--transition-fast);
  }

  .prompt-input:hover,
  .prompt-input:focus {
    border-color: var(--color-border-focus);
    outline: none;
  }

  .prompt-input::placeholder {
    color: var(--color-text-muted);
  }

  /* Action Button */
  .action-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-md);
    padding: var(--space-lg) var(--space-2xl);
    background: linear-gradient(135deg, var(--color-accent-primary), #f97316);
    border: none;
    border-radius: var(--radius-md);
    color: white;
    font-family: var(--font-display);
    font-size: var(--text-lg);
    font-weight: 600;
    cursor: pointer;
    transition: all var(--transition-base);
    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.35);
    align-self: center;
    min-width: 240px;
  }

  .action-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(245, 158, 11, 0.45);
  }

  .action-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .action-button.generate {
    background: linear-gradient(135deg, #10b981, #059669);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
  }

  .action-button.generate:hover:not(:disabled) {
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.45);
  }

  .btn-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  /* History Panel */
  .history-panel {
    background: var(--color-bg-card);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
    padding: var(--space-xl);
    backdrop-filter: blur(10px);
  }

  .panel-title {
    font-family: var(--font-display);
    font-size: var(--text-lg);
    font-weight: 500;
    color: var(--color-text-primary);
    margin: 0 0 var(--space-lg) 0;
  }

  .job-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
    max-height: 280px;
    overflow-y: auto;
  }

  .job-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-md);
    background: var(--color-bg-tertiary);
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    text-align: left;
    cursor: pointer;
    transition: all var(--transition-fast);
    width: 100%;
  }

  .job-item:hover:not(:disabled) {
    background: var(--color-bg-hover);
    border-color: var(--color-border);
  }

  .job-item:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .job-item.completed {
    border-left: 3px solid var(--color-success);
  }

  .job-item.failed {
    border-left: 3px solid var(--color-error);
  }

  .job-item.processing {
    border-left: 3px solid var(--color-accent-primary);
  }

  .job-content {
    display: flex;
    flex-direction: column;
    gap: var(--space-xs);
    min-width: 0;
  }

  .job-name {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--color-text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .job-prompt {
    font-size: var(--text-xs);
    color: var(--color-text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .job-meta {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    flex-shrink: 0;
  }

  .status-badge {
    padding: var(--space-xs) var(--space-sm);
    border-radius: 100px;
    font-size: var(--text-xs);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .status-badge.completed {
    background: rgba(16, 185, 129, 0.15);
    color: var(--color-success);
  }

  .status-badge.failed {
    background: rgba(239, 68, 68, 0.15);
    color: var(--color-error);
  }

  .status-badge.processing {
    background: var(--color-accent-subtle);
    color: var(--color-accent-secondary);
  }

  .status-badge.pending {
    background: var(--color-bg-secondary);
    color: var(--color-text-muted);
  }

  .delete-action {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: var(--radius-sm);
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all var(--transition-fast);
    padding: 0;
  }

  .delete-action:hover {
    background: rgba(239, 68, 68, 0.15);
    color: var(--color-error);
  }

  .delete-action svg {
    width: 14px;
    height: 14px;
  }
</style>
