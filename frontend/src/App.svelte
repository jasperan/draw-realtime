<script lang="ts">
  import { onMount } from 'svelte';

  // State
  let settings: any = null;
  let selectedModel = '';
  let selectedPreset = '';
  let prompt = '';

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

  onMount(async () => {
    // Load settings
    const res = await fetch('/api/settings');
    settings = await res.json();
    selectedModel = settings.default_model;
    selectedPreset = settings.default_preset;
    // Set initial prompt from default preset
    if (settings.presets && settings.presets[selectedPreset]) {
      prompt = settings.presets[selectedPreset].prompt;
    } else {
      prompt = settings.default_prompt;
    }

    // Load server videos
    const videosRes = await fetch('/api/videos');
    const videosData = await videosRes.json();
    serverVideos = videosData.videos || [];

    // Load existing jobs
    await refreshJobs();
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
      selectedServerVideo = ''; // Deselect server video
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
    if (!inputVideoUrl) {
      alert('Please select or upload a video first');
      return;
    }

    isProcessing = true;
    currentJob = null;
    outputVideoUrl = '';

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

        // Update real-time preview frames with cache-busting
        if (currentJob.status === 'processing' && currentJob.preview_frame && currentJob.input_frame) {
          previewTimestamp = Date.now();
          previewInputUrl = `/api/preview/${currentJob.input_frame}?t=${previewTimestamp}`;
          previewOutputUrl = `/api/preview/${currentJob.preview_frame}?t=${previewTimestamp}`;
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
</script>

<main>
  <header>
    <h1>StreamDiffusion Video-to-Video</h1>
    <p class="subtitle">Transform videos with AI-powered diffusion</p>
  </header>

  <div class="container">
    <!-- Real-time Frame Comparison (during processing) -->
    {#if isProcessing && previewInputUrl && previewOutputUrl}
      <div class="realtime-comparison">
        <h3>Real-time Processing Preview</h3>
        <div class="comparison-grid">
          <div class="comparison-frame">
            <span class="frame-title">Original Frame</span>
            <img src={previewInputUrl} alt="Original frame" />
          </div>
          <div class="comparison-arrow">→</div>
          <div class="comparison-frame">
            <span class="frame-title">Generated Frame</span>
            <img src={previewOutputUrl} alt="Generated frame" />
          </div>
        </div>
        <div class="comparison-stats">
          <span>Frame {currentJob?.current_frame || 0} / {currentJob?.total_frames || 0}</span>
          <span class="separator">|</span>
          <span>{currentJob?.processing_fps || 0} fps</span>
          <span class="separator">|</span>
          <span>ETA: {formatTime(currentJob?.eta_seconds || 0)}</span>
          <span class="separator">|</span>
          <span class="progress-text">{(currentJob?.progress || 0).toFixed(1)}%</span>
        </div>
        <div class="progress-container large">
          <div class="progress-bar" style="width: {currentJob?.progress || 0}%"></div>
        </div>
      </div>
    {/if}

    <!-- Video Comparison -->
    <div class="video-grid">
      <div class="video-box">
        <h3>Input Video</h3>
        {#if inputVideoUrl}
          <video
            bind:this={inputVideoEl}
            src={inputVideoUrl}
            controls
            loop
            on:play={handleInputPlay}
            on:pause={handleInputPause}
            on:seeked={handleInputSeek}
          ></video>
        {:else}
          <div class="placeholder">
            <span>Select or upload a video to begin</span>
          </div>
        {/if}
      </div>

      <div class="video-box">
        <h3>Output Video</h3>
        {#if outputVideoUrl}
          <video
            bind:this={outputVideoEl}
            src={outputVideoUrl}
            controls
            loop
            on:play={handleOutputPlay}
            on:pause={handleOutputPause}
            on:seeked={handleOutputSeek}
          ></video>
        {:else if isProcessing}
          <div class="placeholder processing-placeholder">
            <div class="processing-indicator">
              <div class="spinner"></div>
              <span>Processing in progress...</span>
              <span class="processing-detail">See real-time preview above</span>
            </div>
          </div>
        {:else}
          <div class="placeholder">
            <span>Output will appear here after processing</span>
          </div>
        {/if}
      </div>
    </div>

    <!-- Playback Controls -->
    {#if inputVideoUrl && outputVideoUrl}
      <div class="playback-controls">
        <button on:click={playBoth}>Play Both</button>
        <button on:click={pauseBoth}>Pause Both</button>
        <button on:click={restartBoth}>Restart</button>
        <label class="sync-toggle">
          <input type="checkbox" bind:checked={syncPlayback} />
          Sync Playback
        </label>
      </div>
    {/if}

    <!-- Controls Panel -->
    <div class="controls">
      <div class="control-row">
        <!-- Source Selection -->
        <div class="control-group source-group">
          <label>Video Source</label>
          <div class="source-options">
            <label class="file-button">
              Upload MP4
              <input type="file" accept="video/*" on:change={handleFileUpload} />
            </label>

            {#if serverVideos.length > 0}
              <select
                value={selectedServerVideo}
                on:change={(e) => {
                  const video = serverVideos.find(v => v.name === e.currentTarget.value);
                  if (video) selectServerVideo(video);
                }}
              >
                <option value="" disabled>Select Server Video</option>
                {#each serverVideos as video}
                  <option value={video.name}>{video.name} ({video.duration}s)</option>
                {/each}
              </select>
            {/if}
          </div>

          {#if uploadedFile}
            <span class="selected-file">Uploaded: {uploadedFile.original_name}</span>
          {:else if selectedServerVideo}
            <span class="selected-file">Selected: {selectedServerVideo}</span>
          {/if}
        </div>

        <!-- Model Selection -->
        <div class="control-group">
          <label>Model</label>
          {#if settings}
            <select bind:value={selectedModel}>
              {#each Object.entries(settings.models) as [key, desc]}
                <option value={key}>{desc}</option>
              {/each}
            </select>
          {/if}
        </div>
      </div>

      <!-- Style Preset -->
      <div class="control-group">
        <label>Style Preset</label>
        <div class="preset-grid">
          {#if settings?.presets}
            {#each Object.entries(settings.presets) as [key, preset]}
              <button
                class="preset-btn"
                class:active={selectedPreset === key}
                on:click={() => selectPreset(key)}
              >
                {preset.description}
              </button>
            {/each}
          {/if}
        </div>
      </div>

      <!-- Prompt -->
      <div class="control-group">
        <label>Prompt (editable)</label>
        <input
          type="text"
          bind:value={prompt}
          placeholder="Enter prompt for transformation..."
        />
      </div>

      <!-- Process Button -->
      <button
        class="process-btn"
        on:click={startProcessing}
        disabled={isProcessing || !inputVideoUrl}
      >
        {#if isProcessing}
          Processing...
        {:else}
          Process Video
        {/if}
      </button>
    </div>

    <!-- Job History -->
    {#if jobHistory.length > 0}
      <div class="history">
        <h3>Processing History</h3>
        <div class="job-list">
          {#each jobHistory as job}
            <div
              class="job-item"
              class:completed={job.status === 'completed'}
              class:failed={job.status === 'failed'}
              class:processing={job.status === 'processing'}
              on:click={() => loadJob(job)}
              on:keydown={(e) => e.key === 'Enter' && loadJob(job)}
              role="button"
              tabindex="0"
            >
              <div class="job-info">
                <span class="job-name">{job.input_path}</span>
                <span class="job-prompt">{job.prompt.substring(0, 50)}{job.prompt.length > 50 ? '...' : ''}</span>
              </div>
              <div class="job-status">
                {#if job.status === 'processing'}
                  <span class="status processing">{job.progress.toFixed(0)}%</span>
                {:else if job.status === 'completed'}
                  <span class="status completed">Done</span>
                {:else if job.status === 'failed'}
                  <span class="status failed">Failed</span>
                {:else}
                  <span class="status pending">Pending</span>
                {/if}
                {#if job.status === 'completed' || job.status === 'failed'}
                  <button class="delete-btn" on:click={(e) => deleteJob(job, e)}>x</button>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    background: #0a0a12;
    color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
  }

  header {
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 1px solid #333;
  }

  h1 {
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
  }

  .subtitle {
    color: #888;
    margin: 8px 0 0 0;
  }

  .container {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .video-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .video-box {
    background: #16213e;
    border-radius: 12px;
    padding: 16px;
  }

  .video-box h3 {
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0 0 12px 0;
    color: #888;
  }

  .video-box video {
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    border-radius: 8px;
    background: #0f0f1a;
  }

  .placeholder {
    width: 100%;
    aspect-ratio: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #0f0f1a;
    border-radius: 8px;
    color: #666;
    gap: 12px;
  }

  .placeholder.processing {
    color: #60a5fa;
  }

  .progress-container {
    width: 80%;
    height: 8px;
    background: #1f2937;
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    transition: width 0.3s ease;
  }

  .progress-percent {
    font-size: 1.2rem;
    font-weight: 600;
  }

  .preview-frame {
    width: 100%;
    height: 100%;
    object-fit: contain;
    position: absolute;
    top: 0;
    left: 0;
    border-radius: 8px;
  }

  .progress-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0, 0, 0, 0.85));
    padding: 20px;
    border-radius: 0 0 8px 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .progress-stats {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #ddd;
  }

  .eta {
    color: #60a5fa;
    font-weight: 500;
  }

  .frame-label {
    position: absolute;
    bottom: 10px;
    left: 10px;
    background: rgba(0, 0, 0, 0.7);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #60a5fa;
  }

  .placeholder.processing {
    position: relative;
    padding: 0;
  }

  .preset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
  }

  .preset-btn {
    background: #1f2937;
    border: 1px solid #374151;
    color: #ccc;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.8rem;
    transition: all 0.2s;
    text-align: center;
  }

  .preset-btn:hover {
    background: #374151;
    color: #fff;
  }

  .preset-btn.active {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-color: #60a5fa;
    color: #fff;
    font-weight: 500;
  }

  /* Real-time Comparison Section */
  .realtime-comparison {
    background: linear-gradient(135deg, #1e3a5f, #16213e);
    border: 2px solid #3b82f6;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
  }

  .realtime-comparison h3 {
    text-align: center;
    color: #60a5fa;
    margin: 0 0 16px 0;
    font-size: 1.1rem;
  }

  .comparison-grid {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
  }

  .comparison-frame {
    flex: 1;
    max-width: 400px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .comparison-frame img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    border-radius: 8px;
    background: #0f0f1a;
  }

  .frame-title {
    font-size: 0.85rem;
    color: #888;
    text-align: center;
  }

  .comparison-arrow {
    font-size: 2rem;
    color: #60a5fa;
    padding: 0 10px;
  }

  .comparison-stats {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
    color: #ccc;
    font-size: 0.9rem;
  }

  .separator {
    color: #444;
  }

  .progress-text {
    color: #60a5fa;
    font-weight: 600;
  }

  .progress-container.large {
    margin-top: 12px;
    height: 10px;
  }

  /* Processing indicator */
  .processing-placeholder {
    background: #0f0f1a;
  }

  .processing-indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: #60a5fa;
  }

  .processing-detail {
    font-size: 0.8rem;
    color: #666;
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #1f2937;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .playback-controls {
    display: flex;
    gap: 12px;
    justify-content: center;
    align-items: center;
    padding: 12px;
    background: #16213e;
    border-radius: 12px;
  }

  .playback-controls button {
    background: #3b82f6;
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .playback-controls button:hover {
    background: #2563eb;
  }

  .sync-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #888;
    cursor: pointer;
    margin-left: 20px;
  }

  .sync-toggle input {
    width: 18px;
    height: 18px;
  }

  .controls {
    background: #16213e;
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .control-row {
    display: flex;
    gap: 20px;
  }

  .control-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
  }

  .source-group {
    flex: 2;
  }

  .control-group label {
    font-size: 0.85rem;
    color: #888;
    font-weight: 500;
  }

  .source-options {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }

  .selected-file {
    font-size: 0.8rem;
    color: #60a5fa;
  }

  button, select, .file-button {
    background: #1f2937;
    border: 1px solid #374151;
    color: #eee;
    padding: 10px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s;
  }

  button:hover:not(:disabled), select:hover, .file-button:hover {
    background: #374151;
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .file-button {
    display: inline-block;
  }

  .file-button input {
    display: none;
  }

  input[type="text"] {
    background: #1f2937;
    border: 1px solid #374151;
    color: #eee;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.95rem;
    width: 100%;
    box-sizing: border-box;
  }

  input[type="text"]:focus {
    outline: none;
    border-color: #3b82f6;
  }

  .process-btn {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border: none;
    padding: 14px 28px;
    font-size: 1rem;
    font-weight: 600;
    align-self: center;
  }

  .process-btn:hover:not(:disabled) {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
  }

  .history {
    background: #16213e;
    border-radius: 12px;
    padding: 16px;
  }

  .history h3 {
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0 0 12px 0;
    color: #888;
  }

  .job-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 200px;
    overflow-y: auto;
  }

  .job-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: #1f2937;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
  }

  .job-item:hover {
    background: #374151;
  }

  .job-item.completed {
    border-left: 3px solid #4ade80;
  }

  .job-item.failed {
    border-left: 3px solid #f87171;
  }

  .job-item.processing {
    border-left: 3px solid #60a5fa;
  }

  .job-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .job-name {
    font-size: 0.9rem;
  }

  .job-prompt {
    font-size: 0.75rem;
    color: #888;
  }

  .job-status {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status {
    font-size: 0.8rem;
    padding: 4px 8px;
    border-radius: 4px;
  }

  .status.completed {
    background: #166534;
    color: #4ade80;
  }

  .status.failed {
    background: #7f1d1d;
    color: #f87171;
  }

  .status.processing {
    background: #1e3a5f;
    color: #60a5fa;
  }

  .status.pending {
    background: #374151;
    color: #9ca3af;
  }

  .delete-btn {
    background: transparent;
    border: none;
    color: #888;
    padding: 4px 8px;
    font-size: 1rem;
    cursor: pointer;
  }

  .delete-btn:hover {
    color: #f87171;
    background: transparent;
  }

  @media (max-width: 900px) {
    .video-grid {
      grid-template-columns: 1fr;
    }

    .control-row {
      flex-direction: column;
    }
  }
</style>
