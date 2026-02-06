<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  interface StyleInfo {
    slug: string;
    description: string;
  }

  interface MultiStyleJob {
    job_id: string;
    input_path: string;
    status: string;
    description: string;
    current_style_index: number;
    current_style_name: string;
    style_progress: number;
    overall_progress: number;
    completed_outputs: string[];
    grid_output: string | null;
    error: string | null;
    total_styles: number;
  }

  interface ServerVideo {
    name: string;
    duration: number;
    fps: number;
  }

  let styles: StyleInfo[] = [];
  let serverVideos: ServerVideo[] = [];
  let selectedVideo = '';
  let uploadedFile = '';
  let customDescription = '';
  let useCustomDescription = false;

  let currentJob: MultiStyleJob | null = null;
  let isProcessing = false;
  let error = '';
  let pollInterval: number | null = null;

  onMount(async () => {
    await loadStyles();
    await loadServerVideos();
  });

  onDestroy(() => {
    if (pollInterval) {
      clearInterval(pollInterval);
    }
  });

  async function loadStyles() {
    try {
      const response = await fetch('/api/multistyle/styles');
      if (response.ok) {
        const data = await response.json();
        styles = data.styles || [];
      }
    } catch (e) {
      console.error('Failed to load styles:', e);
    }
  }

  async function loadServerVideos() {
    try {
      const response = await fetch('/api/videos');
      if (response.ok) {
        const data = await response.json();
        serverVideos = data.videos || [];
      }
    } catch (e) {
      console.error('Failed to load videos:', e);
    }
  }

  async function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files || !input.files[0]) return;

    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        uploadedFile = data.filename;
        selectedVideo = '';
      } else {
        error = 'Failed to upload video';
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Upload failed';
    }
  }

  async function startProcessing() {
    if (!selectedVideo && !uploadedFile) {
      error = 'Please select or upload a video';
      return;
    }

    error = '';
    isProcessing = true;

    const formData = new FormData();
    if (uploadedFile) {
      formData.append('uploaded_file', uploadedFile);
    } else {
      formData.append('video_name', selectedVideo);
    }

    if (useCustomDescription && customDescription.trim()) {
      formData.append('custom_description', customDescription.trim());
    }

    try {
      const response = await fetch('/api/multistyle/process', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        currentJob = await response.json();
        startPolling();
      } else {
        const data = await response.json();
        error = data.detail || 'Failed to start processing';
        isProcessing = false;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Processing failed';
      isProcessing = false;
    }
  }

  function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      if (!currentJob) return;

      try {
        const response = await fetch(`/api/multistyle/job/${currentJob.job_id}`);
        if (response.ok) {
          currentJob = await response.json();

          if (currentJob.status === 'completed' || currentJob.status === 'failed') {
            if (pollInterval) {
              clearInterval(pollInterval);
              pollInterval = null;
            }
            isProcessing = false;

            if (currentJob.status === 'failed') {
              error = currentJob.error || 'Processing failed';
            }
          }
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }, 1000);
  }

  function getOutputUrl(jobId: string, outputPath: string): string {
    const filename = outputPath.split('/').pop() || outputPath;
    return `/api/multistyle/output/${jobId}/${encodeURIComponent(filename)}`;
  }

  function getStatusText(status: string): string {
    switch (status) {
      case 'pending': return 'Pending...';
      case 'analyzing': return 'Analyzing with LLaVA...';
      case 'generating': return 'Generating styles with FLUX...';
      case 'creating_grid': return 'Creating comparison grid...';
      case 'completed': return 'Complete!';
      case 'failed': return 'Failed';
      default: return status;
    }
  }

  function getStyleDisplayName(slug: string): string {
    return slug.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  }
</script>

<div class="multistyle-tab">
  <div class="tab-header">
    <h2>Multi-Style Generator</h2>
    <p class="tab-description">
      Analyze your video with LLaVA AI, then generate 5 artistic style variations using FLUX.
    </p>
  </div>

  <div class="content-grid">
    <!-- Left: Input Section -->
    <div class="input-section">
      <h3>1. Select Video</h3>

      <div class="input-options">
        <div class="option-group">
          <label>Upload a video:</label>
          <input
            type="file"
            accept="video/*"
            on:change={handleFileUpload}
            disabled={isProcessing}
          />
          {#if uploadedFile}
            <div class="selected-file">Selected: {uploadedFile}</div>
          {/if}
        </div>

        <div class="divider">or</div>

        <div class="option-group">
          <label>Choose from server:</label>
          <select bind:value={selectedVideo} disabled={isProcessing} on:change={() => uploadedFile = ''}>
            <option value="">-- Select a video --</option>
            {#each serverVideos as video}
              <option value={video.name}>{video.name} ({video.duration}s)</option>
            {/each}
          </select>
        </div>
      </div>

      <div class="description-section">
        <label class="checkbox-label">
          <input type="checkbox" bind:checked={useCustomDescription} disabled={isProcessing} />
          Use custom description (skip LLaVA analysis)
        </label>

        {#if useCustomDescription}
          <textarea
            bind:value={customDescription}
            placeholder="Describe what's happening in the video..."
            disabled={isProcessing}
            rows="3"
          ></textarea>
        {/if}
      </div>

      <button
        class="start-btn"
        on:click={startProcessing}
        disabled={isProcessing || (!selectedVideo && !uploadedFile)}
      >
        {isProcessing ? 'Processing...' : 'Generate Styles'}
      </button>

      {#if error}
        <div class="error-message">{error}</div>
      {/if}
    </div>

    <!-- Right: Info Section -->
    <div class="info-section">
      <h3>Styles Generated</h3>
      <ul class="styles-list">
        {#each styles as style, i}
          <li class:active={currentJob && currentJob.current_style_index === i && currentJob.status === 'generating'}>
            <span class="style-num">{i + 1}</span>
            <span class="style-name">{getStyleDisplayName(style.slug)}</span>
            {#if currentJob && currentJob.completed_outputs.length > i}
              <span class="check">✓</span>
            {/if}
          </li>
        {/each}
      </ul>
    </div>
  </div>

  <!-- Progress Section -->
  {#if currentJob}
    <div class="progress-section">
      <h3>Progress</h3>

      <div class="status-row">
        <span class="status-label">Status:</span>
        <span class="status-value {currentJob.status}">{getStatusText(currentJob.status)}</span>
      </div>

      {#if currentJob.description}
        <div class="description-row">
          <span class="desc-label">LLaVA Description:</span>
          <span class="desc-value">"{currentJob.description}"</span>
        </div>
      {/if}

      {#if currentJob.status === 'generating'}
        <div class="current-style">
          Processing: <strong>{getStyleDisplayName(currentJob.current_style_name)}</strong>
          ({currentJob.current_style_index + 1}/{currentJob.total_styles})
        </div>
      {/if}

      <div class="progress-bar-container">
        <div class="progress-bar" style="width: {currentJob.overall_progress}%"></div>
      </div>
      <div class="progress-text">{currentJob.overall_progress.toFixed(1)}%</div>
    </div>
  {/if}

  <!-- Results Section -->
  {#if currentJob && currentJob.completed_outputs.length > 0}
    <div class="results-section">
      <h3>Generated Videos</h3>

      <div class="results-grid">
        {#each currentJob.completed_outputs as output, i}
          <div class="result-card">
            <video
              src={getOutputUrl(currentJob.job_id, output)}
              controls
              preload="metadata"
            ></video>
            <div class="result-label">{getStyleDisplayName(styles[i]?.slug || '')}</div>
            <a
              href={getOutputUrl(currentJob.job_id, output)}
              download
              class="download-link"
            >Download</a>
          </div>
        {/each}
      </div>

      {#if currentJob.grid_output}
        <div class="grid-section">
          <h4>Comparison Grid</h4>
          <video
            src={getOutputUrl(currentJob.job_id, currentJob.grid_output)}
            controls
            class="grid-video"
          ></video>
          <a
            href={getOutputUrl(currentJob.job_id, currentJob.grid_output)}
            download
            class="download-link"
          >Download Grid</a>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .multistyle-tab {
    background: #16213e;
    border-radius: 12px;
    padding: 24px;
  }

  .tab-header {
    margin-bottom: 24px;
  }

  .tab-header h2 {
    margin: 0 0 8px 0;
    color: #eee;
    font-size: 1.5rem;
  }

  .tab-description {
    color: #8899aa;
    margin: 0;
  }

  .content-grid {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 24px;
    margin-bottom: 24px;
  }

  .input-section, .info-section {
    background: #0d1b3a;
    border-radius: 8px;
    padding: 20px;
  }

  .input-section h3, .info-section h3 {
    margin: 0 0 16px 0;
    color: #eee;
    font-size: 1.1rem;
  }

  .input-options {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .option-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .option-group label {
    color: #aabbcc;
    font-size: 0.9rem;
  }

  .option-group input[type="file"],
  .option-group select {
    background: #1a2a4a;
    border: 1px solid #2a4a8a;
    border-radius: 6px;
    padding: 10px;
    color: #eee;
  }

  .selected-file {
    color: #8f8;
    font-size: 0.85rem;
  }

  .divider {
    text-align: center;
    color: #667788;
    padding: 4px 0;
  }

  .description-section {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #2a4a8a;
  }

  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #aabbcc;
    cursor: pointer;
  }

  .description-section textarea {
    width: 100%;
    margin-top: 12px;
    background: #1a2a4a;
    border: 1px solid #2a4a8a;
    border-radius: 6px;
    padding: 10px;
    color: #eee;
    resize: vertical;
    font-family: inherit;
  }

  .start-btn {
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    background: linear-gradient(135deg, #4a7aea, #6a5acd);
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
  }

  .start-btn:hover:not(:disabled) {
    opacity: 0.9;
  }

  .start-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-message {
    margin-top: 12px;
    padding: 10px;
    background: rgba(255, 100, 100, 0.1);
    border: 1px solid #f66;
    border-radius: 6px;
    color: #f88;
    font-size: 0.9rem;
  }

  .styles-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .styles-list li {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 6px;
    background: #1a2a4a;
    transition: background 0.2s;
  }

  .styles-list li.active {
    background: #2a4a8a;
  }

  .style-num {
    width: 24px;
    height: 24px;
    background: #2a4a8a;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    color: #aabbcc;
  }

  .style-name {
    flex: 1;
    color: #ddd;
    font-size: 0.9rem;
  }

  .check {
    color: #8f8;
  }

  .progress-section {
    background: #0d1b3a;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
  }

  .progress-section h3 {
    margin: 0 0 16px 0;
    color: #eee;
    font-size: 1.1rem;
  }

  .status-row, .description-row {
    margin-bottom: 12px;
  }

  .status-label, .desc-label {
    color: #8899aa;
    margin-right: 8px;
  }

  .status-value {
    font-weight: 600;
  }

  .status-value.analyzing { color: #ffa; }
  .status-value.generating { color: #8af; }
  .status-value.creating_grid { color: #af8; }
  .status-value.completed { color: #8f8; }
  .status-value.failed { color: #f88; }

  .desc-value {
    color: #ddd;
    font-style: italic;
  }

  .current-style {
    color: #aabbcc;
    margin-bottom: 12px;
  }

  .progress-bar-container {
    height: 12px;
    background: #1a2a4a;
    border-radius: 6px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #4a7aea, #8f8);
    transition: width 0.3s ease;
  }

  .progress-text {
    text-align: center;
    color: #aabbcc;
    margin-top: 8px;
    font-size: 0.9rem;
  }

  .results-section {
    background: #0d1b3a;
    border-radius: 8px;
    padding: 20px;
  }

  .results-section h3, .results-section h4 {
    margin: 0 0 16px 0;
    color: #eee;
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .result-card {
    background: #16213e;
    border-radius: 8px;
    overflow: hidden;
  }

  .result-card video {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    background: #000;
  }

  .result-label {
    padding: 10px;
    color: #eee;
    font-weight: 500;
    text-align: center;
  }

  .download-link {
    display: block;
    padding: 10px;
    text-align: center;
    background: #2a5a3a;
    color: #8f8;
    text-decoration: none;
    transition: background 0.2s;
  }

  .download-link:hover {
    background: #3a7a4a;
  }

  .grid-section {
    border-top: 1px solid #2a4a8a;
    padding-top: 20px;
    text-align: center;
  }

  .grid-video {
    max-width: 100%;
    max-height: 500px;
    border-radius: 8px;
    margin-bottom: 12px;
  }

  @media (max-width: 768px) {
    .content-grid {
      grid-template-columns: 1fr;
    }

    .results-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
