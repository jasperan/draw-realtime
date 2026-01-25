<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  // State
  let settings: any = null;
  let connected = false;
  let userId = crypto.randomUUID();
  let ws: WebSocket | null = null;
  let streamUrl = '';

  // Input state
  let sourceType: 'webcam' | 'upload' | 'server_video' = 'webcam';
  let selectedModel = '';
  let prompt = '';
  let serverVideos: any[] = [];
  let selectedVideo = '';

  // Media elements
  let videoEl: HTMLVideoElement;
  let canvasEl: HTMLCanvasElement;
  let mediaStream: MediaStream | null = null;
  let uploadedVideo: HTMLVideoElement | null = null;

  // Processing state
  let isProcessing = false;
  let fps = 0;
  let lastFrameTime = 0;
  let frameCount = 0;

  onMount(async () => {
    // Load settings
    const res = await fetch('/api/settings');
    settings = await res.json();
    selectedModel = settings.default_model;
    prompt = settings.default_prompt;

    // Load server videos
    const videosRes = await fetch('/api/videos');
    const videosData = await videosRes.json();
    serverVideos = videosData.videos || [];

    // Start
    await connect();
  });

  onDestroy(() => {
    disconnect();
  });

  async function connect() {
    if (connected) return;

    // Start WebSocket
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${wsProtocol}//${window.location.host}/api/ws/${userId}`);

    ws.onopen = () => {
      connected = true;
      streamUrl = `/api/stream/${userId}`;
      startProcessing();
    };

    ws.onclose = () => {
      connected = false;
      isProcessing = false;
    };

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (data.status === 'send_frame' && isProcessing) {
        await sendFrame();
      }
    };
  }

  function disconnect() {
    isProcessing = false;
    if (ws) {
      ws.close();
      ws = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    connected = false;
  }

  async function startProcessing() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Start appropriate source
    if (sourceType === 'webcam') {
      await startWebcam();
    } else if (sourceType === 'upload' && uploadedVideo) {
      uploadedVideo.play();
    }

    isProcessing = true;
    await sendFrame();
  }

  async function startWebcam() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 512, height: 512 }
      });
      if (videoEl) {
        videoEl.srcObject = mediaStream;
        await videoEl.play();
      }
    } catch (e) {
      console.error('Webcam error:', e);
    }
  }

  async function sendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !isProcessing) return;

    // Calculate FPS
    const now = performance.now();
    if (lastFrameTime > 0) {
      frameCount++;
      if (now - lastFrameTime > 1000) {
        fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
        frameCount = 0;
        lastFrameTime = now;
      }
    } else {
      lastFrameTime = now;
    }

    // Request next frame
    ws.send(JSON.stringify({ status: 'next_frame' }));

    // Send parameters
    ws.send(JSON.stringify({
      prompt,
      model: selectedModel,
      source_type: sourceType,
      video_name: sourceType === 'server_video' ? selectedVideo : null
    }));

    // For webcam/upload, capture and send frame
    if (sourceType !== 'server_video') {
      const frameBlob = await captureFrame();
      if (frameBlob) {
        ws.send(frameBlob);
      } else {
        ws.send(new Uint8Array(0));
      }
    }
  }

  async function captureFrame(): Promise<Blob | null> {
    if (!canvasEl) return null;

    const ctx = canvasEl.getContext('2d');
    if (!ctx) return null;

    let source: HTMLVideoElement | null = null;
    if (sourceType === 'webcam' && videoEl) {
      source = videoEl;
    } else if (sourceType === 'upload' && uploadedVideo) {
      source = uploadedVideo;
    }

    if (!source || source.readyState < 2) return null;

    // Draw to canvas
    ctx.drawImage(source, 0, 0, 512, 512);

    // Convert to blob
    return new Promise((resolve) => {
      canvasEl.toBlob((blob) => resolve(blob), 'image/jpeg', 0.8);
    });
  }

  function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    // Stop current processing
    isProcessing = false;

    // Create video element for uploaded file
    if (uploadedVideo) {
      uploadedVideo.pause();
      uploadedVideo.src = '';
    }

    uploadedVideo = document.createElement('video');
    uploadedVideo.src = URL.createObjectURL(file);
    uploadedVideo.loop = true;
    uploadedVideo.muted = true;
    uploadedVideo.playsInline = true;

    uploadedVideo.onloadeddata = () => {
      sourceType = 'upload';
      if (connected) {
        uploadedVideo?.play();
        isProcessing = true;
        sendFrame();
      }
    };
  }

  function selectSource(type: 'webcam' | 'upload' | 'server_video') {
    if (sourceType === type) return;

    isProcessing = false;

    // Stop webcam if switching away
    if (sourceType === 'webcam' && mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }

    // Stop uploaded video if switching away
    if (sourceType === 'upload' && uploadedVideo) {
      uploadedVideo.pause();
    }

    sourceType = type;

    if (connected) {
      startProcessing();
    }
  }

  function selectServerVideo(name: string) {
    selectedVideo = name;
    selectSource('server_video');
  }
</script>

<main>
  <header>
    <h1>StreamDiffusion Real-Time Demo</h1>
    <div class="status">
      {#if connected}
        <span class="connected">Connected</span>
        <span class="fps">{fps} FPS</span>
      {:else}
        <span class="disconnected">Disconnected</span>
      {/if}
    </div>
  </header>

  <div class="container">
    <div class="video-grid">
      <div class="video-box">
        <h3>Input</h3>
        <video bind:this={videoEl} autoplay playsinline muted></video>
        <canvas bind:this={canvasEl} width="512" height="512" style="display:none"></canvas>
      </div>
      <div class="video-box">
        <h3>Output</h3>
        {#if streamUrl}
          <img src={streamUrl} alt="AI Output" />
        {:else}
          <div class="placeholder">Waiting for connection...</div>
        {/if}
      </div>
    </div>

    <div class="controls">
      <div class="control-group">
        <label>Source</label>
        <div class="source-buttons">
          <button
            class:active={sourceType === 'webcam'}
            on:click={() => selectSource('webcam')}
          >
            Webcam
          </button>
          <label class="file-button" class:active={sourceType === 'upload'}>
            Upload MP4
            <input type="file" accept="video/*" on:change={handleFileUpload} />
          </label>
          {#if serverVideos.length > 0}
            <select
              value={selectedVideo}
              on:change={(e) => selectServerVideo(e.currentTarget.value)}
              class:active={sourceType === 'server_video'}
            >
              <option value="" disabled>Server Videos</option>
              {#each serverVideos as video}
                <option value={video.name}>{video.name} ({video.duration}s)</option>
              {/each}
            </select>
          {/if}
        </div>
      </div>

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

      <div class="control-group">
        <label>Prompt</label>
        <input
          type="text"
          bind:value={prompt}
          placeholder="Enter prompt..."
        />
      </div>
    </div>
  </div>
</main>

<style>
  main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding-bottom: 15px;
    border-bottom: 1px solid #333;
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
  }

  .status {
    display: flex;
    gap: 15px;
    align-items: center;
  }

  .connected {
    color: #4ade80;
  }

  .disconnected {
    color: #f87171;
  }

  .fps {
    color: #60a5fa;
    font-family: monospace;
  }

  .container {
    display: flex;
    flex-direction: column;
    gap: 30px;
  }

  .video-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .video-box {
    background: #16213e;
    border-radius: 12px;
    padding: 15px;
  }

  .video-box h3 {
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 10px;
    color: #888;
  }

  .video-box video,
  .video-box img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 8px;
    background: #0f0f1a;
  }

  .placeholder {
    width: 100%;
    aspect-ratio: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f0f1a;
    border-radius: 8px;
    color: #666;
  }

  .controls {
    background: #16213e;
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 15px;
  }

  .control-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .control-group label {
    font-size: 0.85rem;
    color: #888;
    font-weight: 500;
  }

  .source-buttons {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
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

  button:hover, select:hover, .file-button:hover {
    background: #374151;
  }

  button.active, select.active, .file-button.active {
    background: #3b82f6;
    border-color: #3b82f6;
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
  }

  input[type="text"]:focus {
    outline: none;
    border-color: #3b82f6;
  }

  @media (max-width: 768px) {
    .video-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
