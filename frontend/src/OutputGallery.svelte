<script lang="ts">
  import { onMount } from 'svelte';

  interface OutputVideo {
    filename: string;
    size_bytes: number;
    size_human: string;
    duration: number;
    width: number;
    height: number;
    created_at: string;
    thumbnail: string;
  }

  let outputs: OutputVideo[] = [];
  let loading = true;
  let error = '';
  let sortBy: 'date' | 'name' | 'size' | 'duration' = 'date';
  let filterText = '';
  let selectedVideo: OutputVideo | null = null;
  let deleteConfirm: OutputVideo | null = null;

  onMount(() => {
    loadOutputs();
  });

  async function loadOutputs() {
    loading = true;
    error = '';
    try {
      const response = await fetch('/api/outputs');
      if (!response.ok) throw new Error('Failed to load outputs');
      const data = await response.json();
      outputs = data.outputs || [];
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load outputs';
    } finally {
      loading = false;
    }
  }

  async function deleteVideo(video: OutputVideo) {
    try {
      const response = await fetch(`/api/output/${encodeURIComponent(video.filename)}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete video');
      outputs = outputs.filter(o => o.filename !== video.filename);
      deleteConfirm = null;
      if (selectedVideo?.filename === video.filename) {
        selectedVideo = null;
      }
    } catch (e) {
      alert('Failed to delete video: ' + (e instanceof Error ? e.message : 'Unknown error'));
    }
  }

  function formatDate(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(0);
    return `${mins}:${secs.padStart(2, '0')}`;
  }

  $: sortedOutputs = [...outputs]
    .filter(o => filterText === '' || o.filename.toLowerCase().includes(filterText.toLowerCase()))
    .sort((a, b) => {
      switch (sortBy) {
        case 'name': return a.filename.localeCompare(b.filename);
        case 'size': return b.size_bytes - a.size_bytes;
        case 'duration': return b.duration - a.duration;
        default: return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
    });
</script>

<div class="output-gallery">
  <div class="gallery-header">
    <h3>Output Library</h3>
    <div class="gallery-controls">
      <input
        type="text"
        placeholder="Filter..."
        bind:value={filterText}
        class="filter-input"
      />
      <select bind:value={sortBy} class="sort-select">
        <option value="date">Newest First</option>
        <option value="name">Name</option>
        <option value="size">Size</option>
        <option value="duration">Duration</option>
      </select>
      <button class="refresh-btn" on:click={loadOutputs} disabled={loading}>
        {loading ? '...' : '↻'}
      </button>
    </div>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {:else if loading}
    <div class="loading">Loading outputs...</div>
  {:else if sortedOutputs.length === 0}
    <div class="empty-state">
      {filterText ? 'No videos match your filter' : 'No output videos yet'}
    </div>
  {:else}
    <div class="video-grid">
      {#each sortedOutputs as video}
        <div class="video-card">
          <div class="thumbnail-container" on:click={() => selectedVideo = video} on:keydown={(e) => e.key === 'Enter' && (selectedVideo = video)} role="button" tabindex="0">
            <img
              src={video.thumbnail}
              alt={video.filename}
              loading="lazy"
              on:error={(e) => e.currentTarget.style.display = 'none'}
            />
            <div class="play-overlay">▶</div>
            <div class="duration-badge">{formatDuration(video.duration)}</div>
          </div>
          <div class="video-info">
            <div class="video-name" title={video.filename}>
              {video.filename.length > 30 ? video.filename.slice(0, 27) + '...' : video.filename}
            </div>
            <div class="video-meta">
              <span>{video.size_human}</span>
              <span>{video.width}×{video.height}</span>
            </div>
            <div class="video-date">{formatDate(video.created_at)}</div>
          </div>
          <div class="video-actions">
            <a
              href={`/api/output/${encodeURIComponent(video.filename)}`}
              download={video.filename}
              class="action-btn download-btn"
              title="Download"
            >↓</a>
            <button
              class="action-btn delete-btn"
              title="Delete"
              on:click|stopPropagation={() => deleteConfirm = video}
            >×</button>
          </div>
        </div>
      {/each}
    </div>
  {/if}

  <div class="gallery-footer">
    {sortedOutputs.length} video{sortedOutputs.length !== 1 ? 's' : ''}
    {filterText && outputs.length !== sortedOutputs.length ? ` (${outputs.length} total)` : ''}
  </div>
</div>

<!-- Video Player Modal -->
{#if selectedVideo}
  <div class="modal-overlay" on:click={() => selectedVideo = null} on:keydown={(e) => e.key === 'Escape' && (selectedVideo = null)} role="dialog" tabindex="-1">
    <div class="modal-content" on:click|stopPropagation on:keydown|stopPropagation>
      <div class="modal-header">
        <span class="modal-title">{selectedVideo.filename}</span>
        <button class="modal-close" on:click={() => selectedVideo = null}>×</button>
      </div>
      <video
        src={`/api/output/${encodeURIComponent(selectedVideo.filename)}`}
        controls
        autoplay
        class="modal-video"
      >
        Your browser does not support video playback.
      </video>
      <div class="modal-info">
        <span>{selectedVideo.size_human}</span>
        <span>{selectedVideo.width}×{selectedVideo.height}</span>
        <span>{formatDuration(selectedVideo.duration)}</span>
        <span>{formatDate(selectedVideo.created_at)}</span>
      </div>
    </div>
  </div>
{/if}

<!-- Delete Confirmation Modal -->
{#if deleteConfirm}
  <div class="modal-overlay" on:click={() => deleteConfirm = null} on:keydown={(e) => e.key === 'Escape' && (deleteConfirm = null)} role="dialog" tabindex="-1">
    <div class="confirm-modal" on:click|stopPropagation on:keydown|stopPropagation>
      <div class="confirm-title">Delete Video?</div>
      <div class="confirm-message">
        Are you sure you want to delete<br/>
        <strong>{deleteConfirm.filename}</strong>?
      </div>
      <div class="confirm-actions">
        <button class="cancel-btn" on:click={() => deleteConfirm = null}>Cancel</button>
        <button class="delete-confirm-btn" on:click={() => { if (deleteConfirm) deleteVideo(deleteConfirm); }}>Delete</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .output-gallery {
    background: #16213e;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
  }

  .gallery-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
  }

  .gallery-header h3 {
    margin: 0;
    font-size: 1.2rem;
    color: #eee;
  }

  .gallery-controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .filter-input {
    background: #0d1b3a;
    border: 1px solid #2a4a8a;
    border-radius: 6px;
    padding: 6px 12px;
    color: #eee;
    width: 150px;
  }

  .filter-input:focus {
    outline: none;
    border-color: #4a7aea;
  }

  .sort-select {
    background: #0d1b3a;
    border: 1px solid #2a4a8a;
    border-radius: 6px;
    padding: 6px 12px;
    color: #eee;
    cursor: pointer;
  }

  .refresh-btn {
    background: #2a4a8a;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    color: #eee;
    cursor: pointer;
    font-size: 1.1rem;
  }

  .refresh-btn:hover:not(:disabled) {
    background: #3a5a9a;
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .video-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
  }

  .video-card {
    background: #0d1b3a;
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .video-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .thumbnail-container {
    position: relative;
    aspect-ratio: 16 / 9;
    background: #000;
    cursor: pointer;
    overflow: hidden;
  }

  .thumbnail-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .play-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 2.5rem;
    color: rgba(255, 255, 255, 0.8);
    opacity: 0;
    transition: opacity 0.2s;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
  }

  .thumbnail-container:hover .play-overlay {
    opacity: 1;
  }

  .duration-badge {
    position: absolute;
    bottom: 6px;
    right: 6px;
    background: rgba(0, 0, 0, 0.75);
    color: #fff;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
  }

  .video-info {
    padding: 10px 12px 6px;
  }

  .video-name {
    color: #eee;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .video-meta {
    display: flex;
    gap: 12px;
    color: #8899aa;
    font-size: 0.75rem;
  }

  .video-date {
    color: #667788;
    font-size: 0.7rem;
    margin-top: 4px;
  }

  .video-actions {
    display: flex;
    padding: 6px 12px 10px;
    gap: 8px;
  }

  .action-btn {
    flex: 1;
    padding: 6px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    text-align: center;
    text-decoration: none;
  }

  .download-btn {
    background: #2a5a3a;
    color: #8f8;
  }

  .download-btn:hover {
    background: #3a7a4a;
  }

  .delete-btn {
    background: #5a2a3a;
    color: #f88;
  }

  .delete-btn:hover {
    background: #7a3a4a;
  }

  .loading, .empty-state, .error-message {
    text-align: center;
    padding: 40px;
    color: #8899aa;
  }

  .error-message {
    color: #f88;
  }

  .gallery-footer {
    margin-top: 16px;
    text-align: center;
    color: #667788;
    font-size: 0.85rem;
  }

  /* Modal styles */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 20px;
  }

  .modal-content {
    background: #16213e;
    border-radius: 12px;
    max-width: 90vw;
    max-height: 90vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #2a4a8a;
  }

  .modal-title {
    color: #eee;
    font-size: 0.9rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: calc(100% - 40px);
  }

  .modal-close {
    background: none;
    border: none;
    color: #8899aa;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0 8px;
  }

  .modal-close:hover {
    color: #fff;
  }

  .modal-video {
    max-width: 100%;
    max-height: 70vh;
    background: #000;
  }

  .modal-info {
    display: flex;
    gap: 16px;
    padding: 12px 16px;
    color: #8899aa;
    font-size: 0.85rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .confirm-modal {
    background: #16213e;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    max-width: 400px;
  }

  .confirm-title {
    color: #eee;
    font-size: 1.2rem;
    margin-bottom: 12px;
  }

  .confirm-message {
    color: #8899aa;
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .confirm-message strong {
    color: #eee;
    word-break: break-all;
  }

  .confirm-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .cancel-btn, .delete-confirm-btn {
    padding: 10px 24px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .cancel-btn {
    background: #2a4a8a;
    color: #eee;
  }

  .cancel-btn:hover {
    background: #3a5a9a;
  }

  .delete-confirm-btn {
    background: #8a2a3a;
    color: #fff;
  }

  .delete-confirm-btn:hover {
    background: #aa3a4a;
  }

  @media (max-width: 600px) {
    .gallery-header {
      flex-direction: column;
      align-items: stretch;
    }

    .gallery-controls {
      justify-content: space-between;
    }

    .filter-input {
      flex: 1;
    }

    .video-grid {
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    }
  }
</style>
