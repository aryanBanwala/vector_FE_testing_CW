# app.py

from flask import Flask, request, render_template_string
import os
from dotenv import load_dotenv
from embeddings.text_embed import get_text_embedding
from db.qdrant import search_similar_vectors

load_dotenv()  # QDRANT_URL, QDRANT_API_KEY

app = Flask(__name__)
COLLECTIONS = ["feeds_clips_10", "feeds_clips_100", "feeds_clips_1000"]
K_OPTIONS   = [5, 10, 15, 50, 100]

HTML = """
<!doctype html>
<title>Vector Search</title>
<style>
  body { font-family: sans-serif; padding: 20px; }
  form { margin-bottom: 20px; }
  .grid { display: flex; flex-wrap: wrap; gap: 12px; }
  .item {
    width: 200px;
    position: relative;
    background: #000;
  }
  .item video,
  .item canvas {
    display: block;
    width: 100%;
    height: auto;
    cursor: pointer;
  }
  .play-btn {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0, 0, 0, 0.5);
    border: none;
    color: white;
    font-size: 24px;
    padding: 8px 12px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.4;
    pointer-events: none;
  }
  .item small {
    display: block;
    margin-top: 4px;
    color: #fff;
    text-align: center;
  }
</style>

<h1>Vector Search</h1>
<form method="post">
  <label>Query:
    <input name="query" size="40" value="{{ request.form.get('query','') }}">
  </label>
  &nbsp;&nbsp;
  <label>Collection:
    <select name="collection">
      {% for col in collections %}
        <option value="{{ col }}" {% if col==selected %}selected{% endif %}>{{ col }}</option>
      {% endfor %}
    </select>
  </label>
  &nbsp;&nbsp;
  <label>Results (k):
    <select name="top_k">
      {% for k in k_options %}
        <option value="{{ k }}" {% if k==selected_k %}selected{% endif %}>{{ k }}</option>
      {% endfor %}
    </select>
  </label>
  <button type="submit">Search</button>
</form>

{% if results %}
  <h2>Top {{ selected_k }} Results from "{{ selected }}"</h2>
  <div class="grid">
    {% for r in results %}
      <div class="item">
        <video preload="metadata" data-id="{{ r.id }}">
          <source src="{{ r.file }}" type="video/mp4">
          Your browser doesn’t support HTML5 video.
        </video>
        <button class="play-btn">►</button>
        <small>{{ r.id }}<br>score: {{ r.score }}</small>
      </div>
    {% endfor %}
  </div>
{% endif %}

<script>
// Pause all videos except the one playing
function pauseOthers(active) {
  document.querySelectorAll('.item video').forEach(v => {
    if (v !== active) v.pause();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.item').forEach(item => {
    const video = item.querySelector('video');
    const btn   = item.querySelector('.play-btn');

    // load a bit to get first frame
    video.addEventListener('loadedmetadata', () => {
      video.currentTime = 0.1;
    });

    // when seeking done, draw thumbnail
    video.addEventListener('seeked', () => {
      if (video.dataset.thumbDone) return;
      const w = video.videoWidth, h = video.videoHeight;
      const canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      canvas.getContext('2d').drawImage(video, 0, 0, w, h);
      video.style.display = 'none';
      item.insertBefore(canvas, video);
      video.dataset.thumbDone = 'true';
    });

    // toggle play/pause & thumbnail removal
    function toggle() {
      if (video.paused) {
        pauseOthers(video);
        const canvas = item.querySelector('canvas');
        if (canvas) canvas.remove();
        video.style.display = 'block';
        video.play();
      } else {
        video.pause();
      }
    }

    // click anywhere in the item toggles
    item.addEventListener('click', e => {
      toggle();
    });

    // update the play button icon
    video.addEventListener('play',  () => { btn.textContent = '❚❚'; });
    video.addEventListener('pause', () => { btn.textContent = '►'; });

    // ensure thumbnail generation
    video.load();
  });
});
</script>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    results    = []
    selected   = COLLECTIONS[0]
    selected_k = K_OPTIONS[0]
    if request.method == "POST":
        q          = request.form["query"]
        selected   = request.form["collection"]
        selected_k = int(request.form["top_k"])
        vec        = get_text_embedding(q)
        hits       = search_similar_vectors(selected, vec, top_k=selected_k)
        for hit in hits:
            results.append({
                "id":    hit.id,
                "score": f"{hit.score:.4f}",
                "file":  hit.payload.get("fileurl","")
            })
    return render_template_string(
        HTML,
        collections=COLLECTIONS,
        k_options=K_OPTIONS,
        selected=selected,
        selected_k=selected_k,
        results=results,
        request=request
    )

if __name__ == "__main__":
    app.run(debug=True)