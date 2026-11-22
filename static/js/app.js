/* InstaSafe – single shared script (replaces the 4 duplicated inline copies).
   Duplicate click handlers used to be registered in base.html AND index.html/profile.html/
   liked_posts.html/search.html, so every like/follow click fired twice and cancelled itself. */
(function () {
  'use strict';
  const BLOCKED = document.body.dataset.blocked === '1';
  const PAGE = document.body.dataset.page;

  const $ = (sel, root = document) => root.querySelector(sel);

  function toast(html, type = 'warning', ms = 7000) {
    const el = document.createElement('div');
    el.className = 'alert alert-' + type;
    el.innerHTML = html;
    const host = $('.main-container');
    if (host) host.insertBefore(el, host.firstChild); else document.body.appendChild(el);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    setTimeout(() => el.remove(), ms);
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
      credentials: 'same-origin',
    });
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (res.status === 401) { location.href = '/login'; }
    return { ok: res.ok, status: res.status, data };
  }

  /* ---- flash messages auto-hide ---- */
  setTimeout(() => document.querySelectorAll('.alert').forEach(a => a.remove()), 8000);

  /* ---- search ---- */
  const search = $('#searchInput');
  if (search) search.addEventListener('keydown', e => {
    if (e.key === 'Enter') location.href = '/search?q=' + encodeURIComponent(search.value.trim());
  });

  /* ---- avatar fallback (images that fail to load) ---- */
  document.addEventListener('error', e => {
    const img = e.target;
    if (img.tagName === 'IMG' && img.classList.contains('profile-pic-img') && !img.dataset.failed) {
      img.dataset.failed = '1';
      img.style.display = 'none';
      const d = document.createElement('div');
      d.className = 'default-profile';
      d.textContent = (img.alt || 'U').charAt(0).toUpperCase();
      img.parentNode.appendChild(d);
    }
  }, true);

  /* ---- upload modal ---- */
  window.showUploadModal = function () {
    if (BLOCKED) { toast('<strong>❌ Your account is restricted.</strong> You cannot upload new posts.', 'error'); return; }
    $('#uploadModal').style.display = 'flex';
  };
  window.hideUploadModal = function () { $('#uploadModal').style.display = 'none'; };
  const modal = $('#uploadModal');
  if (modal) modal.addEventListener('click', e => { if (e.target === modal) window.hideUploadModal(); });

  /* ---- image validation + caption counter ---- */
  const OK_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
  document.addEventListener('change', e => {
    const input = e.target;
    if (input.type === 'file' && input.name === 'image' && input.files[0]) {
      const f = input.files[0];
      if (!OK_TYPES.includes(f.type)) { alert('Please select a valid image (JPEG, PNG, GIF, BMP, WEBP).'); input.value = ''; }
      else if (f.size > 16 * 1024 * 1024) { alert('Image must be smaller than 16 MB.'); input.value = ''; }
    }
  });
  const caption = $('textarea[name="caption"]');
  const counter = $('#charCount');
  if (caption && counter) caption.addEventListener('input', () => {
    const n = caption.value.length;
    counter.textContent = n;
    counter.style.color = n > 450 ? 'var(--danger-color)' : n > 400 ? 'var(--warning-color)' : 'var(--text-secondary)';
  });

  /* ---- busy state on normal POST forms (upload analysis can take a few seconds) ---- */
  document.addEventListener('submit', e => {
    const form = e.target;
    if ((form.method || '').toLowerCase() !== 'post') return;
    const btn = form.querySelector('button[type="submit"]');
    if (!btn) return;
    btn.dataset.html = btn.innerHTML;
    btn.innerHTML = '<div class="loading"></div>' + (form.dataset.busyText ? ' ' + form.dataset.busyText : '');
    setTimeout(() => { btn.disabled = true; }, 0);   // disable AFTER the browser has built the form data
  });
  window.addEventListener('pageshow', () => {          // back button restores the page with a stuck button
    document.querySelectorAll('button[type="submit"][data-html]').forEach(b => { b.innerHTML = b.dataset.html; b.disabled = false; });
  });

  /* ---- comments ---- */
  document.addEventListener('submit', async e => {
    const form = e.target.closest('form.comment-form[data-post-id]');
    if (!form) return;
    e.preventDefault();
    const postId = form.dataset.postId;
    const input = form.querySelector('.comment-input');
    const btn = form.querySelector('.comment-btn');
    const text = input.value.trim();
    if (!text || btn.disabled) return;

    const original = btn.innerHTML;
    btn.innerHTML = '<div class="loading"></div>';
    btn.disabled = true;
    try {
      const { ok, data } = await postJSON('/comment/' + postId, { comment: text });
      if (ok && data.success) {
        const c = data.comment;
        const wrap = document.createElement('div');
        wrap.className = 'comment' + (c.bullying_detected ? ' bullying-detected' : '');
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:flex-start;gap:8px;';
        const a = document.createElement('strong'); a.style.fontSize = '14px'; a.textContent = c.author;
        const t = document.createElement('span'); t.style.cssText = 'flex:1;font-size:14px;'; t.textContent = c.text;   // textContent => no XSS
        row.append(a, t); wrap.appendChild(row);
        if (c.bullying_detected) {
          const w = document.createElement('div');
          w.style.cssText = 'color:var(--danger-color);font-size:11px;margin-top:4px;';
          w.textContent = '🛡️ Cyberbullying detected';
          wrap.appendChild(w);
        }
        if (c.emoji && c.emoji.total > 0) {                       // emoji analysis of the comment
          const eb = document.createElement('div'); eb.className = 'comment-emoji';
          if (c.emoji.flags && c.emoji.flags.length) eb.title = c.emoji.flags.join(' · ');
          const glyphs = document.createElement('span');
          glyphs.textContent = c.emoji.top.map(t => t.emoji + (t.count > 1 ? '×' + t.count : '')).join(' ');
          const tone = document.createElement('span'); tone.textContent = c.emoji.sentiment + ' emoji tone';
          eb.append(glyphs, tone);
          if (c.emoji.risk_level !== 'low') {
            const rk = document.createElement('span'); rk.className = 'ce-risk-' + c.emoji.risk_level;
            rk.textContent = (c.emoji.risk_level === 'high' ? 'High' : 'Medium') + ' emoji risk';
            eb.appendChild(rk);
          }
          wrap.appendChild(eb);
        }
        const ts = document.createElement('div');
        ts.style.cssText = 'font-size:11px;color:var(--text-secondary);margin-top:2px;'; ts.textContent = 'Just now';
        wrap.appendChild(ts);
        const list = document.getElementById('comments-' + postId);
        if (list) list.appendChild(wrap);
        input.value = '';

        if (c.bullying_detected) {
          toast('<strong>🚨 Cyberbullying detected!</strong><br>Reputation decreased by ' + c.reputation_loss.toFixed(2) +
                '.<br>Current score: ' + c.user_reputation.toFixed(1) + '/10' +
                (data.restricted ? '<br><strong>❌ Your account is now restricted.</strong>' : ''), 'warning');
        }
        if (data.restricted) setTimeout(() => location.reload(), 2500);   // re-render page in restricted mode
      } else {
        toast(data.error || 'Failed to post comment', 'error');
        if (data.restricted) setTimeout(() => location.reload(), 2000);
      }
    } catch (err) {
      console.error(err);
      toast('Network error. Please try again.', 'error');
    } finally {
      btn.innerHTML = original;
      btn.disabled = BLOCKED;
    }
  });

  /* ---- clicks: like / share / follow / focus-comment (ONE delegated handler) ---- */
  document.addEventListener('click', async e => {
    const like = e.target.closest('.like-btn');
    if (like && like.dataset.postId) {
      const id = like.dataset.postId;
      const { ok, data } = await postJSON('/like/' + id);
      if (ok) {
        like.classList.toggle('liked', data.liked);
        const c = document.getElementById('like-count-' + id);
        if (c) c.textContent = data.like_count + ' likes';
        if (PAGE === 'liked_posts' && !data.liked) { const card = like.closest('.post-card'); if (card) card.style.display = 'none'; }
      }
      return;
    }
    const share = e.target.closest('.share-btn');
    if (share && share.dataset.postId) {
      const id = share.dataset.postId;
      const { ok, data } = await postJSON('/share/' + id);
      if (ok) {
        share.classList.add('shared');
        setTimeout(() => share.classList.remove('shared'), 1000);
        const c = document.getElementById('share-count-' + id);
        if (c) c.textContent = data.share_count + ' shares';
      }
      return;
    }
    const follow = e.target.closest('.follow-btn');
    if (follow) {
      const { ok, data } = await postJSON('/follow/' + follow.dataset.userId);
      if (!ok) { toast(data.error || 'Something went wrong', 'error'); return; }
      if (follow.dataset.reload) { location.reload(); return; }
      if (data.following !== undefined) {
        follow.textContent = data.following ? 'Following' : 'Follow';
        follow.classList.toggle('btn-primary', !data.following);
        follow.classList.toggle('btn-outline', data.following);
      } else if (data.request_sent !== undefined) {
        follow.textContent = data.request_sent ? 'Requested' : 'Follow';
      }
      return;
    }
    const focus = e.target.closest('[data-focus-comment]');
    if (focus) { const i = document.getElementById('comment-input-' + focus.dataset.focusComment); if (i) i.focus(); }
  });
})();
