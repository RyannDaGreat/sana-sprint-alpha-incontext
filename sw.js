// The site's one service worker (manifest §12 "PWA", m00836): the journal and the app keep working when closed and
// reopened, also offline after one online visit. publish_pages.build_site writes this file to the site root with
// VERSION (hash of the precached files) and PRECACHE (the app shell) filled in, so every publish rolls out a new cache.
// Strategies: app-shell HTML network-first; hashed app assets cache-first; other journal pages stale-while-revalidate;
// journal images (raw.githubusercontent.com, original lossless PNGs, never re-encoded) cached as viewed with an LRU
// byte cap. Model weights (Hugging Face) are not touched: the app keeps them in OPFS.
// Cross-origin isolation: GitHub Pages cannot send COOP/COEP headers, so this worker adds them to the app's pages
// (COEP credentialless: cross-origin no-cors loads still work, without cookies), which makes crossOriginIsolated true
// and lets ONNX Runtime's wasm use threads (SharedArrayBuffer).
const VERSION = "89b8073e50462638";
const PRECACHE = ["", "index.html", "pwa/manifest.webmanifest", "pwa/register.js", "eval_grid.js", "eval_labels.js", "eval_grid.css", "webgpu/", "webgpu/assets/index-BbGOAu18.js", "webgpu/assets/index-wKsOFb5M.css", "webgpu/assets/inter-cyrillic-ext-wght-normal-BOeWTOD4.woff2", "webgpu/assets/inter-cyrillic-wght-normal-DqGufNeO.woff2", "webgpu/assets/inter-greek-ext-wght-normal-DlzME5K_.woff2", "webgpu/assets/inter-greek-wght-normal-CkhJZR-_.woff2", "webgpu/assets/inter-latin-ext-wght-normal-DO1Apj_S.woff2", "webgpu/assets/inter-latin-wght-normal-Dx4kXJAl.woff2", "webgpu/assets/inter-vietnamese-wght-normal-CBcvBZtf.woff2", "webgpu/assets/jost-cyrillic-wght-normal-CjwCQKo9.woff2", "webgpu/assets/jost-latin-ext-wght-normal-BDUtSsKd.woff2", "webgpu/assets/jost-latin-wght-normal-ObQm3Zd1.woff2", "webgpu/assets/ort-wasm-simd-threaded.asyncify-AkdHdxrx.wasm", "webgpu/assets/pipeline-BT2hJmQm.js", "webgpu/assets/pipeline-DYwTyzQi.js", "webgpu/assets/segment-i664Uxl9.js", "webgpu/assets/splitter-CYjV9PgM.js", "webgpu/index.html", "webgpu/ort-memory64/ort-wasm-simd-threaded.asyncify.mjs", "webgpu/ort-memory64/ort-wasm-simd-threaded.asyncify.wasm"];  // paths relative to this file
const SHELL = `shell-${VERSION}`;
const PAGES = 'pages';
const IMAGES = 'images';
const IMAGE_CAP_BYTES = 1.5e9;  // journal PNGs are 1–3 MB each: about 600 images kept, oldest-used dropped first
const TOUCH_AFTER_MS = 3600e3;  // refresh an image's LRU time at most once an hour (each refresh rewrites the entry)
const ISOLATED_PATHS = ['webgpu/'];  // pages that get COOP/COEP (threads for ONNX Runtime wasm)
const IMAGE_HOST = 'raw.githubusercontent.com';
const ROOT = new URL('./', self.location).href;

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(SHELL).then((c) => c.addAll(PRECACHE.map((p) => new URL(p, ROOT).href))));
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key.startsWith('shell-') && key !== SHELL) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener('message', (event) => { if (event.data === 'SKIP_WAITING') self.skipWaiting(); });

/**
 * Pure function. Whether a same-origin site path gets COOP/COEP.
 *
 * @param {string} path - path relative to the site root, e.g. "webgpu/" or "webgpu/index.html"
 * @returns {boolean}
 *
 * @example isolatedPath("webgpu/index.html") // true
 * @example isolatedPath("compare/scrub.html") // false
 */
function isolatedPath(path) { return ISOLATED_PATHS.some((p) => path.startsWith(p)); }

/**
 * Pure function. A copy of response with the cross-origin isolation headers added.
 *
 * @param {Response} response
 * @returns {Response}
 */
function withIsolation(response) {
  if (response.status === 0) return response;  // opaque: headers can't be set
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

/** Command. Network first (and refresh the cache), the cache when offline. */
async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  } catch (error) {
    const hit = (await cache.match(request, { ignoreSearch: true })) || (await caches.match(request, { ignoreSearch: true }));
    if (hit) return hit;
    throw error;
  }
}

/** Command. The cached copy at once, refreshed in the background; the network when nothing is cached. */
async function staleWhileRevalidate(event, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(event.request, { ignoreSearch: true });
  const refresh = fetch(event.request).then(async (response) => {
    if (response.ok) await cache.put(event.request, response.clone());
    return response;
  });
  if (hit) { event.waitUntil(refresh.catch(() => {})); return hit; }  // offline refresh failures are expected
  return refresh;
}

/** Command. Cache first (precached hashed assets), the network otherwise. */
async function cacheFirst(request) {
  return (await caches.match(request, { ignoreSearch: true })) || networkFirst(request, PAGES);
}

/**
 * Pure function. Which entries to evict to bring the total under a byte cap, oldest-used first.
 *
 * @param {{key: string, bytes: number, time: number}[]} entries
 * @param {number} cap - bytes
 * @returns {string[]} keys to delete
 *
 * @example evictions([{key: 'a', bytes: 5, time: 1}, {key: 'b', bytes: 5, time: 2}], 6) // ['a']
 * @example evictions([{key: 'a', bytes: 5, time: 1}], 6) // []
 */
function evictions(entries, cap) {
  let total = entries.reduce((s, e) => s + e.bytes, 0);
  const out = [];
  for (const e of [...entries].sort((a, b) => a.time - b.time)) {
    if (total <= cap) break;
    out.push(e.key); total -= e.bytes;
  }
  return out;
}

/** Command. Stores a CORS response with its size and use time in headers (the LRU bookkeeping). */
async function putImage(cache, url, response) {
  const body = await response.blob();
  const headers = new Headers(response.headers);
  headers.set('x-sw-bytes', String(body.size));
  headers.set('x-sw-time', String(Date.now()));
  await cache.put(url, new Response(body, { status: 200, headers }));
}

/** Command. Drops least-recently-used images until the cache is under IMAGE_CAP_BYTES. */
async function trimImages(cache) {
  const entries = [];
  for (const request of await cache.keys()) {
    const r = await cache.match(request);
    entries.push({ key: request.url, bytes: Number(r.headers.get('x-sw-bytes')), time: Number(r.headers.get('x-sw-time')) });
  }
  for (const key of evictions(entries, IMAGE_CAP_BYTES)) await cache.delete(key);
}

/** Command. A journal image: the cached original bytes, or fetched with CORS (so its size is known) and cached. */
async function journalImage(event) {
  const url = event.request.url;
  const cache = await caches.open(IMAGES);
  const hit = await cache.match(url);
  if (hit) {
    if (Date.now() - Number(hit.headers.get('x-sw-time')) > TOUCH_AFTER_MS) event.waitUntil(putImage(cache, url, hit.clone()));
    return hit;
  }
  const response = await fetch(new Request(url, { mode: 'cors', credentials: 'omit' }));
  if (response.ok) event.waitUntil(putImage(cache, url, response.clone()).then(() => trimImages(cache)));
  return response;
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.hostname === IMAGE_HOST) { event.respondWith(journalImage(event)); return; }
  if (!url.href.startsWith(ROOT)) return;  // Hugging Face model files and other hosts: untouched (OPFS holds the weights)
  const path = url.href.slice(ROOT.length);
  if (path.startsWith('models/')) return;  // the models index placeholder: always the network
  if (request.mode === 'navigate') {
    const page = path === '' || path === 'index.html' || isolatedPath(path) ? networkFirst(request, PAGES) : staleWhileRevalidate(event, PAGES);
    event.respondWith(isolatedPath(path) ? page.then(withIsolation) : page);
    return;
  }
  if (PRECACHE.includes(path)) { event.respondWith(cacheFirst(request)); return; }
  event.respondWith(staleWhileRevalidate(event, PAGES));  // masks, journal scripts, report data
});
