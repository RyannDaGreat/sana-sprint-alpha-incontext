// Registers the site's service worker (manifest §12 "PWA", m00836) from any page: the site root is this script's
// parent folder's parent. Shows "Update available — reload" when a newer publish is waiting. On the app's pages, reloads
// once after the worker first takes control, so its COOP/COEP headers make the page cross-origin isolated.
(() => {
  if (!('serviceWorker' in navigator)) return;
  const root = new URL('../', document.currentScript.src);
  const RELOADED_KEY = 'sw-isolation-reload';
  const needsIsolation = location.href.startsWith(new URL('webgpu/', root).href);

  /** Command. Shows a small fixed bar offering to switch to the waiting worker. */
  function offerUpdate(worker) {
    const bar = document.createElement('button');
    bar.textContent = 'Update available — reload';
    bar.style.cssText = 'position:fixed;right:max(16px,env(safe-area-inset-right,0px));bottom:max(16px,env(safe-area-inset-bottom,0px));'
      + 'z-index:9999;padding:0.6em 1em;font:14px system-ui,sans-serif;border-radius:8px;border:1px solid currentColor;cursor:pointer;'
      + 'background:Canvas;color:CanvasText';
    bar.onclick = () => worker.postMessage('SKIP_WAITING');
    document.body.append(bar);
  }

  navigator.serviceWorker.register(new URL('sw.js', root), { scope: root.href }).then((reg) => {
    if (reg.waiting && navigator.serviceWorker.controller) offerUpdate(reg.waiting);
    reg.addEventListener('updatefound', () => {
      const worker = reg.installing;
      worker.addEventListener('statechange', () => {
        if (worker.state === 'installed' && navigator.serviceWorker.controller) offerUpdate(worker);
      });
    });
  }).catch((error) => console.error('service worker registration failed', error));

  let switching = false;
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (switching) return;
    switching = true;
    location.reload();  // the new worker controls the page now (first install or an accepted update)
  });

  if (needsIsolation && !self.crossOriginIsolated && navigator.serviceWorker.controller && !sessionStorage.getItem(RELOADED_KEY)) {
    sessionStorage.setItem(RELOADED_KEY, '1');  // once per tab: never loop if a browser refuses isolation
    location.reload();
  }
})();
