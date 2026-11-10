/* Apply the saved theme before first paint, so nothing flashes. */
(function () {
  try {
    var saved = localStorage.getItem('fast_theme');
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.dataset.theme = saved;
    }
  } catch (e) { /* private mode */ }
})();
