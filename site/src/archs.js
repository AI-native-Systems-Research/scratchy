const DATA = JSON.parse(document.getElementById('archdata').textContent);
const A = DATA.archs, ORDER = DATA.order;
const $ = id => document.getElementById(id);

// The select is rendered with its initial (empty) selection, so the page is
// correct before the component modules finish loading; L and R are the
// authority afterwards. R === '' means "no diff — just show L".
let L = DATA.left, R = DATA.right;

// diff-view-element ships without Python highlighting; load the grammar into
// its own PrismJS instance once, the first time a diff is actually shown —
// not on page load, since most visits never open one. Must wait for the
// element to actually be upgraded first: its register <script type=module>
// loads asynchronously, so on a fresh page load (or a #left..right deep
// link) this can run before `.highlighter`/`.requestUpdate` exist yet.
let pythonLoaded = false;
async function ensurePythonHighlighting() {
  if (pythonLoaded) return;
  const [{ loader }] = await Promise.all([
    import(DATA.prismPythonUrl),
    customElements.whenDefined('diff-view-element'),
  ]);
  if (pythonLoaded) return;
  pythonLoaded = true;
  loader($('diffview').highlighter);
  $('diffview').requestUpdate();
}

function render() {
  // cds-tile's own shadow CSS sets `:host(cds-tile) { display: block }`
  // unconditionally, which an author-origin rule always beats the UA-only
  // `[hidden] { display: none }` default — so toggling `.hidden` on a
  // cds-tile does nothing. Inline `style.display` outranks any stylesheet
  // rule (short of !important), including that one.
  const diffing = R !== '' && R !== L;
  $('archpane').style.display = diffing ? 'none' : '';
  $('diffpane').style.display = diffing ? '' : 'none';

  if (!diffing) {
    $('code').innerHTML = A[L].html.join('\n');
    $('path').textContent = A[L].path;
    $('meta').textContent = A[L].lines + ' lines';
  } else {
    const diffview = $('diffview');
    diffview.oldValue = A[L].raw.join('\n');
    diffview.newValue = A[R].raw.join('\n');
    $('diffhead').textContent = `${A[L].path} → ${A[R].path}`;
    ensurePythonHighlighting();
  }

  for (const link of document.querySelectorAll('#model-nav cds-side-nav-link[data-arch]')) {
    link.toggleAttribute('active', link.dataset.arch === L);
  }

  history.replaceState(null, '', diffing ? `#${L}..${R}` : `#${L}`);
}

function pick(side, name) {
  if (side === 'left') {
    if (!A[name]) return;
    L = name;
  } else {
    if (name !== '' && !A[name]) return;
    R = name;
    $('right').value = name;
  }
  render();
}

$('right').addEventListener('cds-select-selected', e => pick('right', e.detail.value));

for (const link of document.querySelectorAll('#model-nav cds-side-nav-link[data-arch]')) {
  link.addEventListener('click', e => {
    e.preventDefault();
    pick('left', link.dataset.arch);
  });
}

// A #left..right fragment makes one specific comparison linkable (and #left
// alone links just that model), and stays live afterward so the side-nav's
// own generated hrefs (and back/forward) keep working.
function applyHash() {
  const [left, right] = decodeURIComponent(location.hash.slice(1)).split('..');
  if (A[left]) pick('left', left);
  pick('right', right && A[right] ? right : '');
  render();
}
window.addEventListener('hashchange', applyHash);
applyHash();
