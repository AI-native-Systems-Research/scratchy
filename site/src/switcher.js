// Ties a content switcher or tabs to the address, and lays full-width line tabs
// out: a hash naming one of its panels selects it (on load, and when the hash
// changes), and choosing a panel puts its id in the address to link to.
// replaceState, not a new entry or a jump. Carbon keeps no address of its
// own. Runs right after its switcher,
// before Carbon has upgraded it: a `value` set then loses to the item rendered
// selected, so the hash waits until the switcher has first rendered.
(function (switcher) {
  var tag = switcher.localName;
  var item = tag === 'cds-tabs' ? 'cds-tab' : 'cds-content-switcher-item';
  function fromHash() {
    var id = decodeURIComponent(location.hash.slice(1));
    if (id && switcher.querySelector(item + '[value="' + CSS.escape(id) + '"]')) {
      switcher.value = id;
    }
  }
  customElements
    .whenDefined(tag)
    .then(function () { return switcher.updateComplete; })
    .then(function () {
      fromHash();
      window.addEventListener('hashchange', fromHash);
    });
  // Full-width line tabs: Carbon spreads only contained tabs across their row, by rules inside
  // its shadow DOM that no page CSS reaches. The same rules, given to a line tabs' shadow root,
  // and each tab's link stretched to its share.
  if (tag === 'cds-tabs' && switcher.hasAttribute('full-width') && switcher.getAttribute('type') !== 'contained') {
    var adopt = function (root, css) {
      var sheet = new CSSStyleSheet();
      sheet.replaceSync(css);
      root.adoptedStyleSheets = root.adoptedStyleSheets.concat(sheet);
    };
    customElements
      .whenDefined(tag)
      .then(function () { return switcher.updateComplete; })
      .then(function () {
        adopt(switcher.shadowRoot,
          ':host { display: flex; inline-size: 100% }' +
          '.cds--tabs-nav-content-container { overflow: visible; flex: 1 1 0% }' +
          '.cds--tab--list, .cds--tabs-nav, .cds--tabs-nav-content { inline-size: 100% }' +
          '.cds--tab--list { display: flex }' +
          '::slotted(cds-tab) { flex: 1 1 0%; min-inline-size: 0 }');
        switcher.querySelectorAll('cds-tab').forEach(function (t) {
          t.updateComplete.then(function () {
            adopt(t.shadowRoot, '.cds--tabs__nav-link { inline-size: 100%; max-inline-size: none }');
          });
        });
      });
  }
  switcher.addEventListener(tag + '-selected', function (e) {
    var id = e.detail && e.detail.item && e.detail.item.value;
    if (id) history.replaceState(null, '', '#' + id);
  });
})(document.currentScript.previousElementSibling);
