// Ties a content switcher to the address: a hash naming one of its panels
// selects it (on load, and when the hash changes), and choosing a panel puts
// its id in the address to link to. replaceState, not a new entry or a jump.
// Carbon keeps no address of its own. Runs right after its switcher, before
// Carbon has upgraded it: a `value` set then loses to the item rendered
// selected, so the hash waits until the switcher has first rendered.
(function (switcher) {
  function fromHash() {
    var id = decodeURIComponent(location.hash.slice(1));
    if (id && switcher.querySelector('cds-content-switcher-item[value="' + CSS.escape(id) + '"]')) {
      switcher.value = id;
    }
  }
  customElements
    .whenDefined('cds-content-switcher')
    .then(function () { return switcher.updateComplete; })
    .then(function () {
      fromHash();
      window.addEventListener('hashchange', fromHash);
    });
  switcher.addEventListener('cds-content-switcher-selected', function (e) {
    var id = e.detail && e.detail.item && e.detail.item.value;
    if (id) history.replaceState(null, '', '#' + id);
  });
})(document.currentScript.previousElementSibling);
