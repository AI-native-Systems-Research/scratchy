// "On this page": once zero-md has rendered the chapter, list its h2s, each
// with its h3s nested under it, in Carbon's list components.
(function () {
  var chapter = document.querySelector('zero-md');
  var toc = document.getElementById('toc');
  function item(h) {
    var li = document.createElement('cds-list-item');
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    a.addEventListener('click', function (e) {
      e.preventDefault();
      chapter.goto('#' + h.id);
    });
    li.appendChild(a);
    return li;
  }
  chapter.addEventListener('zero-md-rendered', function () {
    var headings = chapter.shadowRoot.querySelectorAll('.markdown-body h2, .markdown-body h3');
    toc.replaceChildren();
    var last = null;
    headings.forEach(function (h) {
      if (h.tagName === 'H3' && last) {
        var nested = last.querySelector('cds-unordered-list');
        if (!nested) {
          nested = document.createElement('cds-unordered-list');
          nested.setAttribute('nested', '');
          last.appendChild(nested);
        }
        nested.appendChild(item(h));
      } else {
        last = item(h);
        toc.appendChild(last);
      }
    });
  });
})();
