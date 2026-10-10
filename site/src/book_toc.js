(function () {
  var main = document.querySelector('.docs-content zero-md');
  var toc = document.querySelector('.page-toc');
  main.addEventListener('zero-md-rendered', function () {
    var root = main.shadowRoot;
    var headings = root.querySelectorAll('.markdown-body h2, .markdown-body h3');
    toc.innerHTML = '';
    if (!headings.length) return;
    var list = document.createElement('ul');
    headings.forEach(function (h) {
      var li = document.createElement('li');
      li.className = h.tagName.toLowerCase();
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      a.addEventListener('click', function (e) {
        e.preventDefault();
        main.goto('#' + h.id);
      });
      li.appendChild(a);
      list.appendChild(li);
    });
    toc.appendChild(list);
  });
})();
