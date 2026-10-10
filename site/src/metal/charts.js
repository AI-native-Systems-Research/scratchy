// Draws every [data-chart] with Carbon Charts. The data and options are
// rendered by the site build; this only adds what JSON can't carry: a
// tooltip that also lists any point's note (e.g. "3 of 12 untimed").
(function () {
  document.querySelectorAll('[data-chart]').forEach(function (el) {
    var spec = JSON.parse(el.getAttribute('data-chart'));
    spec.options.tooltip = {
      customHTML: function (points, html) {
        var notes = points.filter(function (p) { return p.note; });
        if (!notes.length) return html;
        var box = document.createElement('div');
        box.className = 'mchartnote';
        notes.forEach(function (p) {
          var line = document.createElement('div');
          line.textContent = p.group + ': ' + p.note;
          box.appendChild(line);
        });
        return html + box.outerHTML;
      },
    };
    new Charts.LineChart(el, spec);
  });
})();
