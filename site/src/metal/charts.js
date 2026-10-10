// Draws every [data-chart] with Carbon Charts, from the data and options the
// site build rendered. A ratio heatmap stores log2(ratio), so a factor of two
// either way is the same distance from ×1; its tooltip shows the ratio
// itself, "×1.23", through Carbon's valueFormatter.
(function () {
  var charts = { line: Charts.LineChart, heatmap: Charts.HeatmapChart };
  document.querySelectorAll('[data-chart]').forEach(function (el) {
    var spec = JSON.parse(el.getAttribute('data-chart'));
    if (spec.ratio) {
      var total = spec.options.locale.translations.total;
      spec.options.tooltip = {
        valueFormatter: function (value, label) {
          return label === total ? '×' + Math.pow(2, value).toFixed(2) : value;
        },
      };
    }
    new charts[spec.kind](el, { data: spec.data, options: spec.options });
  });
})();
