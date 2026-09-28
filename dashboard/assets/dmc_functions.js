// Formatter functions used by Dash Mantine charts ("functions as props").
// Referenced from Python as e.g. valueFormatter={"function": "percent"}.
var dmcfuncs = (window.dashMantineFunctions = window.dashMantineFunctions || {});

// 0.034 -> "3.4%"
dmcfuncs.percent = function (value) {
  return value == null ? "" : (value * 100).toFixed(1) + "%";
};

// 12.345 -> "$12.3B"
dmcfuncs.billions = function (value) {
  return value == null ? "" : "$" + value.toFixed(1) + "B";
};

// 0.021 -> "+2.1 pp" (difference between two shares)
dmcfuncs.pctPoints = function (value) {
  return value == null ? "" : (value >= 0 ? "+" : "") + (value * 100).toFixed(1) + " pp";
};

// 0.5 -> "0.50"
dmcfuncs.decimal2 = function (value) {
  return value == null ? "" : value.toFixed(2);
};

// Bar colour: teal for positive, red for negative
dmcfuncs.signColor = function (value) {
  return value >= 0 ? "teal.6" : "red.6";
};

// Bar colour for drift: red above the 0.5 threshold, blue otherwise
dmcfuncs.driftColor = function (value) {
  return value > 0.5 ? "red.6" : "blue.6";
};
