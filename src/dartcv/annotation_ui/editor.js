/* Pure editor operations shared by the browser and dependency-free Node tests. */
(function (root) {
  "use strict";
  const clone = value => JSON.parse(JSON.stringify(value));
  const parts = ["flight", "shaft_barrel", "uncertain"];
  class History {
    constructor(document) { this.reset(document); }
    reset(document) { this.document = clone(document); this.past = []; this.future = []; }
    commit(document) {
      if (JSON.stringify(document) === JSON.stringify(this.document)) return;
      this.past.push(clone(this.document));
      if (this.past.length > 100) this.past.shift();
      this.document = clone(document);
      this.future = [];
    }
    undo() {
      if (!this.past.length) return;
      this.future.push(this.document);
      this.document = this.past.pop();
    }
    redo() {
      if (!this.future.length) return;
      this.past.push(this.document);
      this.document = this.future.pop();
    }
  }
  function regions(document, dartIndex) {
    const dart = document.darts[dartIndex];
    const result = [];
    if (dart) {
      dart.visible.forEach((region, index) => result.push({
        kind: "visible", index, part: region.part, points: region.points
      }));
      dart.silhouette.forEach((points, index) => result.push({
        kind: "silhouette", index, part: "silhouette", points
      }));
    }
    document.ignore.forEach((points, index) => result.push({
      kind: "ignore", index, part: "ignore", points
    }));
    return result;
  }
  function addPolygon(document, dartIndex, layer, points) {
    if (points.length < 3) throw new Error("A polygon needs at least three vertices.");
    if (layer === "ignore") document.ignore.push(clone(points));
    else {
      const dart = document.darts[dartIndex];
      if (!dart) throw new Error("Select or add a dart first.");
      if (layer === "silhouette") dart.silhouette.push(clone(points));
      else if (parts.includes(layer)) dart.visible.push({part: layer, points: clone(points)});
      else throw new Error("Unknown polygon layer.");
    }
  }
  function polygonPoints(document, dartIndex, region) {
    if (region.kind === "ignore") return document.ignore[region.index];
    const dart = document.darts[dartIndex];
    return region.kind === "visible" ? dart.visible[region.index].points : dart.silhouette[region.index];
  }
  function removePolygon(document, dartIndex, region) {
    if (region.kind === "ignore") document.ignore.splice(region.index, 1);
    else document.darts[dartIndex][region.kind === "visible" ? "visible" : "silhouette"].splice(region.index, 1);
  }
  function setTip(dart, point) {
    dart.tip_status = point === null ? "unknown" : "exact";
    dart.tip = point === null ? null : clone(point);
  }
  function setScore(dart, number, multiplier = "single") {
    if (number === null) { dart.score = null; return; }
    if (![0, ...Array.from({length: 20}, (_, i) => i + 1), 25].includes(number)) {
      throw new Error("Score number must be 0, 1-20, or 25.");
    }
    if (!["single", "double", "triple"].includes(multiplier) ||
        (number === 0 && multiplier !== "single") ||
        (number === 25 && multiplier === "triple")) throw new Error("Invalid scoring ring for this number.");
    dart.score = {number, multiplier};
  }
  function nearestEdge(points, point) {
    let result = {index: -1, distance: Infinity};
    points.forEach((a, index) => {
      const b = points[(index + 1) % points.length];
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const length = dx * dx + dy * dy;
      const t = length ? Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length)) : 0;
      const distance = Math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy);
      if (distance < result.distance) result = {index, distance};
    });
    return result;
  }
  function completeErrors(document) {
    return document.darts.flatMap(dart => dart.visible.length ? [] : [`${dart.id} needs a visible region (or remove this dart).`]);
  }
  const api = {clone, parts, History, regions, addPolygon, polygonPoints, removePolygon, setTip, setScore, nearestEdge, completeErrors};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DartEditor = api;
})(typeof window === "undefined" ? globalThis : window);
