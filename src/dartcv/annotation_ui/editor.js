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
  function setTip(dart, point, status = "exact") {
    if (!["exact", "approximate"].includes(status)) throw new Error("A placed tip must be exact or approximate.");
    dart.tip_status = point === null ? "unknown" : status;
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
  function addVertex(points, point, tolerance) {
    const edge = nearestEdge(points, point);
    if (edge.distance > tolerance) throw new Error("Click an edge of the selected polygon to add a vertex.");
    points.splice(edge.index + 1, 0, clone(point));
  }
  function removeVertex(points, index) {
    if (index < 0 || index >= points.length) throw new Error("Click a vertex of the selected polygon to remove it.");
    if (points.length <= 3) throw new Error("A polygon needs three vertices. Delete the polygon instead.");
    points.splice(index, 1);
  }
  function completeErrors(document) {
    return document.darts.flatMap(dart => dart.visible.length ? [] : [`${dart.id} needs a visible region (or remove this dart).`]);
  }
  function intersections(points) {
    const found = [];
    const epsilon = 1e-7;
    const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
    const subtract = (a, b) => [a[0] - b[0], a[1] - b[1]];
    const add = point => {
      if (!found.some(p => Math.hypot(p[0] - point[0], p[1] - point[1]) < epsilon)) found.push(point);
    };
    const onSegment = (p, a, b) => {
      const ab = subtract(b, a), ap = subtract(p, a);
      return Math.abs(cross(ab, ap)) <= epsilon * Math.max(1, Math.hypot(...ab)) &&
        p[0] >= Math.min(a[0], b[0]) - epsilon && p[0] <= Math.max(a[0], b[0]) + epsilon &&
        p[1] >= Math.min(a[1], b[1]) - epsilon && p[1] <= Math.max(a[1], b[1]) + epsilon;
    };
    for (let i = 0; i < points.length; i++) {
      const a = points[i], b = points[(i + 1) % points.length], r = subtract(b, a);
      if (Math.hypot(...r) < epsilon) add(a);
      for (let j = i + 1; j < points.length; j++) {
        const c = points[j], d = points[(j + 1) % points.length], s = subtract(d, c);
        const adjacent = j === i + 1 || (i === 0 && j === points.length - 1);
        const shared = j === i + 1 ? b : a;
        const report = point => {
          if (!adjacent || Math.hypot(point[0] - shared[0], point[1] - shared[1]) > epsilon) add(point);
        };
        const denominator = cross(r, s), offset = subtract(c, a);
        if (Math.abs(denominator) > epsilon) {
          const t = cross(offset, s) / denominator, u = cross(offset, r) / denominator;
          if (t >= -epsilon && t <= 1 + epsilon && u >= -epsilon && u <= 1 + epsilon) {
            report([a[0] + t * r[0], a[1] + t * r[1]]);
          }
        } else if (Math.abs(cross(offset, r)) <= epsilon) {
          for (const p of [a, b, c, d]) if (onSegment(p, a, b) && onSegment(p, c, d)) report(p);
        }
      }
    }
    return found;
  }
  const api = {clone, parts, History, regions, addPolygon, polygonPoints, removePolygon, setTip, setScore, nearestEdge, addVertex, removeVertex, completeErrors, intersections};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DartEditor = api;
})(typeof window === "undefined" ? globalThis : window);
