"use strict";
(() => {
  const E = window.DartEditor;
  const $ = id => document.getElementById(id);
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  const state = {
    history: null, saved: "", revision: null, token: null, photos: [], index: -1,
    dart: 0, region: -1, image: null, scale: 1, x: 0, y: 0,
    mode: "edit", pending: [], layer: null, pendingDart: null,
    drag: null, space: false, busy: false
  };
  const colors = {
    flight: "#53cfff", shaft_barrel: "#ffbd59", uncertain: "#7bf586",
    silhouette: "#f59bff", ignore: "#d5d5d5"
  };
  const doc = () => state.history.document;
  const dirty = () => state.history && (JSON.stringify(doc()) !== state.saved || state.pending.length > 0);
  const activeDart = () => doc().darts[state.dart];
  const regions = () => E.regions(doc(), state.dart);
  const selected = () => regions()[state.region];
  function message(text, error = false) {
    $("message").textContent = text;
    $("message").className = error ? "error" : "";
  }
  async function api(path, options = {}) {
    const response = await fetch(path, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
    return body;
  }
  function run(action) {
    return async event => {
      if (state.busy) return;
      try { await action(event); }
      catch (error) { message(error.message, true); }
    };
  }
  function mutate(change) {
    const next = E.clone(doc());
    change(next);
    state.history.commit(next);
    refresh();
  }
  function option(value, text) {
    const node = document.createElement("option");
    node.value = value; node.textContent = text;
    return node;
  }
  function refresh() {
    if (!state.history) return;
    $("dirty").textContent = dirty() ? "Unsaved changes" : "Saved / unchanged";
    $("status").value = doc().status;
    $("darts").replaceChildren(...doc().darts.map((dart, index) => option(index, dart.id)));
    state.dart = Math.min(state.dart, doc().darts.length - 1);
    $("darts").value = state.dart;
    const dart = activeDart();
    $("contact").disabled = !dart;
    $("contact").value = dart ? dart.contact : "unknown";
    const score = dart?.score;
    $("score-number").disabled = !dart;
    $("score-number").value = score ? String(score.number) : "";
    $("score-multiplier").disabled = !dart || !score || score.number === 0;
    $("score-multiplier").value = score?.multiplier || "single";
    $("score-multiplier").querySelector('[value="triple"]').disabled = score?.number === 25;
    $("score-points").textContent = score
      ? `Manual label: ${score.number * {single: 1, double: 2, triple: 3}[score.multiplier]} points`
      : "Score not labeled (optional).";
    $("tip-status").textContent = !dart ? "No dart selected." : (
      dart.tip_status === "exact" ? `Exact tip: ${dart.tip.map(n => n.toFixed(1)).join(", ")}` : "Tip: hidden / unknown"
    );
    $("regions").replaceChildren(...regions().map((region, index) =>
      option(index, `${region.kind === "visible" ? "Visible " : ""}${region.part} #${region.index + 1}`)));
    if (state.region >= regions().length) state.region = -1;
    $("regions").value = state.region;
    $("undo").disabled = !state.history.past.length && !state.pending.length;
    $("redo").disabled = !state.history.future.length || state.pending.length > 0;
    for (const id of ["tip", "unknown-tip", "remove-dart"]) $(id).disabled = !dart;
    $("copy").disabled = !selected() || selected().kind !== "visible";
    $("delete").disabled = !selected();
    $("finish").disabled = state.mode !== "draw";
    render();
  }
  function cancelMode() {
    state.mode = "edit"; state.pending = []; state.layer = null; state.pendingDart = null;
    refresh();
  }
  function requireNoDrawing() {
    if (state.pending.length) throw new Error("Finish (Enter) or cancel the current polygon first.");
  }
  function leaveAllowed() {
    return !dirty() || window.confirm("Discard unsaved changes on this photo?");
  }
  async function load(index, confirmLeave = true) {
    if (index < 0 || index >= state.photos.length) return;
    if (confirmLeave && !leaveAllowed()) { $("photos").value = state.index; return; }
    state.busy = true;
    const name = state.photos[index].name;
    message(`Loading ${name}...`);
    try {
      const result = await api(`/api/annotation?name=${encodeURIComponent(name)}`);
      const image = new Image();
      image.src = `/api/pixels?name=${encodeURIComponent(name)}`;
      await image.decode();
      if (image.naturalWidth !== result.annotation.width || image.naturalHeight !== result.annotation.height) {
        throw new Error("Displayed photograph dimensions do not match the annotation.");
      }
      state.index = index;
      state.history = new E.History(result.annotation);
      state.saved = JSON.stringify(result.annotation);
      state.revision = result.revision;
      state.image = image; state.dart = 0; state.region = -1;
      state.mode = "edit"; state.pending = []; state.drag = null;
      $("photos").value = index;
      fit(); refresh();
      message(`${name} — source ${doc().width} × ${doc().height}. Choose a dart and draw its visible parts.`);
    } catch (error) {
      $("photos").value = state.index;
      throw error;
    } finally { state.busy = false; }
  }
  function fit() {
    if (!state.image) return;
    state.scale = Math.min(canvas.clientWidth / doc().width, canvas.clientHeight / doc().height) * 0.95;
    state.x = (canvas.clientWidth - doc().width * state.scale) / 2;
    state.y = (canvas.clientHeight - doc().height * state.scale) / 2;
    render();
  }
  function cursor(event) {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }
  function source(point, clamp = false) {
    let x = (point[0] - state.x) / state.scale, y = (point[1] - state.y) / state.scale;
    if (clamp) {
      x = Math.max(0, Math.min(doc().width - 1, x));
      y = Math.max(0, Math.min(doc().height - 1, y));
    } else if (x < 0 || y < 0 || x > doc().width - 1 || y > doc().height - 1) {
      throw new Error("Click inside the photograph.");
    }
    return [x, y];
  }
  function zoom(factor, point = [canvas.clientWidth / 2, canvas.clientHeight / 2]) {
    const before = [(point[0] - state.x) / state.scale, (point[1] - state.y) / state.scale];
    const fitScale = Math.min(canvas.clientWidth / doc().width, canvas.clientHeight / doc().height);
    state.scale = Math.max(fitScale / 3, Math.min(20, state.scale * factor));
    state.x = point[0] - before[0] * state.scale;
    state.y = point[1] - before[1] * state.scale;
    render();
  }
  function path(points, close = true) {
    ctx.beginPath();
    points.forEach((point, i) => i ? ctx.lineTo(...point) : ctx.moveTo(...point));
    if (close) ctx.closePath();
  }
  function polygon(points, color, highlighted, inferred) {
    if (!points.length) return;
    path(points);
    ctx.fillStyle = color + (inferred ? "12" : "30");
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = (highlighted ? 3 : 1.5) / state.scale;
    ctx.setLineDash(inferred ? [8 / state.scale, 5 / state.scale] : []);
    ctx.stroke();
    ctx.setLineDash([]);
    if (highlighted) {
      for (const point of points) {
        ctx.beginPath(); ctx.arc(...point, 4 / state.scale, 0, 2 * Math.PI);
        ctx.fillStyle = color; ctx.fill();
      }
    }
  }
  function render() {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    if (!state.image) return;
    ctx.translate(state.x, state.y); ctx.scale(state.scale, state.scale);
    ctx.drawImage(state.image, 0, 0);
    doc().darts.forEach((dart, dartIndex) => {
      dart.visible.forEach((region, index) => polygon(
        region.points, colors[region.part],
        dartIndex === state.dart && selected()?.kind === "visible" && selected().index === index, false
      ));
      if ($("show-inferred").checked) dart.silhouette.forEach((points, index) => polygon(
        points, colors.silhouette,
        dartIndex === state.dart && selected()?.kind === "silhouette" && selected().index === index, true
      ));
      if (dart.tip) {
        const [x, y] = dart.tip, radius = 9 / state.scale;
        ctx.strokeStyle = dartIndex === state.dart ? "#ffff60" : "#ffffff";
        ctx.lineWidth = 2 / state.scale;
        ctx.beginPath(); ctx.moveTo(x - radius, y); ctx.lineTo(x + radius, y);
        ctx.moveTo(x, y - radius); ctx.lineTo(x, y + radius); ctx.stroke();
        ctx.font = `${13 / state.scale}px system-ui`;
        ctx.fillStyle = ctx.strokeStyle; ctx.fillText(dart.id, x + radius, y - radius);
      }
    });
    doc().ignore.forEach((points, index) => polygon(
      points, colors.ignore, selected()?.kind === "ignore" && selected().index === index, true
    ));
    if (state.pending.length) {
      path(state.pending, false); ctx.strokeStyle = colors[state.layer];
      ctx.lineWidth = 2 / state.scale; ctx.stroke();
      for (const point of state.pending) {
        ctx.beginPath(); ctx.arc(...point, 3 / state.scale, 0, Math.PI * 2); ctx.fillStyle = "#ffffff"; ctx.fill();
      }
    }
  }
  function nearestVertex(points, point) {
    let index = -1, distance = 9 / state.scale;
    points.forEach((vertex, i) => {
      const d = Math.hypot(vertex[0] - point[0], vertex[1] - point[1]);
      if (d < distance) { index = i; distance = d; }
    });
    return index;
  }
  canvas.addEventListener("pointerdown", run(event => {
    if (!state.history || state.drag) return;
    canvas.focus();
    const screen = cursor(event);
    if (event.button === 1 || (event.button === 0 && state.space)) {
      event.preventDefault();
      state.drag = {kind: "pan", screen, x: state.x, y: state.y};
    } else if (event.button === 0) {
      const point = source(screen);
      if (state.mode === "draw") {
        state.pending.push(point); refresh(); return;
      }
      if (state.mode === "tip") {
        mutate(next => E.setTip(next.darts[state.dart], point));
        state.mode = "edit"; message("Exact tip placed. Set board contact separately."); return;
      }
      const region = selected();
      const vertex = region ? nearestVertex(region.points, point) : -1;
      const tip = activeDart()?.tip;
      if (vertex >= 0) state.drag = {kind: "vertex", vertex, region, original: E.clone(doc())};
      else if (tip && Math.hypot(tip[0] - point[0], tip[1] - point[1]) < 12 / state.scale) {
        state.drag = {kind: "tip", original: E.clone(doc())};
      } else {
        const hit = regions().findIndex(region => {
          path(region.points);
          return ctx.isPointInPath(
            (screen[0]) * (window.devicePixelRatio || 1),
            (screen[1]) * (window.devicePixelRatio || 1)
          );
        });
        state.region = hit; refresh(); return;
      }
    } else return;
    canvas.setPointerCapture(event.pointerId);
  }));
  canvas.addEventListener("pointermove", event => {
    if (!state.drag) return;
    if (state.drag.kind === "pan") {
      const point = cursor(event);
      state.x = state.drag.x + point[0] - state.drag.screen[0];
      state.y = state.drag.y + point[1] - state.drag.screen[1]; render(); return;
    }
    const point = source(cursor(event), true);
    if (state.drag.kind === "vertex") E.polygonPoints(doc(), state.dart, state.drag.region)[state.drag.vertex] = point;
    else E.setTip(activeDart(), point);
    render();
  });
  function endDrag(cancel = false) {
    if (!state.drag) return;
    if (state.drag.original) {
      const changed = E.clone(doc());
      state.history.document = state.drag.original;
      if (!cancel) state.history.commit(changed);
    }
    state.drag = null; refresh();
  }
  canvas.addEventListener("pointerup", () => endDrag());
  canvas.addEventListener("pointercancel", () => endDrag(true));
  canvas.addEventListener("contextmenu", run(event => {
    event.preventDefault();
    if (!state.history) return;
    if (state.mode === "draw") { state.pending.pop(); refresh(); return; }
    const region = selected();
    if (!region) return;
    const index = nearestVertex(region.points, source(cursor(event), true));
    if (index < 0) return;
    if (region.points.length <= 3) throw new Error("A polygon needs three vertices. Delete the polygon instead.");
    mutate(next => E.polygonPoints(next, state.dart, region).splice(index, 1));
  }));
  canvas.addEventListener("dblclick", run(event => {
    if (state.mode !== "edit" || !selected()) return;
    endDrag();
    const region = selected(), point = source(cursor(event));
    const edge = E.nearestEdge(region.points, point);
    if (edge.distance <= 10 / state.scale) mutate(next => E.polygonPoints(next, state.dart, region).splice(edge.index + 1, 0, point));
  }));
  canvas.addEventListener("wheel", event => {
    if (!state.history || state.busy) return;
    event.preventDefault(); zoom(Math.exp(-event.deltaY * 0.0015), cursor(event));
  }, {passive: false});
  function finish() {
    if (state.mode !== "draw") return;
    if (state.pending.length < 3) throw new Error("Click at least three vertices before finishing.");
    const kind = state.layer === "ignore" ? "ignore" : state.layer === "silhouette" ? "silhouette" : "visible";
    mutate(next => E.addPolygon(next, state.pendingDart, state.layer, state.pending));
    cancelMode();
    state.region = regions().findLastIndex(region => region.kind === kind);
    refresh();
    message("Polygon added. Select its entry to edit vertices. Save reports invalid/self-crossing outlines.");
  }
  async function save(next = false) {
    requireNoDrawing();
    if (doc().status === "completed") {
      const errors = E.completeErrors(doc());
      if (errors.length) throw new Error(errors.join(" "));
    }
    state.busy = true;
    try {
      const result = await api(`/api/annotation?name=${encodeURIComponent(doc().image)}`, {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-Annotation-Token": state.token},
        body: JSON.stringify({annotation: doc(), revision: state.revision})
      });
      state.revision = result.revision; state.saved = JSON.stringify(doc());
      state.photos[state.index].status = doc().status;
      $("photos").options[state.index].textContent = `${doc().image} [${doc().status}]`;
      refresh(); message(`Saved ${doc().image} (${doc().status}).`);
    } finally { state.busy = false; }
    if (next) await load(state.index + 1, false);
  }
  function bind(id, callback) { $(id).addEventListener("click", run(callback)); }
  bind("draw", () => {
    requireNoDrawing();
    if ($("layer").value !== "ignore" && !activeDart()) throw new Error("Add/select a dart first.");
    state.mode = "draw"; state.layer = $("layer").value; state.pendingDart = state.dart; state.pending = [];
    message(`Drawing ${state.layer}. Click visible boundary vertices, then Enter. Space+drag pans.`); refresh();
  });
  bind("finish", finish);
  bind("cancel", () => { cancelMode(); message("Drawing canceled; existing polygons are unchanged."); });
  bind("tip", () => {
    requireNoDrawing(); state.mode = "tip"; message("Click the physical tip. Pan/zoom first if needed.");
  });
  bind("unknown-tip", () => { requireNoDrawing(); state.mode = "edit"; mutate(next => E.setTip(next.darts[state.dart], null)); });
  bind("add-dart", () => {
    requireNoDrawing(); state.mode = "edit";
    mutate(next => {
      let i = 1; while (next.darts.some(d => d.id === `dart-${i}`)) i++;
      next.darts.push({id: `dart-${i}`, visible: [], silhouette: [], tip_status: "unknown", tip: null, contact: "unknown", score: null});
    });
    state.dart = doc().darts.length - 1; state.region = -1; refresh();
  });
  bind("remove-dart", () => {
    requireNoDrawing();
    if (!window.confirm(`Remove ${activeDart().id} and all its regions/tip?`)) return;
    mutate(next => next.darts.splice(state.dart, 1)); state.region = -1; state.mode = "edit"; refresh();
  });
  bind("delete", () => { requireNoDrawing(); mutate(next => E.removePolygon(next, state.dart, selected())); state.region = -1; refresh(); });
  bind("copy", () => {
    requireNoDrawing();
    mutate(next => next.darts[state.dart].silhouette.push(E.clone(selected().points)));
    message("Copied to inferred silhouette; visible ground truth is unchanged.");
  });
  bind("undo", () => {
    if (state.pending.length) state.pending.pop();
    else { state.history.undo(); state.mode = "edit"; }
    refresh();
  });
  bind("redo", () => { requireNoDrawing(); state.history.redo(); refresh(); });
  bind("fit", fit);
  bind("zoom-in", () => zoom(1.3));
  bind("zoom-out", () => zoom(1 / 1.3));
  bind("previous", () => load(state.index - 1));
  bind("next", () => load(state.index + 1));
  bind("unfinished", () => {
    for (let i = 1; i < state.photos.length; i++) {
      const index = (state.index + i) % state.photos.length;
      if (state.photos[index].status !== "completed") return load(index);
    }
    message("No other unfinished photos.");
  });
  bind("save", () => save());
  bind("save-next", () => save(true));
  bind("download", () => {
    requireNoDrawing();
    const url = URL.createObjectURL(new Blob([JSON.stringify(doc(), null, 2)], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = doc().image + ".json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    message("Downloaded a copy; this does not mark the server file saved.");
  });
  $("photos").addEventListener("change", run(event => load(Number(event.target.value))));
  $("darts").addEventListener("change", run(event => {
    requireNoDrawing(); state.dart = Number(event.target.value); state.region = -1; state.mode = "edit"; refresh();
  }));
  $("regions").addEventListener("change", run(event => {
    requireNoDrawing(); state.region = Number(event.target.value); state.mode = "edit"; refresh();
  }));
  $("contact").addEventListener("change", run(event => mutate(next => { next.darts[state.dart].contact = event.target.value; })));
  $("score-number").append(option("", "Not labeled / unknown"), option("0", "Miss (0)"),
    ...Array.from({length: 20}, (_, i) => option(String(i + 1), String(i + 1))),
    option("25", "Bull (25; double = 50)"));
  $("score-number").addEventListener("change", run(event => {
    const number = event.target.value === "" ? null : Number(event.target.value);
    // A newly chosen number starts at single rather than inheriting another dart/ring.
    mutate(next => E.setScore(next.darts[state.dart], number));
  }));
  $("score-multiplier").addEventListener("change", run(event => {
    mutate(next => E.setScore(next.darts[state.dart], activeDart().score.number, event.target.value));
  }));
  $("status").addEventListener("change", run(event => {
    requireNoDrawing();
    const value = event.target.value;
    if (value === "completed") {
      const errors = E.completeErrors(doc());
      if (errors.length) { refresh(); throw new Error(errors.join(" ")); }
      if (!window.confirm("Confirm every visible dart is labeled. Unmarked pixels outside ignore regions will count as background.")) { refresh(); return; }
    }
    mutate(next => { next.status = value; });
  }));
  $("show-inferred").addEventListener("change", render);
  window.addEventListener("keydown", run(event => {
    if (!state.history || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement.tagName)) return;
    if (event.code === "Space") { state.space = true; event.preventDefault(); }
    if (event.key === "Enter") { event.preventDefault(); finish(); }
    if (event.key === "Escape") cancelMode();
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault(); $(event.shiftKey ? "redo" : "undo").click();
    }
  }));
  window.addEventListener("keyup", event => { if (event.code === "Space") state.space = false; });
  window.addEventListener("blur", () => { state.space = false; endDrag(true); });
  window.addEventListener("beforeunload", event => { if (dirty()) { event.preventDefault(); event.returnValue = ""; } });
  new ResizeObserver(render).observe($("viewport"));
  const help = {
    photos: "Select a local photograph. Unsaved edits trigger a discard warning.",
    previous: "Open the previous photograph; save first to keep your changes.",
    next: "Open the next photograph; save first to keep your changes.",
    unfinished: "Jump to the next new or draft photograph.",
    save: "Validate and save labels to disk. Draft work can be resumed later.",
    "save-next": "Save this photograph, then open the next one.",
    download: "Download a JSON copy. This does not save changes to the server.",
    status: "Completed asserts that every visible dart is labeled. Unmarked pixels become background except ignore regions.",
    darts: "Each dart has its own visible regions, optional silhouette, tip, contact and score.",
    "add-dart": "Add another dart. Three is a convenience default, not a required count.",
    "remove-dart": "Remove this dart and all its labels. Undo restores it.",
    contact: "Embedded means the tip enters the board. Not embedded is a loose/missed dart. Choose unknown if unclear.",
    tip: "Click the exact physical endpoint/contact location. Include visible metal tip pixels in the shaft/barrel polygon too.",
    "unknown-tip": "No tip coordinate is saved. Use when the tip is hidden or cannot be identified; never guess.",
    "score-number": "Optional human-labeled sector number, not total points. Leave unknown when unsure. Use 0 for a miss or 25 for bull.",
    "score-multiplier": "Single is default. Double multiplies by 2; triple by 3. Bull: single 25 or double 50. Score is not inferred from the tip.",
    layer: "Visible flight: fins. Shaft/barrel: body, shaft and visible metal tip. Uncertain part: definitely dart, part unclear. Inferred silhouette: includes hidden shape, not ground truth. Ignore: dart/background truly ambiguous.",
    draw: "Click boundary vertices in order, then Enter or Finish. Separate disconnected visible pieces into multiple polygons.",
    finish: "Close the polygon with at least three vertices. Self-crossing outlines are rejected on save.",
    cancel: "Discard only the unfinished polygon; saved regions remain unchanged.",
    regions: "Select a polygon to edit. Drag vertices, double-click an edge to insert, right-click a vertex to remove.",
    copy: "Create an independent inferred silhouette from the selected visible region. Editing it leaves visible ground truth unchanged.",
    delete: "Delete the selected polygon only, not the dart. Undo restores it.",
    undo: "Undo the last label edit; while drawing, remove the last vertex.",
    redo: "Restore an undone label edit. A new edit clears redo history.",
    fit: "Fit the entire original photograph into the canvas without changing labels.",
    "zoom-in": "Zoom in around the canvas center. The mouse wheel zooms around the pointer.",
    "zoom-out": "Zoom out. Hold Space and drag or use the middle button to pan.",
    "show-inferred": "Show/hide optional complete inferred silhouettes. They never count as visible segmentation truth.",
  };
  for (const [id, text] of Object.entries(help)) {
    const control = $(id);
    control.title = text;
    const label = control.closest("label");
    const badge = document.createElement("span");
    badge.className = "context-help"; badge.tabIndex = 0;
    badge.textContent = "?"; badge.setAttribute("aria-label", `Help: ${text}`);
    const popup = document.createElement("span");
    popup.className = "help-popup"; popup.textContent = text;
    badge.append(popup);
    if (label) label.append(badge); else control.after(badge);
  }
  async function start() {
    state.token = (await api("/api/session")).token;
    state.photos = await api("/api/images");
    $("photos").replaceChildren(...state.photos.map((photo, index) => option(index, `${photo.name} [${photo.status}]`)));
    if (!state.photos.length) throw new Error("No supported photographs found.");
    await load(0, false);
  }
  start().catch(error => message(error.message, true));
})();
