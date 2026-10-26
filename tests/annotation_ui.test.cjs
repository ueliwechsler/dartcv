const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const E = require("../src/dartcv/annotation_ui/editor.js");

class Element {
  constructor(tagName = "BUTTON") {
    this.tagName = tagName;
    this.listeners = {};
    this.attributes = {};
    this.value = "";
    this.checked = true;
    this.options = [];
  }
  addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); }
  async emit(type, event = {}) {
    for (const handler of this.listeners[type] || []) {
      await handler({target: this, preventDefault() {}, ...event});
    }
  }
  click() { return this.disabled ? Promise.resolve() : this.emit("click"); }
  focus() { document.activeElement = this; }
  setAttribute(name, value) { this.attributes[name] = value; }
  replaceChildren(...children) { this.options = children; }
  append(...children) { this.options.push(...children); }
  closest() { return null; }
  after() {}
  querySelector() { return new Element(); }
}

const elements = new Map();
const document = {
  activeElement: new Element("BODY"),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  createElement: tag => new Element(tag.toUpperCase()),
};
const $ = id => document.getElementById(id);
for (const id of ["status", "darts", "regions", "photos", "contact"]) $(id).tagName = "SELECT";
const canvas = $("canvas");
canvas.tagName = "CANVAS";
canvas.clientWidth = canvas.clientHeight = 100;
canvas.getBoundingClientRect = () => ({left: 0, top: 0});
canvas.setPointerCapture = () => {};
canvas.getContext = () => new Proxy({}, {
  get: () => () => {},
  set: (target, key, value) => { target[key] = value; return true; },
});
const window = new Element("WINDOW");
window.DartEditor = E;
window.confirm = () => true;
window.devicePixelRatio = 1;
const initial = {
  schema_version: 1, image: "one.jpg", width: 100, height: 100,
  sha256: "a".repeat(64), status: "ai-draft",
  darts: [{
    id: "dart-1", visible: [{part: "flight", points: [[10,10],[80,10],[80,80],[10,80]]}],
    silhouette: [], tip_status: "unknown", tip: null, contact: "unknown", score: null,
  }], ignore: [],
};
let saved;
const context = {
  window, document, console,
  Image: class { naturalWidth = 100; naturalHeight = 100; async decode() {} },
  ResizeObserver: class { observe() {} },
  fetch: async (url, options = {}) => {
    let body;
    if (url === "/api/session") body = {token: "local"};
    else if (url === "/api/folders") body = {folders: ["single_dart"], default: "single_dart"};
    else if (url.startsWith("/api/images")) body = [
      {name: "one.jpg", status: "ai-draft"}, {name: "two.jpg", status: "completed"},
      {name: "three.jpg", status: "ai-draft"},
    ];
    else if (options.method === "POST") {
      saved = JSON.parse(options.body).annotation;
      body = {revision: "saved", status: saved.status};
    } else body = {
      annotation: {...E.clone(initial), image: new URL(url, "http://localhost").searchParams.get("name")},
      revision: "original",
    };
    return {ok: true, json: async () => body};
  },
};

async function key(value, extra = {}) {
  await window.emit("keydown", {key: value, ...extra});
}
async function point(x, y, type = "pointerdown", extra = {}) {
  await canvas.emit(type, {clientX: 2.5 + x * .95, clientY: 2.5 + y * .95, button: 0, ...extra});
}
async function save() {
  await $("save").click();
  return saved.darts[0].visible[0].points;
}

async function main() {
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../src/dartcv/annotation_ui/app.js"), "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal($("status").value, "ai-draft");
  assert.match($("photos").options[0].textContent, /\[ai-draft\]/);
  $("regions").value = "0";
  await $("regions").emit("change");
  canvas.focus();
  await key("a");
  assert.equal($("add-vertex").attributes["aria-pressed"], "true");
  await point(35, 10);
  await point(60, 10);
  assert.equal((await save()).length, 6);
  assert.equal(saved.status, "ai-draft");
  await key("Escape");
  assert.equal($("add-vertex").attributes["aria-pressed"], "false");
  await key("z", {ctrlKey: true});
  assert.equal((await save()).length, 5);
  await key("z", {ctrlKey: true, shiftKey: true});
  assert.equal((await save()).length, 6);
  await $("remove-vertex").click();
  assert.equal($("remove-vertex").attributes["aria-pressed"], "true");
  await point(35, 10);
  await point(60, 10);
  await point(10, 10);
  assert.equal((await save()).length, 3);
  await point(80, 10);
  assert.match($("message").textContent, /three vertices/);
  assert.equal((await save()).length, 3);
  await key("Escape");
  const field = new Element("INPUT");
  field.focus();
  await key("a");
  assert.equal($("add-vertex").attributes["aria-pressed"], "false");
  canvas.focus();
  await key("r", {ctrlKey: true});
  assert.equal($("remove-vertex").attributes["aria-pressed"], "false");
  await $("add-vertex").click();
  await point(5, 95);
  assert.match($("message").textContent, /Click an edge/);
  assert.equal((await save()).length, 3);
  await key("Escape");
  await point(80, 45, "dblclick");
  assert.equal((await save()).length, 4);
  await point(80, 45, "contextmenu");
  assert.equal((await save()).length, 3);
  assert.match($("add-vertex").title, /Double-click an edge/);
  assert.match($("remove-vertex").title, /Right-click a vertex/);
  $("status").value = "completed";
  await $("status").emit("change");
  await save();
  assert.equal(saved.status, "completed");
  await $("unfinished").click();
  assert.equal($("photos").value, 2);
  assert.equal($("status").value, "ai-draft");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
