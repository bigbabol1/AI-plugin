/*
 * AI Plugin sidebar page.
 *
 * One plain web component, no build step and no Home Assistant internal
 * elements (they are not a public interface and change between releases).
 * Talks only to the admin-only `ai_plugin/*` websocket commands.
 */

const TABS = [
  { id: "core", label: "Core", icon: "M12 2l8.66 5v10L12 22l-8.66-5V7z" },
  { id: "model", label: "Model", icon: "M4 4h16v16H4zM9 9h6v6H9z" },
  { id: "dialog", label: "Dialog", icon: "M4 5h16v10H8l-4 4z" },
  { id: "voice", label: "Voice", icon: "M12 3a3 3 0 013 3v6a3 3 0 01-6 0V6a3 3 0 013-3zM5 11a7 7 0 0014 0M12 18v3" },
  { id: "memory", label: "Memory", icon: "M6 3h12v18l-6-4-6 4z" },
  { id: "tools", label: "Tools", icon: "M14 6l4 4-8 8H6v-4zM16 4l4 4" },
];

// Which tab shows which fields of the "advanced" section. Fields not named
// here land in the Dialog tab's "Other" card, so a new option is never lost.
const ADVANCED_GROUPS = {
  model: [
    ["Model tuning", ["context_window", "enable_thinking", "temperature", "top_p", "max_tokens",
      "keep_alive", "max_tool_iterations", "response_timeout", "prune_tool_schemas", "summarization_enabled"]],
  ],
  dialog: [
    ["Conversation", ["system_prompt", "voice_mode", "continue_conversation", "follow_up_delay"]],
    ["Satellites", ["self_echo_filter", "timer_announce", "feedback_loop_devices", "offline_fallback"]],
    ["Location", ["location_bias", "location_entity"]],
    ["This page", ["show_panel"]],
  ],
};

const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ago = (iso) => {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
};

const when = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  const year = d.getFullYear() !== new Date().getFullYear() ? { year: "numeric" } : {};
  return d.toLocaleString(undefined, { weekday: "short", day: "2-digit", month: "short", ...year, hour: "2-digit", minute: "2-digit" });
};

// dd.mm.yyyy, HH:MM whatever the browser's locale.
const dateTime = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getDate())}.${p(d.getMonth() + 1)}.${d.getFullYear()}, ${p(d.getHours())}:${p(d.getMinutes())}`;
};

const icon = (path, size = 18) =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="${path}"/></svg>`;

const STYLE = `
:host {
  --bg: #04060c;
  --bg2: #0a0f1d;
  --glass: rgba(14, 22, 42, 0.62);
  --glass-hi: rgba(22, 34, 64, 0.75);
  --line: rgba(120, 200, 230, 0.13);
  --line-hi: rgba(120, 200, 230, 0.34);
  --cyan: #62d3ea;
  --violet: #9184f0;
  --magenta: #e077c8;
  --lime: #6dffb0;
  --amber: #ffc04d;
  --red: #ff5370;
  --text: #dbe7ff;
  --muted: #7d8bb0;
  --mono: ui-monospace, "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: "Inter", "Segoe UI", Roboto, system-ui, sans-serif;
  display: block;
  height: 100vh;
  height: 100dvh;
  overflow-y: auto;
  overflow-x: hidden;
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}
* { box-sizing: border-box; }
.space {
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(145,132,240,0.08), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(98,211,234,0.06), transparent 60%),
    linear-gradient(180deg, #050814 0%, #03050a 100%);
}
.grid {
  position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: .22;
  background-image:
    linear-gradient(rgba(59,232,255,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(59,232,255,0.05) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: radial-gradient(ellipse at 50% 30%, #000 30%, transparent 80%);
  -webkit-mask-image: radial-gradient(ellipse at 50% 30%, #000 30%, transparent 80%);
}
.wrap { position: relative; z-index: 1; max-width: 1240px; margin: 0 auto; padding: 18px 20px 120px; }

header { display: flex; align-items: center; gap: 14px; padding: 6px 0 16px; min-width: 0; }
header > * { flex: none; }
header > .brand { flex: 0 1 auto; min-width: 0; }
.brand h1 { overflow: hidden; text-overflow: ellipsis; }
.menu { display: none; background: none; border: 0; color: var(--text); padding: 6px; cursor: pointer; }
:host([narrow]) .menu { display: inline-flex; }
.logo { width: 42px; height: 42px; position: relative; flex: none; }
.logo svg { position: absolute; inset: 0; }
@keyframes spin { to { transform: rotate(360deg); } }
.brand h1 {
  margin: 0; font-size: 17px; letter-spacing: .2em; font-weight: 600; text-transform: uppercase; white-space: nowrap;
  color: #e8f1ff;
}
.brand small { display: block; font-family: var(--mono); color: var(--muted); font-size: 11px; letter-spacing: .2em; text-transform: uppercase; margin-top: 2px; }
.hspace { flex: 1; }
.pill {
  font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line); background: var(--glass);
  display: inline-flex; align-items: center; gap: 8px; white-space: nowrap;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); box-shadow: 0 0 6px currentColor; }
.dot.ok { background: var(--lime); color: transparent; }
.dot.warn { background: var(--amber); color: var(--amber); animation: pulse 1.2s ease-in-out infinite; }
.dot.bad { background: var(--red); color: var(--red); animation: pulse .8s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .35; } }
select.entry { max-width: 220px; }

nav {
  display: flex; gap: 4px; padding: 4px; border: 1px solid var(--line); border-radius: 14px;
  background: rgba(8,12,24,.7); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  position: sticky; top: 8px; z-index: 5; overflow-x: auto; scrollbar-width: none;
}
nav button {
  flex: 1 0 auto; display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  background: none; border: 0; color: var(--muted); padding: 10px 14px; border-radius: 10px; cursor: pointer;
  font-family: var(--mono); font-size: 12px; letter-spacing: .18em; text-transform: uppercase; transition: .2s;
}
nav button:hover { color: var(--text); background: rgba(59,232,255,.06); }
nav button.on {
  color: #fff; background: rgba(98,211,234,.1);
  box-shadow: inset 0 0 0 1px var(--line-hi);
}
nav button.on svg { color: var(--cyan); }
nav .badge { width: 7px; height: 7px; border-radius: 50%; background: var(--magenta); }

main { margin-top: 18px; display: grid; gap: 16px; }
.cols { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }
@media (max-width: 980px) { .cols { grid-template-columns: minmax(0, 1fr); } }
.card {
  position: relative; border-radius: 18px; padding: 18px 18px 16px; container-type: inline-size; min-width: 0;
  background: linear-gradient(160deg, var(--glass-hi), var(--glass));
  border: 1px solid var(--line);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 16px 40px -30px rgba(0,0,0,.9);
}
.card::before {
  content: ""; position: absolute; left: 18px; right: 18px; top: -1px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent); opacity: .25;
}
@keyframes rise { from { opacity: 0; transform: translateY(6px); } }
.card h2 {
  margin: 0 0 4px; font-family: var(--mono); font-weight: 600; font-size: 12px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--cyan); display: flex; align-items: center; gap: 10px;
}
.card h2::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, var(--line-hi), transparent); }
.card .sub { color: var(--muted); font-size: 12.5px; line-height: 1.5; margin: 4px 0 12px; white-space: pre-line; }
.wide { grid-column: 1 / -1; }

/* hero */
  position: absolute; inset: 40px; border-radius: 50%;
  background: radial-gradient(circle at 40% 35%, #dff8ff, var(--cyan) 38%, rgba(98,211,234,.15) 70%, transparent 74%);
  box-shadow: 0 0 22px rgba(98,211,234,.35);
}
.big { font-size: 22px; font-weight: 600; letter-spacing: .02em; margin: 2px 0 6px; word-break: break-all; }
.linkhead { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 8px 0 2px; }
.linkhead b { font-family: var(--mono); font-size: 12px; letter-spacing: .14em; text-transform: uppercase; }
.linkgrid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 300px); gap: 18px; align-items: start; }
@container (max-width: 640px) { .linkgrid { grid-template-columns: minmax(0, 1fr); } }
.sparks { display: grid; gap: 12px; }
.spark { padding: 10px 12px; border-radius: 12px; background: rgba(7,12,26,.5); border: 1px solid var(--line); min-width: 0; }
.spark-t { font-family: var(--mono); font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.spark-t span { letter-spacing: .04em; text-transform: none; color: var(--text); }
.spark svg { display: block; width: 100%; height: 54px; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; font-family: var(--mono); font-size: 12.5px; }
.kv dt { color: var(--muted); text-transform: uppercase; letter-spacing: .14em; font-size: 11px; padding-top: 1px; }
.kv dd { margin: 0; color: var(--text); word-break: break-all; }

.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
.tile {
  border-radius: 14px; padding: 14px; background: rgba(7,12,26,.55); border: 1px solid var(--line);
  position: relative; overflow: hidden;
}
.tile b { display: block; font-size: 22px; font-weight: 600; font-family: var(--mono); color: #eef5ff; }
.tile span { font-family: var(--mono); font-size: 10.5px; letter-spacing: .18em; color: var(--muted); text-transform: uppercase; }
.tile.a b { color: var(--amber); }

.bars { display: grid; gap: 8px; }
.bar { display: grid; grid-template-columns: 150px 1fr 40px; gap: 10px; align-items: center; font-family: var(--mono); font-size: 12px; }
.bar .track { height: 6px; border-radius: 6px; background: rgba(255,255,255,.06); overflow: hidden; }
.bar .fill { height: 100%; border-radius: 6px; background: var(--cyan); opacity: .8; }
.bar.amber .fill { background: var(--amber); }
.bar em { font-style: normal; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.feed { display: grid; gap: 10px; max-height: 620px; overflow: auto; padding-right: 4px; }
.turn { border-left: 2px solid var(--cyan); padding: 8px 12px; background: rgba(7,12,26,.5); border-radius: 0 12px 12px 0; }
.turn.echo { border-color: var(--muted); opacity: .7; }
.turn.offline { border-color: var(--amber); }
.turn.error { border-color: var(--red); }
.turn .meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: .06em; }
.turn .q { margin: 6px 0 2px; color: #fff; }
.turn .a { color: var(--text); opacity: .85; font-size: 13px; }
.turn .a::before { content: "↳ "; color: var(--cyan); }
.chip {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--cyan); background: rgba(59,232,255,.06); white-space: nowrap;
}
.chip.v { color: var(--violet); border-color: rgba(157,107,255,.35); background: rgba(157,107,255,.08); }
.chip.a { color: var(--amber); border-color: rgba(255,192,77,.4); background: rgba(255,192,77,.08); }
.chip.r { color: var(--red); border-color: rgba(255,83,112,.4); background: rgba(255,83,112,.08); }
.chip.g { color: var(--lime); border-color: rgba(109,255,176,.35); background: rgba(109,255,176,.07); }
.chip.x { color: var(--muted); }
.chip.m { color: var(--magenta); border-color: rgba(255,79,216,.4); background: rgba(255,79,216,.08); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }
.chip { max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
button.chip.dev { cursor: pointer; font-size: 11.5px; padding: 4px 10px; }
button.chip.dev.off { color: var(--muted); background: transparent; }
button.chip.dev:hover { border-color: var(--line-hi); }
.empty { color: var(--muted); font-family: var(--mono); font-size: 12px; letter-spacing: .1em; padding: 18px 0; text-align: center; }

/* fields */
.fields { display: grid; gap: 4px; }
.field { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, 50%); gap: 6px 18px; align-items: center; padding: 10px 0; border-top: 1px solid rgba(80,220,255,.07); }
.field:first-child { border-top: 0; }
.field.stack { grid-template-columns: 1fr; }
.field.bool { grid-template-columns: minmax(0, 1fr) auto; }
@container (max-width: 470px) { .field:not(.bool) { grid-template-columns: minmax(0, 1fr); } }
.field label { font-weight: 600; font-size: 13.5px; }
.field .help { grid-column: 1 / -1; color: var(--muted); font-size: 12px; line-height: 1.45; margin-top: -2px; }
.field .err { grid-column: 1 / -1; color: var(--red); font-family: var(--mono); font-size: 12px; }
.field.dirty label::after { content: " ●"; color: var(--magenta); }
input, select, textarea {
  width: 100%; font: inherit; color: var(--text); background: rgba(3,6,14,.75);
  border: 1px solid var(--line); border-radius: 10px; padding: 9px 11px; outline: none; transition: .15s;
}
textarea { font-family: var(--mono); font-size: 12.5px; min-height: 180px; resize: vertical; line-height: 1.5; }
input:focus, select:focus, textarea:focus { border-color: var(--line-hi); box-shadow: 0 0 0 3px rgba(98,211,234,.1); }
select option { background: #0a0f1d; }
.row { display: flex; gap: 8px; align-items: center; }
.row > * { min-width: 0; }
input[type=range] { -webkit-appearance: none; appearance: none; height: 4px; padding: 0; border: 0; background: rgba(98,211,234,.35); border-radius: 4px; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #e8f6ff; cursor: pointer; }
input[type=range]::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: #e8f6ff; cursor: pointer; border: 0; }
input.num { width: 92px; flex: none; font-family: var(--mono); text-align: right; }
.unit { font-family: var(--mono); color: var(--muted); font-size: 12px; }

.switch { justify-self: end; position: relative; width: 50px; height: 26px; flex: none; }
.switch input { opacity: 0; width: 100%; height: 100%; margin: 0; position: absolute; cursor: pointer; z-index: 1; }
.switch i { position: absolute; inset: 0; border-radius: 26px; background: rgba(255,255,255,.08); border: 1px solid var(--line); transition: .2s; }
.switch i::after { content: ""; position: absolute; top: 3px; left: 3px; width: 18px; height: 18px; border-radius: 50%; background: var(--muted); transition: .2s; }
.switch input:checked + i { background: rgba(98,211,234,.32); border-color: var(--line-hi); }
.switch input:checked + i::after { left: 27px; background: #fff; }

button.btn {
  font-family: var(--mono); font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
  color: #fff; padding: 9px 16px; border-radius: 10px; cursor: pointer; border: 1px solid var(--line-hi);
  background: rgba(98,211,234,.14); transition: .15s; white-space: nowrap;
}
button.btn:hover { background: rgba(98,211,234,.22); }
button.btn:disabled { opacity: .45; cursor: default; }
button.btn.ghost { background: transparent; border-color: var(--line); color: var(--text); }
button.btn.danger { background: rgba(255,83,112,.12); border-color: rgba(255,83,112,.45); color: #ffd3dc; }
button.btn.sm { padding: 6px 10px; font-size: 11px; }
button.x { background: none; border: 0; color: var(--muted); cursor: pointer; padding: 4px 6px; border-radius: 6px; font-size: 16px; line-height: 1; }
button.x:hover { color: var(--red); background: rgba(255,83,112,.1); }

.savebar {
  position: fixed; left: 50%; bottom: 22px; transform: translateX(-50%); z-index: 20;
  display: flex; align-items: center; gap: 14px; padding: 10px 12px 10px 18px; border-radius: 16px;
  background: rgba(10,14,28,.94); border: 1px solid var(--line-hi);
  box-shadow: 0 20px 50px rgba(0,0,0,.6);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  font-family: var(--mono); font-size: 12px; letter-spacing: .1em; animation: rise .25s ease both;
  max-width: calc(100vw - 32px);
}
.savebar span { color: var(--magenta); text-transform: uppercase; }

.toast {
  position: fixed; right: 20px; top: 20px; z-index: 30; padding: 12px 16px; border-radius: 12px;
  background: rgba(10,14,28,.95); border: 1px solid var(--line-hi); font-family: var(--mono); font-size: 12.5px;
  box-shadow: 0 10px 30px rgba(0,0,0,.5); animation: rise .25s ease both; max-width: 380px;
}
.toast.bad { border-color: rgba(255,83,112,.6); }

.list { display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr); }
.item { display: flex; gap: 10px; align-items: flex-start; padding: 10px 12px; border-radius: 12px; background: rgba(7,12,26,.5); border: 1px solid rgba(80,220,255,.08); }
.item .grow { flex: 1; min-width: 0; }
.item .t { color: var(--muted); font-family: var(--mono); font-size: 11px; letter-spacing: .06em; margin-bottom: 3px; }
.addfact { display: grid; grid-template-columns: minmax(0, 160px) minmax(0, 1fr) auto; gap: 8px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); }
@container (max-width: 470px) { .addfact { grid-template-columns: minmax(0, 1fr); } }
textarea.fact-edit { min-height: 0; font-family: var(--sans); font-size: 13.5px; }
button.x.edit { font-size: 15px; color: var(--text); opacity: .75; }
button.x.edit:hover { opacity: 1; }
button.x.edit:hover { color: var(--cyan); background: rgba(98,211,234,.1); }
.owner { font-family: var(--mono); font-size: 11px; letter-spacing: .2em; text-transform: uppercase; color: var(--violet); margin: 12px 0 6px; }

.people { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); }
.person { border-radius: 16px; padding: 14px; background: rgba(7,12,26,.55); border: 1px solid var(--line); display: grid; gap: 10px; }
.person .head { display: flex; align-items: center; gap: 12px; }
.avatar {
  width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-weight: 800; font-size: 17px; color: #fff;
  background: rgba(98,211,234,.25); border: 1px solid var(--line-hi);
}
.avatar.off { background: rgba(255,255,255,.06); color: var(--muted); box-shadow: none; }
.wave { display: flex; gap: 2px; align-items: flex-end; height: 18px; }
.wave { display: none; }

.server { min-width: 0; border-radius: 14px; padding: 14px; background: rgba(7,12,26,.55); border: 1px solid var(--line); display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; }
.server .top { display: flex; gap: 10px; align-items: center; }
.server .top .grow { flex: 1; min-width: 0; font-family: var(--mono); font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.combo { position: relative; }
.menu-list {
  position: absolute; z-index: 10; left: 0; right: 0; top: calc(100% + 6px); max-height: 300px; overflow: auto;
  background: #0b1222; border: 1px solid var(--line-hi); border-radius: 12px; padding: 6px;
  box-shadow: 0 18px 40px rgba(0,0,0,.6); display: grid; gap: 2px;
}
.menu-list button {
  text-align: left; background: none; border: 0; color: var(--text); padding: 8px 10px; border-radius: 8px; cursor: pointer;
  font-family: var(--mono); font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.menu-list button:hover { background: rgba(98,211,234,.1); }
.menu-list button.on { color: var(--cyan); background: rgba(98,211,234,.08); }
.menu-list button.rescan { color: var(--muted); border-top: 1px solid var(--line); border-radius: 0 0 8px 8px; margin-top: 4px; }
.envrow { display: grid; grid-template-columns: 1fr 1fr auto; gap: 8px; }
.server input { font-family: var(--mono); font-size: 12.5px; }
@media (max-width: 600px) {
  .server .row { flex-wrap: wrap; } .wrap { padding: 12px 12px 120px; } .brand small { display: none; }
  header { gap: 10px; } .brand h1 { letter-spacing: .12em; }
  .pill.quiet { padding: 9px; } .pill.quiet .pt { display: none; }
}
.small { font-size: 12px; color: var(--muted); }
.loading { display: grid; place-items: center; min-height: 50vh; font-family: var(--mono); letter-spacing: .3em; color: var(--cyan); text-transform: uppercase; font-size: 12px; }
.loading i { display: block; width: 160px; height: 2px; margin-top: 14px; background: linear-gradient(90deg, transparent, var(--cyan), transparent); animation: sweep 1.2s linear infinite; background-size: 50% 100%; background-repeat: no-repeat; }
@keyframes sweep { from { background-position: -100% 0; } to { background-position: 200% 0; } }
a { color: var(--cyan); }
`;

class AIPluginPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._tab = this._savedTab();
    this._entryId = null;
    this._entries = [];
    this._status = null;
    this._settings = null; // {sections: {...}}
    this._edits = {}; // section -> {field: value}
    this._errors = {}; // section -> {field: text}
    this._voice = null;
    this._memory = null;
    this._mcp = null;
    this._mcpEdit = null; // working copy of servers
    this._models = null;
    this._busy = false;
    this._started = false;
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._started) {
      this._started = true;
      this._boot();
    }
  }

  get hass() {
    return this._hass;
  }

  set narrow(v) {
    this.toggleAttribute("narrow", !!v);
  }

  set panel(_v) {}
  set route(_v) {}

  connectedCallback() {
    this._timer = setInterval(() => this._tick(), 5000);
    this._vis = () => document.visibilityState === "visible" && this._tick();
    document.addEventListener("visibilitychange", this._vis);
  }

  disconnectedCallback() {
    clearInterval(this._timer);
    document.removeEventListener("visibilitychange", this._vis);
  }

  _savedTab() {
    try {
      const t = localStorage.getItem("ai_plugin_panel_tab");
      if (TABS.some((x) => x.id === t)) return t;
    } catch (_e) {
      /* storage blocked */
    }
    return "core";
  }

  _ws(type, data = {}) {
    const msg = { type, ...data };
    if (this._entryId && !("entry_id" in msg) && type !== "ai_plugin/entries" && type !== "ai_plugin/mcp/preset" && type !== "ai_plugin/memory/forget" && type !== "ai_plugin/memory/edit") {
      msg.entry_id = this._entryId;
    }
    return this._hass.callWS(msg);
  }

  async _boot() {
    this._renderShell();
    try {
      this._entries = await this._ws("ai_plugin/entries");
      const loaded = this._entries.find((e) => e.loaded);
      this._entryId = (loaded || this._entries[0] || {}).entry_id || null;
    } catch (e) {
      this._fatal(e);
      return;
    }
    if (!this._entryId || !this._entries.some((e) => e.loaded)) {
      this._fatal({ message: "No AI Plugin entry is loaded. Check Settings → Devices & services." });
      return;
    }
    await this._loadAll();
  }

  async _loadAll() {
    await Promise.all([this._loadStatus(), this._loadSettings()]);
    this._render();
    this._loadTabData();
  }

  async _loadStatus() {
    try {
      this._status = await this._ws("ai_plugin/status");
    } catch (e) {
      this._status = { error: e.message || String(e) };
    }
  }

  async _loadSettings() {
    try {
      const res = await this._ws("ai_plugin/settings", { language: this._hass.language });
      const sections = {};
      for (const s of res.sections) sections[s.id] = s;
      this._settings = sections;
    } catch (e) {
      this._toast(`Settings: ${e.message || e}`, true);
    }
  }

  async _loadTabData() {
    const t = this._tab;
    try {
      if (t === "voice") this._voice = await this._ws("ai_plugin/voice");
      if (t === "memory") this._memory = await this._ws("ai_plugin/memory");
      if (t === "tools" && !this._mcpEdit) {
        this._mcp = await this._ws("ai_plugin/mcp");
        this._mcpEdit = JSON.parse(JSON.stringify(this._mcp.servers));
      }
    } catch (e) {
      this._toast(e.message || String(e), true);
    }
    this._renderMain();
  }

  async _tick() {
    if (document.visibilityState !== "visible" || !this._entryId || this._busy) return;
    if (this._tab === "core") {
      await this._loadStatus();
      this._renderMain();
      this._renderHeaderStatus();
    }
  }

  // ── rendering ──────────────────────────────────────────────────────────

  _renderShell() {
    this.shadowRoot.innerHTML = `
      <style>${STYLE}</style>
      <div class="space"></div><div class="grid"></div>
      <div class="wrap">
        <header>
          <button class="menu" data-act="menu" title="Menu">${icon("M3 6h18M3 12h18M3 18h18", 22)}</button>
          <div class="logo">
            <svg viewBox="0 0 42 42"><defs><linearGradient id="lg" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="#3be8ff"/><stop offset="1" stop-color="#ff4fd8"/></linearGradient></defs>
              <g class="spin"><polygon points="21,2 38,11.5 38,30.5 21,40 4,30.5 4,11.5" fill="none" stroke="url(#lg)" stroke-width="1.6"/></g>
              <circle cx="21" cy="21" r="6" fill="url(#lg)"/><circle cx="21" cy="21" r="11" fill="none" stroke="#3be8ff" stroke-opacity=".35" stroke-dasharray="2 3"/></svg>
          </div>
          <div class="brand"><h1>AI Plugin</h1><small id="sub">neural home core</small></div>
          <div class="hspace"></div>
          <span id="entrysel"></span>
          <span class="pill" id="hstatus" title="linking"><span class="dot"></span><span class="pt">linking</span></span>
        </header>
        <nav id="nav"></nav>
        <main id="main"><div class="loading">Initialising<i></i></div></main>
      </div>
      <div id="savebar"></div>
      <div id="toast"></div>`;
    const root = this.shadowRoot;
    root.addEventListener("click", (e) => this._onClick(e));
    root.addEventListener("input", (e) => this._onInput(e));
    root.addEventListener("change", (e) => this._onInput(e, true));
  }

  _fatal(e) {
    this.shadowRoot.getElementById("main").innerHTML = `<div class="card"><h2>Offline</h2><div class="sub">${esc(e.message || e)}</div></div>`;
  }

  _render() {
    this._renderNav();
    this._renderHeaderStatus();
    this._renderMain();
  }

  _renderNav() {
    const nav = this.shadowRoot.getElementById("nav");
    nav.innerHTML = TABS.map((t) => {
      const dirty = this._tabDirty(t.id);
      return `<button data-tab="${t.id}" class="${this._tab === t.id ? "on" : ""}">${icon(t.icon, 16)}${t.label}${dirty ? '<span class="badge"></span>' : ""}</button>`;
    }).join("");
    const sel = this.shadowRoot.getElementById("entrysel");
    if (this._entries.length > 1) {
      sel.innerHTML = `<select class="entry" data-act="entry">${this._entries
        .map((e) => `<option value="${esc(e.entry_id)}" ${e.entry_id === this._entryId ? "selected" : ""}>${esc(e.title)}</option>`)
        .join("")}</select>`;
    } else sel.innerHTML = "";
  }

  _health() {
    const s = this._status;
    if (!s || s.error) return ["bad", "no link"];
    if (s.offline_for_s > 0) return ["warn", "fallback"];
    if (!s.health || !s.health.reachable) return ["bad", "model offline"];
    if (s.health.model_listed === false) return ["warn", "model missing"];
    return ["ok", "online"];
  }

  _renderHeaderStatus() {
    const [cls, text] = this._health();
    const el = this.shadowRoot.getElementById("hstatus");
    if (el) {
      el.innerHTML = `<span class="dot ${cls}"></span><span class="pt">${esc(text)}</span>`;
      el.title = text;
      el.classList.toggle("quiet", cls === "ok");
    }
    const sub = this.shadowRoot.getElementById("sub");
    if (sub && this._status && this._status.version) sub.textContent = `neural home core · v${this._status.version}`;
  }

  _renderMain() {
    const main = this.shadowRoot.getElementById("main");
    if (!main || !this._settings) return;
    // Keep the focused field and scroll when re-rendering.
    const scrollY = this.scrollTop;
    const html = {
      core: () => this._viewCore(),
      model: () => this._viewModel(),
      dialog: () => this._viewDialog(),
      voice: () => this._viewVoice(),
      memory: () => this._viewMemory(),
      tools: () => this._viewTools(),
    }[this._tab]();
    main.innerHTML = html;
    this.scrollTop = scrollY;
    this._renderSaveBar();
  }

  _renderSaveBar() {
    const bar = this.shadowRoot.getElementById("savebar");
    const dirty = Object.keys(this._edits).filter((k) => Object.keys(this._edits[k]).length);
    const mcpDirty = this._mcpDirty();
    if (!dirty.length && !mcpDirty) {
      bar.innerHTML = "";
      return;
    }
    const names = dirty.map((d) => (this._settings[d] && this._settings[d].title) || d);
    if (mcpDirty) names.push("MCP servers");
    bar.innerHTML = `<div class="savebar"><span>Unsaved</span><div class="small">${esc(names.join(" · "))}</div>
      <button class="btn ghost sm" data-act="discard">Discard</button>
      <button class="btn sm" data-act="saveall" ${this._busy ? "disabled" : ""}>${this._busy ? "Saving…" : "Save"}</button></div>`;
  }

  _tabDirty(tab) {
    const map = { model: ["provider"], dialog: [], voice: ["voice"], memory: ["memory"], tools: ["tools", "web_search"] };
    const adv = this._edits.advanced ? Object.keys(this._edits.advanced) : [];
    if (tab === "model" || tab === "dialog") {
      const names = new Set((ADVANCED_GROUPS[tab] || []).flatMap((g) => g[1]));
      if (adv.some((f) => names.has(f))) return true;
      if (tab === "dialog") {
        const known = new Set(Object.values(ADVANCED_GROUPS).flat().flatMap((g) => g[1]));
        if (adv.some((f) => !known.has(f))) return true;
      }
    }
    if (tab === "tools" && this._mcpDirty()) return true;
    return (map[tab] || []).some((s) => this._edits[s] && Object.keys(this._edits[s]).length);
  }

  // ── views ──────────────────────────────────────────────────────────────

  _viewCore() {
    const s = this._status;
    if (!s) return `<div class="loading">Reading telemetry<i></i></div>`;
    if (s.error) return `<div class="card"><h2>No link</h2><div class="sub">${esc(s.error)}</div></div>`;
    const [cls, text] = this._health();
    const a = s.activity || {};
    const c = a.counts || {};
    const h = s.health || {};
    const sg = a.safeguards || {};
    const sgTotal = Object.values(sg).reduce((x, y) => x + y, 0);
    const tools = a.tools || {};
    const tmax = Math.max(1, ...Object.values(tools));
    const smax = Math.max(1, ...Object.values(sg));
    const featuresOff = new Set(s.features_off || []);
    const turns = (a.turns || [])
      .map((t) => {
        const kind = t.kind || "answered";
        return `<div class="turn ${esc(kind)}">
          <div class="meta"><span>${esc(ago(t.time))}</span>${t.device_name ? `<span class="chip v">${esc(t.device_name)}</span>` : ""}
          <span>${esc(t.seconds)}s</span>${kind !== "answered" ? `<span class="chip ${kind === "error" ? "r" : kind === "offline" ? "a" : kind === "automation" ? "g" : "x"}">${esc(kind)}</span>` : ""}
          ${(t.tools || []).map((x) => `<span class="chip">${esc(x)}</span>`).join("")}
          ${(t.safeguards || []).map((x) => `<span class="chip a">⚠ ${esc(x)}</span>`).join("")}</div>
          <div class="q">${esc(t.text)}</div>${t.reply ? `<div class="a">${esc(t.reply)}</div>` : ""}</div>`;
      })
      .join("");
    const ol = h.ollama;
    let mem = "";
    if (ol) {
      if (ol.loaded) {
        const until = ol.until ? new Date(ol.until) : null;
        const keep = !until || until.getFullYear() > 2100 ? "kept loaded" : `unloads at ${until.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
        mem = `${esc(ol.size_gb)} GB · ${ol.gpu_pct != null ? `${esc(ol.gpu_pct)}% on GPU` : "?"}${ol.context ? ` · context ${esc(ol.context)}` : ""} · ${esc(keep)}`;
        if (ol.gpu_pct != null && ol.gpu_pct < 90) mem += ` <span style="color:var(--amber)">· partly on CPU, slower</span>`;
      } else {
        mem = `<span style="color:var(--amber)">not loaded — the next reply loads it first and is slow</span>`;
      }
    }
    const lat = (h.history || []).map((x) => (x.ok ? x.ms : null));
    const replies = (a.turns || []).filter((t) => t.kind === "answered").map((t) => t.seconds).reverse();
    return `
      <div class="card"><h2>Model link</h2>
        <div class="linkhead"><span class="dot ${cls}"></span><b>${esc(text)}</b>${h.reachable ? `<span class="small">${esc(h.ms)} ms</span>` : ""}
          <div class="hspace"></div>
          <button class="btn ghost sm" data-act="recheck">Check now</button>
          <button class="btn ghost sm" data-act="reload">Reload plugin</button></div>
        <div class="big">${esc(s.model || "no model")}</div>
        <div class="linkgrid">
          <dl class="kv">
            <dt>Server</dt><dd>${esc(s.server || "-")}</dd>
            ${h.reachable ? `<dt>Models</dt><dd>${esc(h.models)} on server${h.model_listed === false ? ' · <span style="color:var(--amber)">selected model not listed</span>' : ""}</dd>` : `<dt>Error</dt><dd style="color:var(--red)">${esc(h.error || "unreachable")}</dd>`}
            ${mem ? `<dt>In memory</dt><dd>${mem}</dd>` : ""}
            <dt>Running since</dt><dd>${esc(dateTime(a.started))}</dd>
            ${a.last_error ? `<dt>Last error</dt><dd style="color:var(--red)">${esc(a.last_error)}</dd>` : ""}
          </dl>
          <div class="sparks">
            ${this._spark(lat, "Server latency", "ms", (h.history || []).length)}
            ${this._spark(replies, "Reply time, last turns", "s", replies.length)}
          </div>
        </div></div>
      <div class="tiles">
        <div class="tile"><b>${c.turns || 0}</b><span>Turns</span></div>
        <div class="tile v"><b>${a.avg_seconds != null ? a.avg_seconds : "–"}</b><span>Avg seconds</span></div>
        <div class="tile a"><b>${sgTotal}</b><span>Safeguards</span></div>
        <div class="tile m"><b>${c.offline || 0}</b><span>Assist handoffs</span></div>
        <div class="tile"><b>${c.echo || 0}</b><span>Echoes dropped</span></div>
        <div class="tile v"><b>${s.tools_builtin}+${s.mcp.tools}</b><span>Tools HA+MCP</span></div>
        <div class="tile"><b>${s.mcp.connected}/${s.mcp.total}</b><span>MCP linked</span></div>
        <div class="tile m"><b>${s.voice.enabled ? s.voice.people : "off"}</b><span>Voices</span></div>
        <div class="tile v"><b>${s.diary_notes}</b><span>Diary notes</span></div>
        <div class="tile"><b>${s.scheduled}</b><span>Scheduled</span></div>
      </div>
      <div class="cols">
        <div class="card"><h2>Tool usage</h2><div class="sub">Since the last start.</div>
          ${Object.keys(tools).length ? `<div class="bars">${Object.entries(tools).map(([k, v]) => `<div class="bar"><em>${esc(k)}</em><div class="track"><div class="fill" style="width:${(v / tmax) * 100}%"></div></div><span>${v}</span></div>`).join("")}</div>` : `<div class="empty">No tools used yet</div>`}
        </div>
        <div class="card"><h2>Safeguards</h2><div class="sub">Checks that changed a reply since the last start.</div>
          ${sgTotal ? `<div class="bars">${Object.entries(sg).map(([k, v]) => `<div class="bar amber"><em>${esc(k)}</em><div class="track"><div class="fill" style="width:${(v / smax) * 100}%"></div></div><span>${v}</span></div>`).join("")}</div>` : `<div class="empty">None fired</div>`}
          <h2 style="margin-top:18px">Features</h2>
          <div class="chips">${(s.features || []).map((f) => `<span class="chip ${featuresOff.has(f) ? "x" : "g"}">${featuresOff.has(f) ? "○" : "●"} ${esc(this._fieldLabel("tools", f))}</span>`).join("")}</div>
        </div>
      </div>
      <div class="card"><h2>Live feed</h2><div class="sub">The last ${a.turns ? a.turns.length : 0} turns, kept in memory only and cleared on restart. Updates every 5 seconds.</div>
        <div class="feed">${turns || `<div class="empty">Waiting for the first turn</div>`}</div></div>`;
  }

  _spark(values, title, unit, count) {
    const pts = values.filter((v) => v != null);
    if (pts.length < 2) {
      return `<div class="spark"><div class="spark-t">${esc(title)}</div><div class="small">${count ? "collecting…" : "no data yet"}</div></div>`;
    }
    const W = 280, H = 54, max = Math.max(...pts) || 1, n = values.length;
    const x = (i) => (n === 1 ? W / 2 : (i * W) / (n - 1));
    const y = (v) => H - 4 - (v / max) * (H - 10);
    const line = values.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean).join(" ");
    const fails = values.map((v, i) => (v == null ? `<circle cx="${x(i).toFixed(1)}" cy="${H - 3}" r="2.5" fill="var(--red)"/>` : "")).join("");
    const last = pts[pts.length - 1];
    const avg = pts.reduce((p, q) => p + q, 0) / pts.length;
    const fmt = (v) => (unit === "ms" ? Math.round(v) : Math.round(v * 10) / 10);
    return `<div class="spark"><div class="spark-t">${esc(title)}<span>now ${fmt(last)} ${unit} · avg ${fmt(avg)} · max ${fmt(max)}</span></div>
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><polyline points="${line}" fill="none" stroke="var(--cyan)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>${fails}</svg></div>`;
  }

  _fieldLabel(section, name) {
    const sec = this._settings && this._settings[section];
    const f = sec && sec.fields.find((x) => x.name === name);
    return (f && f.label) || name;
  }

  _sectionCard(sid, opts = {}) {
    const sec = this._settings[sid];
    if (!sec) return "";
    const only = opts.only ? new Set(opts.only) : null;
    const skip = opts.skip ? new Set(opts.skip) : null;
    const fields = sec.fields.filter((f) => (!only || only.has(f.name)) && (!skip || !skip.has(f.name)));
    if (!fields.length) return "";
    const order = opts.only ? opts.only : null;
    if (order) fields.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
    return `<div class="card ${opts.wide ? "wide" : ""}"><h2>${esc(opts.title || sec.title || sid)}</h2>
      ${opts.sub !== undefined ? (opts.sub ? `<div class="sub">${esc(opts.sub)}</div>` : "") : sec.description ? `<div class="sub">${esc(sec.description)}</div>` : ""}
      ${opts.before || ""}
      <div class="fields">${fields.map((f) => this._field(sid, f)).join("")}</div>${opts.after || ""}</div>`;
  }

  _value(sid, name) {
    const e = this._edits[sid];
    if (e && name in e) return e[name];
    const sec = this._settings[sid];
    return sec ? sec.values[name] : undefined;
  }

  _field(sid, f) {
    const v = this._value(sid, f.name);
    const dirty = this._edits[sid] && f.name in this._edits[sid];
    const err = this._errors[sid] && this._errors[sid][f.name];
    const sel = f.selector || {};
    const key = Object.keys(sel)[0] || f.type;
    const conf = (sel[key] || {});
    const data = `data-s="${esc(sid)}" data-f="${esc(f.name)}"`;
    let control = "";
    let stack = false;
    if (key === "boolean") {
      control = `<label class="switch"><input type="checkbox" ${data} data-k="bool" ${v ? "checked" : ""}><i></i></label>`;
    } else if (key === "number") {
      const min = conf.min ?? 0, max = conf.max ?? 100, step = conf.step ?? 1;
      const unit = conf.unit_of_measurement ? `<span class="unit">${esc(conf.unit_of_measurement)}</span>` : "";
      const val = v === undefined || v === null ? "" : v;
      const range = conf.mode === "slider" || max - min <= 10
        ? `<input type="range" ${data} data-k="num" min="${min}" max="${max}" step="${step}" value="${esc(val === "" ? min : val)}">` : "";
      control = `<div class="row">${range}<input class="num" type="number" ${data} data-k="num" min="${min}" max="${max}" step="${step}" value="${esc(val)}" placeholder="auto">${unit}</div>`;
    } else if (key === "template" || (key === "text" && conf.multiline)) {
      control = `<textarea ${data} data-k="text" spellcheck="false">${esc(v ?? "")}</textarea>`;
      stack = true;
    } else if (key === "select") {
      const options = (conf.options || []).map((o) => (typeof o === "string" ? { value: o, label: o } : o));
      control = `<select ${data} data-k="text">${options
        .map((o) => `<option value="${esc(o.value)}" ${String(o.value) === String(v) ? "selected" : ""}>${esc(o.label)}</option>`)
        .join("")}</select>`;
    } else if (key === "device") {
      control = this._devicePicker(sid, f, v, data);
      stack = true;
    } else if (key === "entity") {
      const domains = [].concat((conf.filter && conf.filter[0] && conf.filter[0].domain) || conf.domain || []);
      const ids = Object.keys(this._hass.states)
        .filter((id) => !domains.length || domains.includes(id.split(".")[0]))
        .sort();
      control = `<select ${data} data-k="text"><option value="">— none —</option>${ids
        .map((id) => `<option value="${esc(id)}" ${id === v ? "selected" : ""}>${esc((this._hass.states[id].attributes || {}).friendly_name || id)} · ${esc(id)}</option>`)
        .join("")}</select>`;
    } else {
      const type = conf.type === "password" || f.secret ? "password" : conf.type === "url" ? "url" : "text";
      if (f.secret) {
        const edited = dirty;
        const ph = f.is_set ? "•••••••• stored — type to replace" : "not set";
        control = `<div class="row"><input type="password" autocomplete="new-password" ${data} data-k="text" value="${edited ? esc(v) : ""}" placeholder="${esc(edited && v === "" ? "will be removed" : ph)}">
          ${f.is_set ? `<button class="btn ghost sm" data-act="clearsecret" ${data}>Clear</button>` : ""}</div>`;
      } else {
        const isModel = sid === "provider" && f.name === "model";
        control = `<div class="combo"><div class="row"><input type="${type}" ${data} data-k="text" value="${esc(v ?? "")}" spellcheck="false">
          ${isModel ? `<button class="btn ghost sm" data-act="modellist" title="Models on the server">${this._modelOpen ? "▴" : "▾"}</button>` : ""}</div>
          ${isModel && this._modelOpen ? this._modelMenu(v) : ""}</div>`;
      }
    }
    return `<div class="field ${stack ? "stack" : ""} ${key === "boolean" ? "bool" : ""} ${dirty ? "dirty" : ""}"><label>${esc(f.label)}</label>${control}
      ${f.help ? `<div class="help">${esc(f.help)}</div>` : ""}${err ? `<div class="err">✕ ${esc(err)}</div>` : ""}</div>`;
  }

  _modelMenu(current) {
    if (this._modelsError) return `<div class="menu-list"><div class="small" style="color:var(--red);padding:8px">${esc(this._modelsError)}</div></div>`;
    if (!this._models) return `<div class="menu-list"><div class="small" style="padding:8px">Reading models…</div></div>`;
    return `<div class="menu-list">${this._models
      .map((m) => `<button data-act="pickmodel" data-m="${esc(m)}" class="${m === current ? "on" : ""}">${esc(m)}</button>`)
      .join("")}<button data-act="scan" class="rescan">↻ Read the list again</button></div>`;
  }

  _devicePicker(sid, f, v, data) {
    const chosen = new Set(Array.isArray(v) ? v : []);
    const devices = this._hass.devices || {};
    const entities = this._hass.entities || {};
    // Satellites first: these settings are about voice devices.
    const sat = new Set(Object.values(entities).filter((e) => e.entity_id.startsWith("assist_satellite.")).map((e) => e.device_id));
    const ids = Object.keys(devices)
      .filter((id) => sat.has(id) || chosen.has(id))
      .sort((a, b) => String(devices[a].name_by_user || devices[a].name).localeCompare(String(devices[b].name_by_user || devices[b].name)));
    if (!ids.length) return `<div class="small">No voice satellites found.</div>`;
    return `<div class="chips">${ids
      .map((id) => {
        const d = devices[id] || {};
        const on = chosen.has(id);
        return `<button class="chip dev ${on ? "" : "off"}" data-act="devtoggle" ${data} data-id="${esc(id)}">${on ? "◉" : "○"} ${esc(d.name_by_user || d.name || id)}</button>`;
      })
      .join("")}</div>`;
  }

  _viewModel() {
    const scan = "";
    return `<div class="cols">
      ${this._sectionCard("provider", { title: "Model server", sub: "Any OpenAI-compatible server: Ollama, LM Studio, vLLM, OpenRouter… The model must support tools.", after: scan })}
      ${ADVANCED_GROUPS.model.map(([title, names]) => this._sectionCard("advanced", { title, only: names, sub: "" })).join("")}
    </div>`;
  }

  _viewDialog() {
    const known = new Set(Object.values(ADVANCED_GROUPS).flat().flatMap((g) => g[1]));
    const sec = this._settings.advanced;
    const other = sec ? sec.fields.map((f) => f.name).filter((n) => !known.has(n)) : [];
    return `<div class="cols">
      ${ADVANCED_GROUPS.dialog.map(([title, names]) => this._sectionCard("advanced", { title, only: names, sub: "", wide: names.includes("system_prompt") })).join("")}
      ${other.length ? this._sectionCard("advanced", { title: "Other", only: other, sub: "" }) : ""}
    </div>`;
  }

  _viewVoice() {
    const v = this._voice;
    const settings = this._sectionCard("voice", { title: "Recognition" });
    if (!v) return `<div class="cols">${settings}<div class="card"><div class="loading">Scanning voices<i></i></div></div></div>`;
    const sats = v.satellites || [];
    const people = (v.people || [])
      .map((p) => {
        const enrolling = this._enrollFor === p.id;
        return `<div class="person">
          <div class="head"><div class="avatar ${p.samples ? "" : "off"}">${esc((p.name || "?").trim().charAt(0).toUpperCase())}</div>
            <div class="grow"><div style="font-weight:700">${esc(p.name)}</div>
            <div class="small">${p.samples ? `${p.samples} samples` : "not enrolled"}</div></div>
            ${p.samples ? `<div class="wave">${[0, 1, 2, 3, 4].map((i) => `<i style="animation-delay:${i * 0.12}s"></i>`).join("")}</div>` : ""}</div>
          ${p.rooms.length ? `<div class="bars">${p.rooms.map((r) => `<div class="bar" style="grid-template-columns:1fr 70px 26px"><em>${esc(r.room)}</em><div class="track"><div class="fill" style="width:${Math.min(100, r.samples * 10)}%"></div></div><span>${r.samples}</span></div>`).join("")}</div>` : ""}
          ${enrolling ? `<select data-act="enrollsat">${sats.map((s) => `<option value="${esc(s.device_id)}">${esc(s.name)}</option>`).join("")}</select>
            <div class="row"><button class="btn sm" data-act="enrollgo" data-p="${esc(p.id)}">Start on satellite</button><button class="btn ghost sm" data-act="enrollcancel">Cancel</button></div>`
          : `<div class="row">${v.enabled && sats.length ? `<button class="btn sm" data-act="enroll" data-p="${esc(p.id)}">${p.samples ? "Add samples" : "Enroll"}</button>` : ""}
            ${p.samples ? `<button class="btn danger sm" data-act="voicedel" data-p="${esc(p.id)}" data-n="${esc(p.name)}">Delete voice</button>` : ""}</div>`}
        </div>`;
      })
      .join("");
    const notes = [];
    if (!v.enabled) notes.push("Voice recognition is off. Turn it on above and save to enroll voices.");
    if (v.error) notes.push(`Model unavailable: ${v.error}`);
    if ((v.pipelines_pending || []).length) notes.push(`Pipelines not yet using voice recognition: ${v.pipelines_pending.join(", ")}`);
    if ((v.rooms_without_samples || []).length) notes.push(`A few samples on ${v.rooms_without_samples.join(", ")} improve recognition there.`);
    return `<div class="cols">${settings}
      <div class="card"><h2>Signal</h2><div class="sub">Enrolling makes the chosen satellite ask a few questions; answer them in your normal voice.</div>
        ${notes.length ? `<div class="list">${notes.map((n) => `<div class="item"><span class="chip a">!</span><div class="grow small">${esc(n)}</div></div>`).join("")}</div>` : `<div class="chips"><span class="chip g">● pipelines linked</span><span class="chip v">${sats.length} satellites</span></div>`}
      </div>
      <div class="card wide"><h2>People</h2><div class="people">${people || `<div class="empty">No person entities in Home Assistant</div>`}</div></div>
    </div>`;
  }

  _viewMemory() {
    const settings = this._sectionCard("memory", { title: "Memory" });
    const m = this._memory;
    if (!m) return `<div class="cols">${settings}<div class="card"><div class="loading">Reading memory<i></i></div></div></div>`;
    const fe = this._factEdit;
    const facts = (m.facts || [])
      .filter((g) => g.facts.length)
      .map((g) => `<div class="owner">${esc(g.owner)}</div><div class="list">${g.facts
        .map((f, i) => fe && fe.key === g.key && fe.i === i
          ? `<div class="item"><div class="grow"><textarea class="fact-edit" id="factedit" rows="2">${esc(f)}</textarea>
              <div class="row" style="margin-top:6px"><button class="btn sm" data-act="factsave" data-key="${esc(g.key)}" data-i="${i}" data-fact="${esc(f)}">Save</button><button class="btn ghost sm" data-act="factcancel">Cancel</button></div></div></div>`
          : `<div class="item"><div class="grow">${esc(f)}</div>
              <button class="x edit" title="Edit" data-act="factedit" data-key="${esc(g.key)}" data-i="${i}">✎</button>
              <button class="x" title="Forget" data-act="forget" data-key="${esc(g.key)}" data-i="${i}" data-fact="${esc(f)}">×</button></div>`)
        .join("")}</div>`)
      .join("");
    const owners = m.owners || [];
    const add = owners.length
      ? `<div class="addfact"><select id="factowner">${owners.map((o) => `<option value="${esc(o.key)}">${esc(o.owner)}</option>`).join("")}</select>
          <input id="factnew" placeholder="New fact, e.g. prefers tea over coffee"><button class="btn sm" data-act="factadd">Add</button></div>`
      : "";
    const de = this._diaryEdit;
    const diary = (m.diary || [])
      .map((n) => de === n.id
        ? `<div class="item"><div class="grow"><div class="t">${esc(when(n.time))} · ${esc(n.owner)}</div><textarea class="fact-edit" id="diaryedit" rows="3">${esc(n.text)}</textarea>
            <div class="row" style="margin-top:6px"><button class="btn sm" data-act="diarysave" data-id="${esc(n.id)}" data-text="${esc(n.text)}">Save</button><button class="btn ghost sm" data-act="diarycancel">Cancel</button></div></div></div>`
        : `<div class="item"><div class="grow"><div class="t">${esc(when(n.time))} · ${esc(n.owner)}</div>${esc(n.text)}</div>
            <button class="x edit" title="Edit" data-act="diaryedit" data-id="${esc(n.id)}">✎</button>
            <button class="x" title="Delete" data-act="diarydel" data-id="${esc(n.id)}">×</button></div>`)
      .join("");
    const jobs = (m.scheduled || [])
      .map((j) => `<div class="item"><span class="chip ${j.kind === "alarm" ? "m" : j.kind === "command" ? "v" : ""}">${esc(j.kind)}</span><div class="grow"><div class="t">${esc(when(j.due))}${j.repeat ? ` · repeats ${esc(j.repeat)}` : ""}${j.device ? ` · ${esc(j.device)}` : ""}${j.person ? ` · ${esc(j.person)}` : ""}</div>${esc(j.text || "")}</div><button class="x" title="Cancel" data-act="jobdel" data-id="${esc(j.id)}">×</button></div>`)
      .join("");
    return `<div class="cols">${settings}
      <div class="card"><h2>Saved facts</h2><div class="sub">What the assistant was asked to remember, per person.${m.memory_on ? "" : " Memory is off: these are kept and come back when it is on."}</div>
        ${facts || `<div class="empty">Nothing saved</div>`}${add}</div>
      <div class="card"><h2>Scheduled</h2><div class="sub">Alarms, reminders and commands for later.</div>
        <div class="list">${jobs || `<div class="empty">Nothing scheduled</div>`}</div></div>
      <div class="card wide"><h2>Diary</h2><div class="sub">Notes about earlier conversations, newest first.</div>
        <div class="list">${diary || `<div class="empty">No notes</div>`}</div></div>
    </div>`;
  }

  _viewTools() {
    const mcp = this._mcp;
    let servers = `<div class="loading">Probing servers<i></i></div>`;
    if (mcp && this._mcpEdit) {
      const rows = this._mcpEdit
        .map((s, pos) => {
          const st = s.status || { state: "new", tools: 0 };
          const stCls = { connected: "g", builtin: "v", connecting: "a", reconnecting: "a", new: "x" }[st.state] || "r";
          const stText = st.state === "builtin" ? "built in" : st.state === "connected" ? `linked · ${st.tools} tool${st.tools === 1 ? "" : "s"}` : st.state;
          const head = `<div class="top"><span class="chip ${s.transport === "stdio" ? "v" : ""}">${esc(s.transport)}</span>
            <div class="grow">${esc(s.transport === "stdio" ? [s.command, ...(s.args || [])].join(" ") : s.url)}</div>
            <span class="chip ${stCls}">${esc(stText)}</span><button class="x" title="Remove" data-act="mcprm" data-pos="${pos}">×</button></div>`;
          const own = s.own_ha ? `<div class="small">Home Assistant's own tools are built in, so this server is not used.</div>` : "";
          if (s.transport === "stdio") {
            return `<div class="server">${head}${own}
              <div class="row"><input data-mcp="${pos}" data-mf="command" value="${esc(s.command)}" placeholder="command"><input data-mcp="${pos}" data-mf="args" value="${esc((s.args || []).join(" "))}" placeholder="arguments"></div>
              ${(s.env || []).map((e, j) => `<div class="envrow"><input data-mcp="${pos}" data-env="${j}" data-ef="name" value="${esc(e.name)}" placeholder="ENV_NAME"><input type="password" autocomplete="new-password" data-mcp="${pos}" data-env="${j}" data-ef="value" value="${esc(e.value || "")}" placeholder="${e.is_set ? "•••• stored" : "value"}"><button class="x" data-act="envrm" data-pos="${pos}" data-j="${j}">×</button></div>`).join("")}
              <div><button class="btn ghost sm" data-act="envadd" data-pos="${pos}">+ Variable</button></div></div>`;
          }
          return `<div class="server">${head}${own}
            <div class="row"><input type="url" data-mcp="${pos}" data-mf="url" value="${esc(s.url)}" placeholder="https://…/mcp"><input type="password" autocomplete="new-password" data-mcp="${pos}" data-mf="token" value="${esc(s.token || "")}" placeholder="${s.token_set ? "•••• token stored" : "bearer token (optional)"}"></div></div>`;
        })
        .join("");
      const presets = (mcp.presets || []).map((p) => `<option value="${esc(p.key)}">${esc(p.label)}</option>`).join("");
      const preset = this._presetPick ? (mcp.presets || []).find((p) => p.key === this._presetPick) : null;
      servers = `<div class="list" style="gap:12px">${rows || `<div class="empty">No MCP servers</div>`}</div>
        <div class="row" style="margin-top:14px;flex-wrap:wrap">
          <select data-act="presetpick" style="flex:1 1 260px"><option value="">+ Add a popular server…</option>${presets}</select>
          <button class="btn ghost sm" data-act="mcpadd" data-t="http">+ HTTP</button>
          <button class="btn ghost sm" data-act="mcpadd" data-t="stdio">+ stdio</button></div>
        ${preset && preset.needs_value ? `<div class="row" style="margin-top:10px"><input id="presetval" placeholder="${esc(preset.placeholder || preset.value_label || "")}"><button class="btn sm" data-act="presetadd">Add</button></div>` : ""}
        ${this._mcpErrors && this._mcpErrors.length ? `<div class="list" style="margin-top:10px">${this._mcpErrors.map((e) => `<div class="item"><span class="chip r">✕</span><div class="grow small">Server ${e.index + 1}: ${esc(e.field)} — ${esc(e.error.replace(/_/g, " "))}</div></div>`).join("")}</div>` : ""}`;
    }
    return `<div class="cols">
      ${this._sectionCard("tools", { title: "Built-in features" })}
      ${this._sectionCard("web_search", { title: "Web search" })}
      <div class="card wide"><h2>MCP servers</h2><div class="sub">${mcp ? `${esc(mcp.builtin_tools)} Home Assistant tools are built in. ` : ""}Extra tool servers the model can use. Changes are saved with the bar at the bottom.</div>${servers}</div>
    </div>`;
  }

  // ── events ─────────────────────────────────────────────────────────────

  _onInput(e, isChange) {
    const t = e.target;
    if (t.dataset.act === "entry" && isChange) {
      this._entryId = t.value;
      this._edits = {};
      this._mcpEdit = null;
      this._voice = this._memory = this._mcp = null;
      this._loadAll();
      return;
    }
    if (t.dataset.act === "presetpick" && isChange) {
      this._presetPick = t.value || null;
      const p = this._mcp && this._mcp.presets.find((x) => x.key === t.value);
      if (p && !p.needs_value) this._addPreset(p.key);
      else this._renderMain();
      return;
    }
    if (t.dataset.mcp !== undefined) {
      const s = this._mcpEdit[+t.dataset.mcp];
      if (t.dataset.env !== undefined) {
        const env = s.env[+t.dataset.env];
        if (t.dataset.ef === "name") {
          if (env.from === undefined && env.is_set) env.from = env.name;
          env.name = t.value;
        } else env.value = t.value;
      } else if (t.dataset.mf === "args") s.args = t.value.split(/\s+/).filter(Boolean);
      else s[t.dataset.mf] = t.value;
      s._dirty = true;
      this._renderSaveBar();
      this._renderNav();
      return;
    }
    const sid = t.dataset.s, name = t.dataset.f, kind = t.dataset.k;
    if (!sid || !name || !kind) return;
    let value;
    if (kind === "bool") value = t.checked;
    else if (kind === "num") {
      value = t.value === "" ? null : Number(t.value);
      // Mirror slider and box.
      t.parentElement.querySelectorAll(`[data-f="${CSS.escape(name)}"]`).forEach((x) => {
        if (x !== t) x.value = t.value;
      });
    } else value = t.value;
    this._setEdit(sid, name, value);
    if (kind === "bool" || isChange) {
      const field = t.closest(".field");
      if (field) field.classList.add("dirty");
    }
  }

  _setEdit(sid, name, value) {
    const orig = this._settings[sid] ? this._settings[sid].values[name] : undefined;
    this._edits[sid] = this._edits[sid] || {};
    const same = JSON.stringify(orig ?? null) === JSON.stringify(value ?? null) || (orig === undefined && value === "");
    const f = this._settings[sid] && this._settings[sid].fields.find((x) => x.name === name);
    if (f && f.secret ? value === "" : same) delete this._edits[sid][name];
    else this._edits[sid][name] = value;
    if (!Object.keys(this._edits[sid]).length) delete this._edits[sid];
    this._renderSaveBar();
    this._renderNav();
  }

  async _onClick(e) {
    const t = e.target.closest("[data-act],[data-tab]");
    if (this._modelOpen && !e.target.closest(".combo")) {
      this._modelOpen = false;
      this._renderMain();
    }
    if (!t) return;
    if (t.dataset.tab) {
      this._tab = t.dataset.tab;
      try {
        localStorage.setItem("ai_plugin_panel_tab", this._tab);
      } catch (_e) {
        /* storage blocked */
      }
      this._renderNav();
      this._renderMain();
      this._loadTabData();
      if (this._tab === "core") this._tick();
      return;
    }
    const act = t.dataset.act;
    if (act === "recheck") {
      this._status = await this._ws("ai_plugin/status", { refresh: true }).catch((err) => ({ error: err.message || String(err) }));
      this._renderMain();
      this._renderHeaderStatus();
    } else if (act === "reload") {
      if (!confirm("Reload AI Plugin? A conversation in progress is cut off.")) return;
      try {
        await this._ws("ai_plugin/reload");
        this._toast("Reloaded");
      } catch (err) {
        this._toast(err.message || String(err), true);
      }
      setTimeout(() => this._loadAll(), 1500);
    } else if (act === "menu") {
      this.dispatchEvent(new Event("hass-toggle-menu", { bubbles: true, composed: true }));
    } else if (act === "discard") {
      this._edits = {};
      this._errors = {};
      if (this._mcp) this._mcpEdit = JSON.parse(JSON.stringify(this._mcp.servers));
      this._mcpErrors = null;
      this._render();
    } else if (act === "saveall") {
      await this._saveAll();
    } else if (act === "clearsecret") {
      this._edits[t.dataset.s] = this._edits[t.dataset.s] || {};
      this._edits[t.dataset.s][t.dataset.f] = "";
      this._renderMain();
      this._renderNav();
    } else if (act === "devtoggle") {
      e.preventDefault();
      const cur = this._value(t.dataset.s, t.dataset.f);
      const set = new Set(Array.isArray(cur) ? cur : []);
      if (set.has(t.dataset.id)) set.delete(t.dataset.id);
      else set.add(t.dataset.id);
      this._setEdit(t.dataset.s, t.dataset.f, [...set]);
      this._renderMain();
    } else if (act === "scan") {
      this._models = null;
      this._modelOpen = true;
      this._renderMain();
      await this._scanModels();
    } else if (act === "modellist") {
      this._modelOpen = !this._modelOpen;
      this._renderMain();
      if (this._modelOpen && !this._models) await this._scanModels();
    } else if (act === "pickmodel") {
      this._setEdit("provider", "model", t.dataset.m);
      this._modelOpen = false;
      this._renderMain();
    } else if (act === "enroll") {
      this._enrollFor = t.dataset.p;
      this._renderMain();
    } else if (act === "enrollcancel") {
      this._enrollFor = null;
      this._renderMain();
    } else if (act === "enrollgo") {
      const sat = this.shadowRoot.querySelector('[data-act="enrollsat"]');
      try {
        const r = await this._ws("ai_plugin/voice/enroll", { person: t.dataset.p, satellite: sat.value });
        this._toast(r.mode === "started" ? `${r.satellite} is asking now — answer in your normal voice.` : `Say the wake word at ${r.satellite} to start.`);
        this._enrollFor = null;
      } catch (err) {
        this._toast(err.message || String(err), true);
      }
      this._loadTabData();
    } else if (act === "voicedel") {
      if (!confirm(`Delete the stored voice of ${t.dataset.n}?`)) return;
      await this._ws("ai_plugin/voice/delete", { person: t.dataset.p });
      this._toast("Voice deleted");
      this._loadTabData();
    } else if (act === "forget") {
      if (!confirm(`Forget "${t.dataset.fact}"?`)) return;
      const r = await this._ws("ai_plugin/memory/forget", { key: t.dataset.key, index: +t.dataset.i, fact: t.dataset.fact });
      this._toast(r.deleted ? "Forgotten" : "It changed meanwhile — reloaded", !r.deleted);
      this._loadTabData();
    } else if (act === "factedit") {
      this._factEdit = { key: t.dataset.key, i: +t.dataset.i };
      this._renderMain();
      const ta = this.shadowRoot.getElementById("factedit");
      if (ta) ta.focus();
    } else if (act === "factcancel") {
      this._factEdit = null;
      this._renderMain();
    } else if (act === "factsave") {
      const ta = this.shadowRoot.getElementById("factedit");
      const r = await this._ws("ai_plugin/memory/edit", { key: t.dataset.key, index: +t.dataset.i, fact: t.dataset.fact, new: ta ? ta.value : "" });
      this._toast(r.saved ? "Saved" : "Not saved — empty, a duplicate, or it changed meanwhile", !r.saved);
      if (r.saved) this._factEdit = null;
      this._loadTabData();
    } else if (act === "factadd") {
      const owner = this.shadowRoot.getElementById("factowner");
      const input = this.shadowRoot.getElementById("factnew");
      if (!input || !input.value.trim()) return;
      const r = await this._ws("ai_plugin/memory/edit", { key: owner.value, new: input.value });
      this._toast(r.saved ? "Added" : "Not added — empty or already saved", !r.saved);
      this._loadTabData();
    } else if (act === "diaryedit") {
      this._diaryEdit = t.dataset.id;
      this._renderMain();
      const ta = this.shadowRoot.getElementById("diaryedit");
      if (ta) ta.focus();
    } else if (act === "diarycancel") {
      this._diaryEdit = null;
      this._renderMain();
    } else if (act === "diarysave") {
      const ta = this.shadowRoot.getElementById("diaryedit");
      const r = await this._ws("ai_plugin/diary/edit", { note: t.dataset.id, text: ta ? ta.value : "", expected: t.dataset.text });
      this._toast(r.saved ? "Saved" : "Not saved — empty, or it changed meanwhile", !r.saved);
      if (r.saved) this._diaryEdit = null;
      this._loadTabData();
    } else if (act === "diarydel") {
      if (!confirm("Delete this diary note?")) return;
      await this._ws("ai_plugin/diary/delete", { note: t.dataset.id });
      this._loadTabData();
    } else if (act === "jobdel") {
      if (!confirm("Cancel this scheduled item?")) return;
      await this._ws("ai_plugin/schedule/cancel", { job: t.dataset.id });
      this._loadTabData();
    } else if (act === "mcprm") {
      this._mcpEdit.splice(+t.dataset.pos, 1);
      this._mcpEdit._dirty = true;
      this._renderMain();
      this._renderNav();
    } else if (act === "mcpadd") {
      this._mcpEdit.push(t.dataset.t === "stdio"
        ? { index: null, transport: "stdio", command: "", args: [], env: [], _dirty: true }
        : { index: null, transport: "http", url: "", token_set: false, _dirty: true });
      this._renderMain();
      this._renderNav();
    } else if (act === "presetadd") {
      const val = this.shadowRoot.getElementById("presetval");
      await this._addPreset(this._presetPick, val ? val.value : "");
    } else if (act === "envadd") {
      const s = this._mcpEdit[+t.dataset.pos];
      s.env = s.env || [];
      s.env.push({ name: "", value: "", is_set: false });
      s._dirty = true;
      this._renderMain();
    } else if (act === "envrm") {
      const s = this._mcpEdit[+t.dataset.pos];
      s.env.splice(+t.dataset.j, 1);
      s._dirty = true;
      this._renderMain();
      this._renderNav();
    }
  }

  async _addPreset(key, value) {
    try {
      const server = await this._ws("ai_plugin/mcp/preset", value ? { key, value } : { key });
      server._dirty = true;
      this._mcpEdit.push(server);
      this._presetPick = null;
    } catch (err) {
      this._toast(err.message || String(err), true);
    }
    this._renderMain();
    this._renderNav();
  }

  _mcpDirty() {
    if (!this._mcpEdit || !this._mcp) return false;
    if (this._mcpEdit._dirty || this._mcpEdit.length !== this._mcp.servers.length) return true;
    return this._mcpEdit.some((s) => s._dirty);
  }

  async _scanModels() {
    const base = this._value("provider", "base_url");
    const key = this._edits.provider && this._edits.provider.api_key;
    try {
      const r = await this._ws("ai_plugin/models", key ? { base_url: base, api_key: key } : { base_url: base });
      this._models = r.models;
      this._modelsError = r.error || (r.models.length ? null : "The server lists no models.");
    } catch (err) {
      this._modelsError = err.message || String(err);
    }
    this._renderMain();
  }

  async _saveAll() {
    this._busy = true;
    this._renderSaveBar();
    let ok = true;
    // Provider first: the advanced checks ask the (new) model what it can do.
    const order = ["provider", "web_search", "memory", "tools", "advanced", "voice"];
    const sections = Object.keys(this._edits).sort((a, b) => order.indexOf(a) - order.indexOf(b));
    for (const sid of sections) {
      const values = { ...this._settings[sid].values, ...this._edits[sid] };
      for (const f of this._settings[sid].fields) {
        if (f.secret && !(f.name in this._edits[sid])) delete values[f.name];
      }
      try {
        const r = await this._ws("ai_plugin/settings/save", { section: sid, values, language: this._hass.language });
        if (r.saved) {
          delete this._edits[sid];
          delete this._errors[sid];
        } else {
          ok = false;
          this._errors[sid] = r.errors;
          const first = Object.entries(r.errors)[0];
          this._toast(`${this._settings[sid].title || sid}: ${first ? `${this._fieldLabel(sid, first[0])} — ${first[1]}` : "not saved"}`, true);
        }
      } catch (err) {
        ok = false;
        this._toast(err.message || String(err), true);
      }
    }
    if (this._mcpDirty()) {
      const servers = this._mcpEdit.map((s) => {
        const out = { index: s.index, transport: s.transport };
        if (s.transport === "stdio") {
          out.command = s.command;
          out.args = s.args || [];
          out.env = (s.env || []).map((e) => {
            const v = { name: e.name };
            if (e.value) v.value = e.value;
            if (e.from) v.from = e.from;
            return v;
          });
        } else {
          out.url = s.url;
          if (s.token) out.token = s.token;
        }
        return out;
      });
      try {
        const r = await this._ws("ai_plugin/mcp/save", { servers });
        if (r.saved) {
          this._mcpErrors = null;
          this._mcpEdit = null;
        } else {
          ok = false;
          this._mcpErrors = r.errors;
          this._toast("MCP servers not saved", true);
        }
      } catch (err) {
        ok = false;
        this._toast(err.message || String(err), true);
      }
    }
    this._busy = false;
    if (ok) this._toast("Saved — the assistant restarts its link");
    // The entry reloads after a save; give it a moment, then read back.
    setTimeout(async () => {
      await this._loadAll();
    }, 1800);
    this._render();
  }

  _toast(text, bad = false) {
    const el = this.shadowRoot.getElementById("toast");
    el.innerHTML = `<div class="toast ${bad ? "bad" : ""}">${esc(text)}</div>`;
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => (el.innerHTML = ""), bad ? 6000 : 3200);
  }
}

if (!customElements.get("ai-plugin-panel")) customElements.define("ai-plugin-panel", AIPluginPanel);
