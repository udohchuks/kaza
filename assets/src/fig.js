// Shared drawing helpers for the Kaza figures.
// Each figure page calls draw(width, height, parts) with an array of SVG strings.

const STYLE = {
  white: { fill: "#ffffff", stroke: "#1f1f1f", text: "#1f1f1f", sub: "#4a4a4a" },
  frozen: { fill: "#dcdcdc", stroke: "#1f1f1f", text: "#1f1f1f", sub: "#4a4a4a" },
  purple: { fill: "#c9b6e8", stroke: "none", text: "#2e1f4f", sub: "#4b3a73" },
  teal: { fill: "#80cbc4", stroke: "none", text: "#103d39", sub: "#1f5c56" },
  coral: { fill: "#f4978e", stroke: "none", text: "#4a1510", sub: "#6e241c" },
  green: { fill: "#7cb342", stroke: "none", text: "#13280a", sub: "#13280a" },
  blue: { fill: "#4fc3f7", stroke: "none", text: "#062b3d", sub: "#062b3d" },
  amber: { fill: "#ffb74d", stroke: "none", text: "#3d2400", sub: "#5c3600" },
  newlayer: { fill: "#fce1de", stroke: "#1f1f1f", dash: "4 3", text: "#4a1510", sub: "#6e241c" },
};

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Rounded box with a title and an optional subtitle, both centred.
function box(x, y, w, h, title, sub, style = "white", opts = {}) {
  const s = STYLE[style];
  const r = opts.r ?? 8;
  const dash = s.dash ? ` stroke-dasharray="${s.dash}"` : "";
  const sw = s.stroke === "none" ? 0 : 1.3;
  const fs = opts.fs ?? 15;
  const cx = x + w / 2;
  let out = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${s.fill}" stroke="${s.stroke}" stroke-width="${sw}"${dash}/>`;
  if (sub) {
    out += `<text x="${cx}" y="${y + h / 2 - 8}" class="bt" fill="${s.text}" font-size="${fs}">${esc(title)}</text>`;
    out += `<text x="${cx}" y="${y + h / 2 + 11}" class="bs" fill="${s.sub}">${esc(sub)}</text>`;
  } else {
    out += `<text x="${cx}" y="${y + h / 2}" class="bt" fill="${s.text}" font-size="${fs}">${esc(title)}</text>`;
  }
  return out;
}

// Grey container with large rounded corners.
function container(x, y, w, h, fill = "#dcdcdc", r = 40, dashed = false) {
  const d = dashed ? ` stroke-dasharray="2 5" stroke-linecap="round" stroke-width="2.2"` : ` stroke-width="1.5"`;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="#1f1f1f"${d}/>`;
}

// Solid arrow along a list of points; the head sits on the last point.
function arrow(points, opts = {}) {
  const d = points.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" ");
  const dash = opts.dash ? ` stroke-dasharray="${opts.dash}"` : "";
  const color = opts.color ?? "#1f1f1f";
  const head = opts.noHead ? "" : ` marker-end="url(#ah)"`;
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.4"${dash}${head}/>`;
}

// Dotted leader line (no head).
function leader(points) {
  const d = points.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" ");
  return `<path d="${d}" fill="none" stroke="#1f1f1f" stroke-width="1.8" stroke-dasharray="1.5 4" stroke-linecap="round"/>`;
}

// Side label: first line bold black, later lines bold grey. Each entry may be
// a string (grey) or [bold, grey] for a line that mixes both.
function label(x, y, lines, opts = {}) {
  const size = opts.size ?? 19;
  const lh = opts.lh ?? size * 1.3;
  const anchor = opts.anchor ?? "start";
  let out = `<text x="${x}" y="${y}" font-size="${size}" font-weight="600" text-anchor="${anchor}">`;
  lines.forEach((ln, i) => {
    const dy = i === 0 ? 0 : lh;
    if (Array.isArray(ln)) {
      out += `<tspan x="${x}" dy="${dy}" fill="#111">${esc(ln[0])}</tspan><tspan fill="#666"> ${esc(ln[1])}</tspan>`;
    } else {
      out += `<tspan x="${x}" dy="${dy}" fill="${i === 0 && !opts.allGrey ? "#111" : "#666"}">${esc(ln)}</tspan>`;
    }
  });
  return out + "</text>";
}

function text(x, y, s, opts = {}) {
  const cls = opts.mono ? "mono" : "";
  return `<text x="${x}" y="${y}" class="${cls}" font-size="${opts.size ?? 14}" font-weight="${opts.weight ?? 400}" fill="${opts.fill ?? "#1f1f1f"}" text-anchor="${opts.anchor ?? "middle"}">${esc(s)}</text>`;
}

function circleX(cx, cy) {
  return `<circle cx="${cx}" cy="${cy}" r="11" fill="#fff" stroke="#1f1f1f" stroke-width="1.3"/><text x="${cx}" y="${cy}" class="bt" font-size="14" fill="#1f1f1f">×</text>`;
}

function header(title, subtitle) {
  return `<text x="40" y="88" font-size="62" font-weight="700" fill="#555" letter-spacing="-1.5">${esc(title)}</text>` +
    `<text x="42" y="124" font-size="17" fill="#555">${esc(subtitle)}</text>`;
}

function draw(width, height, parts) {
  const defs = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1.5 L9 5 L0 8.5 z" fill="context-stroke"/></marker></defs>`;
  document.body.style.width = width + "px";
  document.body.style.height = height + "px";
  document.getElementById("fig").innerHTML =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${defs}${parts.join("")}</svg>`;
}
