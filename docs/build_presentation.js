// PMG Intelligence — Capstone Presentation
// Run: node build_presentation.js
"use strict";
const pptxgen = require("pptxgenjs");

const OUTPUT = "/Users/alexkatzighera/Documents/Capstone Project/docs/PMG_Intelligence_Presentation.pptx";

// ── Palette (NO # prefix) ─────────────────────────────────────────────────────
const NAVY  = "1E293B";   // slide background
const CARD  = "243B52";   // card / panel background
const CARD2 = "2D4A63";   // lighter card variant
const TEAL  = "38BDF8";   // headings / accents
const AMBER = "F59E0B";   // stats / callouts
const WHITE = "FFFFFF";
const MUTED = "94A3B8";   // secondary text
const DARK  = "0F172A";   // deep navy for bottom bars

// ── Helpers ───────────────────────────────────────────────────────────────────
// Fresh objects every call — pptxgenjs mutates in place
const mkShadow = () => ({ type: "outer", blur: 10, offset: 3, angle: 135, color: "000000", opacity: 0.30 });

function cardShape(slide, x, y, w, h, fill = CARD) {
  slide.addShape("rect", {
    x, y, w, h,
    fill: { color: fill },
    line: { color: "2D4A63", width: 1 },
    shadow: mkShadow(),
  });
}

// Add a label + description inside a card (no icon — guaranteed rendering)
function featureCard(slide, x, y, w, h, label, desc) {
  cardShape(slide, x, y, w, h);
  // Teal top accent bar
  slide.addShape("rect", { x, y, w, h: 0.06, fill: { color: TEAL }, line: { color: TEAL } });
  slide.addText(label, {
    x: x + 0.15, y: y + 0.08, w: w - 0.3, h: 0.38,
    fontSize: 12, fontFace: "Calibri", bold: true, color: TEAL,
    align: "left", valign: "top", margin: 0,
  });
  slide.addText(desc, {
    x: x + 0.15, y: y + 0.48, w: w - 0.3, h: h - 0.58,
    fontSize: 10, fontFace: "Calibri", color: WHITE,
    align: "left", valign: "top", margin: 0,
  });
}

function slideTitle(slide, text) {
  slide.addText(text, {
    x: 0.4, y: 0.18, w: 9.2, h: 0.55,
    fontSize: 26, fontFace: "Calibri", bold: true,
    color: TEAL, align: "left", valign: "middle", margin: 0,
  });
  // Underline accent bar
  slide.addShape("rect", { x: 0.4, y: 0.76, w: 9.2, h: 0.04, fill: { color: CARD2 }, line: { color: CARD2 } });
}

// ── Presentation ──────────────────────────────────────────────────────────────
const pres = new pptxgen();
pres.layout   = "LAYOUT_16x9";   // 10" × 5.625"
pres.title    = "PMG Intelligence";
pres.author   = "Alex Katzighera";
pres.subject  = "Capstone Project 2026";

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Left teal accent bar
  s.addShape("rect", { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: TEAL }, line: { color: TEAL } });

  // Bottom dark panel
  s.addShape("rect", { x: 0, y: 4.4, w: 10, h: 1.225, fill: { color: DARK }, line: { color: DARK } });

  // Teal bottom accent line
  s.addShape("rect", { x: 0, y: 4.38, w: 10, h: 0.05, fill: { color: TEAL }, line: { color: TEAL } });

  // Title
  s.addText("PMG Intelligence", {
    x: 0.5, y: 0.75, w: 9.2, h: 1.4,
    fontSize: 56, fontFace: "Calibri", bold: true,
    color: WHITE, align: "left", valign: "middle", margin: 0,
  });

  // Subtitle
  s.addText("A Conversational Research Data Dictionary Assistant", {
    x: 0.5, y: 2.2, w: 9, h: 0.65,
    fontSize: 20, fontFace: "Calibri",
    color: TEAL, align: "left", valign: "middle", margin: 0,
  });

  // Divider
  s.addShape("rect", { x: 0.5, y: 3.0, w: 4.5, h: 0.04, fill: { color: TEAL }, line: { color: TEAL } });

  // Tagline
  s.addText("Research answers. Instantly. With confidence.", {
    x: 0.5, y: 3.15, w: 9, h: 0.45,
    fontSize: 13, fontFace: "Calibri", italic: true,
    color: MUTED, align: "left", valign: "middle", margin: 0,
  });

  // Credits in bottom bar
  s.addText("Capstone Project  |  2026  |  Alex Katzighera", {
    x: 0.5, y: 4.5, w: 9, h: 0.45,
    fontSize: 13, fontFace: "Calibri",
    color: MUTED, align: "left", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — THE PROBLEM
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "The Research Lookup Problem");

  const boxes = [
    { icon: "01", label: "Hours Wasted",        desc: "Research teams manually scan spreadsheets to find variable definitions and coded response options." },
    { icon: "02", label: "No Cross-Wave View",  desc: "Comparing questions across survey years requires opening multiple files with no easy search." },
    { icon: "03", label: "Locked in Excel",     desc: "Data dictionaries are hard to search, share, or query at scale — knowledge stays siloed." },
  ];

  const bw = 2.93, bh = 2.9, gap = 0.2, startX = 0.4, startY = 0.92;

  boxes.forEach((b, i) => {
    const x = startX + i * (bw + gap);
    cardShape(s, x, startY, bw, bh);

    // Amber number badge
    s.addShape("rect", { x, y: startY, w: bw, h: 0.08, fill: { color: AMBER }, line: { color: AMBER } });
    s.addText(b.icon, {
      x: x + 0.15, y: startY + 0.14, w: 0.55, h: 0.55,
      fontSize: 24, fontFace: "Calibri", bold: true, color: AMBER,
      align: "center", valign: "middle", margin: 0,
    });

    s.addText(b.label, {
      x: x + 0.15, y: startY + 0.75, w: bw - 0.3, h: 0.42,
      fontSize: 15, fontFace: "Calibri", bold: true, color: WHITE,
      align: "left", valign: "middle", margin: 0,
    });
    s.addText(b.desc, {
      x: x + 0.15, y: startY + 1.2, w: bw - 0.3, h: 1.5,
      fontSize: 11, fontFace: "Calibri", color: MUTED,
      align: "left", valign: "top", margin: 0,
    });
  });

  // Bottom tagline
  s.addText("\u201cAnswers buried in files. Time wasted on manual lookups.\u201d", {
    x: 0.4, y: 4.95, w: 9.2, h: 0.45,
    fontSize: 13, fontFace: "Calibri", italic: true, bold: true,
    color: AMBER, align: "center", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — WHAT IS PMG INTELLIGENCE?
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "What Is PMG Intelligence?");

  // Definition box
  s.addShape("rect", {
    x: 0.4, y: 0.95, w: 9.2, h: 1.5,
    fill: { color: CARD }, line: { color: TEAL, width: 2 },
    shadow: mkShadow(),
  });
  s.addShape("rect", { x: 0.4, y: 0.95, w: 0.1, h: 1.5, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText(
    "A natural language chatbot grounded in your actual survey data dictionaries\n\u2014 delivering cited, auditable answers in seconds.",
    {
      x: 0.65, y: 0.95, w: 8.8, h: 1.5,
      fontSize: 16, fontFace: "Calibri", color: WHITE,
      align: "left", valign: "middle", margin: 10,
    }
  );

  // Stat tiles
  const stats = [
    { num: "6",     lbl: "Topic\nCategories" },
    { num: "3",     lbl: "User\nRoles" },
    { num: "100%",  lbl: "Grounded\nAnswers" },
    { num: "<2s",   lbl: "Typical\nResponse" },
  ];
  const tw = 2.05, th = 2.0, tgap = 0.2, tStartX = 0.4, tY = 2.7;

  stats.forEach((st, i) => {
    const x = tStartX + i * (tw + tgap);
    cardShape(s, x, tY, tw, th, CARD2);
    s.addShape("rect", { x, y: tY, w: tw, h: 0.08, fill: { color: AMBER }, line: { color: AMBER } });
    s.addText(st.num, {
      x: x + 0.1, y: tY + 0.2, w: tw - 0.2, h: 0.85,
      fontSize: 38, fontFace: "Calibri", bold: true,
      color: AMBER, align: "center", valign: "middle", margin: 0,
    });
    s.addText(st.lbl, {
      x: x + 0.1, y: tY + 1.1, w: tw - 0.2, h: 0.7,
      fontSize: 11, fontFace: "Calibri", color: MUTED,
      align: "center", valign: "top", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — BY THE NUMBERS
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "By the Numbers");

  const stats = [
    { num: "30+",  lbl: "Features\nShipped" },
    { num: "92",   lbl: "Unit Tests\nPassing" },
    { num: "3",    lbl: "Routing\nModes" },
    { num: "30",   lbl: "Req / Min\nRate Limit" },
    { num: "EN/FR", lbl: "Multilingual\nSupport" },
    { num: "4",    lbl: "Export\nFormats" },
  ];

  // 3-wide × 2-tall stat grid
  const SW = 2.93, SH = 1.95, SGAP = 0.2, SX0 = 0.4, SY0 = 0.95;
  stats.forEach((st, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = SX0 + col * (SW + SGAP);
    const y = SY0 + row * (SH + SGAP);
    cardShape(s, x, y, SW, SH, CARD2);
    s.addShape("rect", { x, y, w: SW, h: 0.08, fill: { color: TEAL }, line: { color: TEAL } });
    s.addText(st.num, {
      x: x + 0.1, y: y + 0.15, w: SW - 0.2, h: 1.0,
      fontSize: 40, fontFace: "Calibri", bold: true,
      color: AMBER, align: "center", valign: "middle", margin: 0,
    });
    s.addText(st.lbl, {
      x: x + 0.1, y: y + 1.25, w: SW - 0.2, h: 0.58,
      fontSize: 11, fontFace: "Calibri", color: MUTED,
      align: "center", valign: "top", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — KEYSTONE FEATURES (2×3 grid)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "What You Can Do");

  const features = [
    { label: "Natural Language Search",    desc: "Ask questions in plain English about any survey variable — no syntax needed." },
    { label: "Variable ID Lookup",         desc: "Instantly fetch any PMG question by ID (e.g. PMG22_WAI_q5a) with full metadata." },
    { label: "Value Label Retrieval",      desc: "Get coded response options displayed inline — no more hunting through Excel columns." },
    { label: "Multi-Survey Topic Scan",    desc: "Find all questions about GICs, credit unions, or robo-advisors across all survey waves." },
    { label: "Session Consistency",        desc: "Repeat questions return the same answer — no drift or contradictions within a session." },
    { label: "Reasoning Transparency",     desc: "Every answer cites the exact record it used (e.g. [PMG22_WAI_q5a]) with key details." },
  ];

  const cw = 2.93, ch = 1.65, gap = 0.2;
  const startX = 0.4, startY = 0.95;

  features.forEach((f, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = startX + col * (cw + gap);
    const y = startY + row * (ch + gap);
    featureCard(s, x, y, cw, ch, f.label, f.desc);
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — ADVANCED INTELLIGENCE (second feature set)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "Advanced Intelligence Features");

  const features = [
    { label: "Confidence-Tier Routing",    desc: "3 tiers: <0.25 deterministic, <0.35 shows warning, above threshold full LLM synthesis. No hallucinated answers." },
    { label: "Follow-Up Suggestion Chips", desc: "Clickable next-question chips auto-generated after every answer — guides analysts to deeper insights." },
    { label: "Feedback System",            desc: "Thumbs up/down + notes logged by trace ID. Every piece of feedback tied to the exact query and records used." },
    { label: "One-Click Exports",          desc: "Download any answer as .txt, .json, .md, or .xlsx instantly — ready for reports and presentations." },
    { label: "SharePoint Connector",       desc: "Direct integration with SharePoint for uploading new survey dictionaries without leaving the browser." },
    { label: "Multilingual Support",       desc: "Queries accepted in English and French (EN/FR). Auto-detected and translated before retrieval." },
  ];

  const cw = 2.93, ch = 1.65, gap = 0.2;
  const startX = 0.4, startY = 0.95;

  features.forEach((f, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = startX + col * (cw + gap);
    const y = startY + row * (ch + gap);
    featureCard(s, x, y, cw, ch, f.label, f.desc);
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — WORKFLOW (Before / After)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "From Spreadsheet to Conversation");

  const COL1_X = 0.4, COL2_X = 5.2, COL_W = 4.4, COL_H = 3.5, COL_Y = 0.95;

  // BEFORE panel
  cardShape(s, COL1_X, COL_Y, COL_W, COL_H);
  s.addShape("rect", { x: COL1_X, y: COL_Y, w: COL_W, h: 0.08, fill: { color: AMBER }, line: { color: AMBER } });
  s.addText("BEFORE", {
    x: COL1_X + 0.15, y: COL_Y + 0.12, w: COL_W - 0.3, h: 0.38,
    fontSize: 14, fontFace: "Calibri", bold: true, color: AMBER,
    align: "left", valign: "middle", margin: 0, charSpacing: 3,
  });

  const beforeSteps = [
    "Research question arises",
    "Email analyst team",
    "Wait for response",
    "Open Excel dictionary",
    "Manual search through rows",
    "Copy result to presentation",
  ];
  beforeSteps.forEach((step, i) => {
    const sy = COL_Y + 0.62 + i * 0.45;
    s.addShape("rect", { x: COL1_X + 0.2, y: sy + 0.05, w: 0.2, h: 0.2, fill: { color: "5A3800" }, line: { color: "5A3800" } });
    s.addText(step, {
      x: COL1_X + 0.52, y: sy, w: COL_W - 0.72, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: MUTED,
      align: "left", valign: "middle", margin: 0,
    });
    if (i < beforeSteps.length - 1) {
      s.addShape("LINE", {
        x: COL1_X + 0.29, y: sy + 0.25, w: 0, h: 0.22,
        line: { color: "5A3800", width: 1.5 },
      });
    }
  });

  // AFTER panel
  cardShape(s, COL2_X, COL_Y, COL_W, COL_H);
  s.addShape("rect", { x: COL2_X, y: COL_Y, w: COL_W, h: 0.08, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText("AFTER", {
    x: COL2_X + 0.15, y: COL_Y + 0.12, w: COL_W - 0.3, h: 0.38,
    fontSize: 14, fontFace: "Calibri", bold: true, color: TEAL,
    align: "left", valign: "middle", margin: 0, charSpacing: 3,
  });

  const afterSteps = [
    "Research question arises",
    "Ask PMG Intelligence",
    "Cited answer with source record",
  ];
  afterSteps.forEach((step, i) => {
    const sy = COL_Y + 0.62 + i * 0.85;
    s.addShape("rect", { x: COL2_X + 0.2, y: sy + 0.05, w: 0.25, h: 0.25, fill: { color: TEAL }, line: { color: TEAL } });
    s.addText(step, {
      x: COL2_X + 0.58, y: sy, w: COL_W - 0.78, h: 0.4,
      fontSize: 13, fontFace: "Calibri", bold: (i === 2), color: i === 2 ? TEAL : WHITE,
      align: "left", valign: "middle", margin: 0,
    });
    if (i < afterSteps.length - 1) {
      s.addShape("LINE", {
        x: COL2_X + 0.32, y: sy + 0.30, w: 0, h: 0.55,
        line: { color: TEAL, width: 2 },
      });
    }
  });

  // Bottom stat
  s.addShape("rect", { x: 0.4, y: 4.65, w: 9.2, h: 0.06, fill: { color: CARD2 }, line: { color: CARD2 } });
  s.addText("What took 30 minutes now takes 30 seconds.", {
    x: 0.4, y: 4.75, w: 9.2, h: 0.55,
    fontSize: 17, fontFace: "Calibri", bold: true, italic: true,
    color: AMBER, align: "center", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — DATA COVERAGE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "What\u2019s Covered");

  // LEFT — Survey waves
  const LX = 0.4, LY = 0.95, LW = 3.4, LH = 4.3;
  cardShape(s, LX, LY, LW, LH);
  s.addShape("rect", { x: LX, y: LY, w: LW, h: 0.08, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText("PMG Survey Waves", {
    x: LX + 0.15, y: LY + 0.15, w: LW - 0.3, h: 0.45,
    fontSize: 14, fontFace: "Calibri", bold: true, color: TEAL,
    align: "left", valign: "middle", margin: 0,
  });

  const waves = ["PMG17_GAM", "PMG18_ROB", "PMG20_GAM", "PMG22_WAI", "+ additional uploads"];
  waves.forEach((w, i) => {
    s.addText(w, {
      x: LX + 0.25, y: LY + 0.75 + i * 0.52, w: LW - 0.4, h: 0.42,
      fontSize: 13, fontFace: "Calibri",
      color: i === 4 ? MUTED : WHITE, italic: i === 4,
      align: "left", valign: "middle", margin: 0,
      bullet: i < 4 ? true : false,
    });
  });

  s.addText("Upload additional surveys via\nAdmin Panel \u2014 index rebuilds automatically.", {
    x: LX + 0.15, y: LY + 3.55, w: LW - 0.3, h: 0.6,
    fontSize: 10, fontFace: "Calibri", italic: true, color: MUTED,
    align: "left", valign: "top", margin: 0,
  });

  // RIGHT — Topic taxonomy tiles
  const topics = [
    "Demographics",
    "Financial Planning",
    "Trust & Sentiment",
    "Digital Behavior",
    "Product Ownership",
    "Provider Relationship",
  ];
  const RX = 4.05, RY = 0.95, TW = 2.8, TH = 1.0, TGAP = 0.2;

  s.addText("Topic Taxonomy", {
    x: RX, y: RY, w: 5.75, h: 0.42,
    fontSize: 14, fontFace: "Calibri", bold: true, color: TEAL,
    align: "left", valign: "middle", margin: 0,
  });

  topics.forEach((t, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = RX + col * (TW + TGAP);
    const y = RY + 0.55 + row * (TH + TGAP);
    s.addShape("rect", {
      x, y, w: TW, h: TH,
      fill: { color: CARD }, line: { color: TEAL, width: 1.5 },
      shadow: mkShadow(),
    });
    s.addText(t, {
      x: x + 0.15, y, w: TW - 0.3, h: TH,
      fontSize: 12, fontFace: "Calibri", bold: true, color: WHITE,
      align: "left", valign: "middle", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — GOVERNANCE & SECURITY
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "Built for Enterprise Confidence");

  const pillars = [
    {
      label: "Role-Based Access",
      desc: "Viewer / Analyst / Admin token tiers with scoped permissions — only the right people see the right data.",
    },
    {
      label: "Full Audit Trail",
      desc: "Every query logged with trace ID, latency, route, and session. Fully replayable for compliance review.",
    },
    {
      label: "PII Redaction",
      desc: "Email addresses and phone numbers stripped at the database layer automatically — before any logging.",
    },
    {
      label: "Consent Lifecycle",
      desc: "Session activation gate, ownership enforcement per token, and automatic 24-hour expiry for idle sessions.",
    },
  ];

  const PW = 4.5, PH = 1.9, PGAP = 0.2, PX0 = 0.4, PY0 = 0.95;

  pillars.forEach((p, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = PX0 + col * (PW + PGAP);
    const y = PY0 + row * (PH + PGAP);

    cardShape(s, x, y, PW, PH);
    s.addShape("rect", { x, y, w: PW, h: 0.08, fill: { color: TEAL }, line: { color: TEAL } });

    // Number badge
    s.addShape("rect", {
      x: x + 0.15, y: y + 0.18, w: 0.42, h: 0.42,
      fill: { color: TEAL }, line: { color: TEAL },
    });
    s.addText(String(i + 1), {
      x: x + 0.15, y: y + 0.18, w: 0.42, h: 0.42,
      fontSize: 14, fontFace: "Calibri", bold: true, color: NAVY,
      align: "center", valign: "middle", margin: 0,
    });

    s.addText(p.label, {
      x: x + 0.7, y: y + 0.18, w: PW - 0.85, h: 0.42,
      fontSize: 14, fontFace: "Calibri", bold: true, color: WHITE,
      align: "left", valign: "middle", margin: 0,
    });
    s.addText(p.desc, {
      x: x + 0.15, y: y + 0.72, w: PW - 0.3, h: 1.05,
      fontSize: 11, fontFace: "Calibri", color: MUTED,
      align: "left", valign: "top", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE — SECURITY HARDENING
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "Security Audit — All Findings Resolved");

  // Intro line
  s.addText("Full third-party security audit completed. Every finding remediated before release.", {
    x: 0.4, y: 0.88, w: 9.2, h: 0.38,
    fontSize: 12, fontFace: "Calibri", italic: true, color: MUTED,
    align: "left", valign: "middle", margin: 0,
  });

  // 4 severity tiles
  const severities = [
    { level: "Critical",  count: "4",  color: "EF4444", fixes: "Jailbreak detection, input sanitization,\nSQL injection guards, auth token hardening" },
    { level: "High",      count: "2",  color: "F97316", fixes: "PII redaction pipeline,\nsession ownership enforcement" },
    { level: "Medium",    count: "5",  color: AMBER,    fixes: "Rate limiting (30 req/min), export validation,\nresponse size caps, logging PII scrub, CORS" },
    { level: "Low",       count: "4",  color: MUTED,    fixes: "Error message verbosity,\ncache key collision prevention, misc hardening" },
  ];

  const TW = 2.1, TH = 3.55, TGAP = 0.25, TX0 = 0.4, TY = 1.38;

  severities.forEach((sv, i) => {
    const x = TX0 + i * (TW + TGAP);
    // Card
    s.addShape("rect", {
      x, y: TY, w: TW, h: TH,
      fill: { color: CARD }, line: { color: sv.color, width: 1.5 },
      shadow: mkShadow(),
    });
    // Top accent bar
    s.addShape("rect", { x, y: TY, w: TW, h: 0.1, fill: { color: sv.color }, line: { color: sv.color } });
    // Big count
    s.addText(sv.count, {
      x: x + 0.1, y: TY + 0.18, w: TW - 0.2, h: 0.85,
      fontSize: 48, fontFace: "Calibri", bold: true,
      color: sv.color, align: "center", valign: "middle", margin: 0,
    });
    // Level label
    s.addText(sv.level, {
      x: x + 0.1, y: TY + 1.1, w: TW - 0.2, h: 0.38,
      fontSize: 13, fontFace: "Calibri", bold: true,
      color: WHITE, align: "center", valign: "middle", margin: 0,
    });
    // Divider
    s.addShape("rect", { x: x + 0.2, y: TY + 1.55, w: TW - 0.4, h: 0.03, fill: { color: CARD2 }, line: { color: CARD2 } });
    // Fix description
    s.addText(sv.fixes, {
      x: x + 0.12, y: TY + 1.65, w: TW - 0.24, h: 1.75,
      fontSize: 10, fontFace: "Calibri", color: MUTED,
      align: "left", valign: "top", margin: 0,
    });
    // Resolved badge
    s.addShape("rect", {
      x: x + 0.15, y: TY + TH - 0.48, w: TW - 0.3, h: 0.32,
      fill: { color: "064E3B" }, line: { color: "10B981", width: 1 },
    });
    s.addText("RESOLVED", {
      x: x + 0.15, y: TY + TH - 0.48, w: TW - 0.3, h: 0.32,
      fontSize: 10, fontFace: "Calibri", bold: true, color: "10B981",
      align: "center", valign: "middle", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE — ROADMAP (5-phase timeline)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "Implementing at Scale");

  const phases = [
    { time: "Week 1\u20132",  label: "Pilot",         desc: "Deploy with 2\u20133 research analysts.\nCollect feedback." },
    { time: "Week 3",        label: "Access Setup",   desc: "Issue Viewer and\nAnalyst tokens to teams." },
    { time: "Week 4",        label: "Data Upload",    desc: "Load all current\nsurvey data dictionaries." },
    { time: "Month 2",       label: "Training",       desc: "Admin training &\ngovernance sign-off." },
    { time: "Month 3+",      label: "Production",     desc: "Cloud/on-prem deploy.\nSSO integration." },
  ];

  const N = phases.length;
  const LINE_Y = 2.55;
  const CIRC_R = 0.38;
  const SPACING = 9.0 / N;
  const CARD_W = 1.65, CARD_H = 1.45;

  // Connecting teal line
  s.addShape("LINE", {
    x: 0.75, y: LINE_Y, w: 8.5, h: 0,
    line: { color: TEAL, width: 2.5 },
  });

  phases.forEach((ph, i) => {
    const cx = 0.75 + i * SPACING + SPACING / 2 - 0.1;

    // Circle background
    s.addShape("ELLIPSE", {
      x: cx - CIRC_R, y: LINE_Y - CIRC_R,
      w: CIRC_R * 2, h: CIRC_R * 2,
      fill: { color: TEAL }, line: { color: TEAL },
    });
    // Number in circle
    s.addText(String(i + 1), {
      x: cx - CIRC_R, y: LINE_Y - CIRC_R,
      w: CIRC_R * 2, h: CIRC_R * 2,
      fontSize: 16, fontFace: "Calibri", bold: true, color: NAVY,
      align: "center", valign: "middle", margin: 0,
    });

    // Time label above
    s.addText(ph.time, {
      x: cx - 0.85, y: LINE_Y - 1.05, w: 1.7, h: 0.38,
      fontSize: 11, fontFace: "Calibri", bold: true, color: AMBER,
      align: "center", valign: "middle", margin: 0,
    });

    // Card below
    const cardX = cx - CARD_W / 2;
    const cardY = LINE_Y + CIRC_R + 0.18;
    cardShape(s, cardX, cardY, CARD_W, CARD_H);
    s.addText(ph.label, {
      x: cardX + 0.1, y: cardY + 0.08, w: CARD_W - 0.2, h: 0.38,
      fontSize: 12, fontFace: "Calibri", bold: true, color: TEAL,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(ph.desc, {
      x: cardX + 0.1, y: cardY + 0.5, w: CARD_W - 0.2, h: CARD_H - 0.6,
      fontSize: 10, fontFace: "Calibri", color: MUTED,
      align: "center", valign: "top", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — LIVE DEMO
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Large title
  s.addText("Let\u2019s See It Live", {
    x: 0.4, y: 0.22, w: 9.2, h: 0.9,
    fontSize: 38, fontFace: "Calibri", bold: true,
    color: TEAL, align: "left", valign: "middle", margin: 0,
  });
  s.addText("Ask PMG Intelligence anything about your survey data", {
    x: 0.4, y: 1.15, w: 9.2, h: 0.45,
    fontSize: 15, fontFace: "Calibri", color: MUTED,
    align: "left", valign: "middle", margin: 0,
  });

  // Thin teal divider
  s.addShape("rect", { x: 0.4, y: 1.65, w: 9.2, h: 0.04, fill: { color: CARD2 }, line: { color: CARD2 } });

  const prompts = [
    "\"What questions in PMG22_WAI relate to GICs?\"",
    "\"What are the value labels for PMG18_ROB question 32?\"",
    "\"Which surveys asked about credit unions?\"",
  ];

  prompts.forEach((p, i) => {
    const py = 1.82 + i * 0.88;
    // Card with teal left border
    s.addShape("rect", {
      x: 0.4, y: py, w: 9.2, h: 0.72,
      fill: { color: CARD }, line: { color: CARD },
      shadow: mkShadow(),
    });
    s.addShape("rect", { x: 0.4, y: py, w: 0.12, h: 0.72, fill: { color: TEAL }, line: { color: TEAL } });
    s.addText(p, {
      x: 0.65, y: py, w: 8.8, h: 0.72,
      fontSize: 14, fontFace: "Consolas", color: WHITE,
      align: "left", valign: "middle", margin: 0,
    });
  });

  // Bottom note
  s.addText("Live demo follows this presentation", {
    x: 0.4, y: 5.2, w: 9.2, h: 0.35,
    fontSize: 12, fontFace: "Calibri", italic: true, bold: true,
    color: AMBER, align: "center", valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Q&A / THANK YOU
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Left teal bar (mirrors title slide)
  s.addShape("rect", { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: TEAL }, line: { color: TEAL } });

  // Bottom dark panel
  s.addShape("rect", { x: 0, y: 4.4, w: 10, h: 1.225, fill: { color: DARK }, line: { color: DARK } });
  s.addShape("rect", { x: 0, y: 4.38, w: 10, h: 0.05, fill: { color: TEAL }, line: { color: TEAL } });

  // Big Q&A text
  s.addText("Questions?", {
    x: 0.5, y: 0.8, w: 9, h: 1.8,
    fontSize: 64, fontFace: "Calibri", bold: true,
    color: WHITE, align: "left", valign: "middle", margin: 0,
  });

  // Tagline
  s.addText("Research answers. Instantly. With confidence.", {
    x: 0.5, y: 2.7, w: 9, h: 0.65,
    fontSize: 20, fontFace: "Calibri", italic: true,
    color: TEAL, align: "left", valign: "middle", margin: 0,
  });

  // Divider
  s.addShape("rect", { x: 0.5, y: 3.45, w: 4, h: 0.04, fill: { color: TEAL }, line: { color: TEAL } });

  // Credits in bottom bar
  s.addText("Alex Katzighera  |  Capstone Project 2026", {
    x: 0.5, y: 4.5, w: 9, h: 0.45,
    fontSize: 13, fontFace: "Calibri",
    color: MUTED, align: "left", valign: "middle", margin: 0,
  });
}

// ── Write file ────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUTPUT })
  .then(() => console.log("✅  Written:", OUTPUT))
  .catch(err => { console.error("❌ Error:", err); process.exit(1); });
