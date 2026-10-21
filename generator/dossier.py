"""
dossier.py — turn the research-workflow JSON (six lenses + skeptic fact-checks +
design panel + judges) into docs/RESEARCH_DOSSIER.md.

    python3 generator/dossier.py <dir with result_*.json> docs/RESEARCH_DOSSIER.md
"""
from __future__ import annotations

import glob
import json
import os
import sys

LENS_ORDER = ["science", "eastern", "indigenous", "modern", "sound_design", "safety_use"]
LENS_TITLES = {
    "science": "1. The science of binaural beats and auditory entrainment",
    "eastern": "2. Eastern, Middle-Eastern and classical-Western sacred sound",
    "indigenous": "3. Indigenous, shamanic and ancestral sound technologies",
    "modern": "4. Modern esoteric and consciousness-technology systems",
    "sound_design": "5. Audio engineering for a long-form entrainment track",
    "safety_use": "6. Safety, contraindications, protocol and honest expectations",
}
LEVEL_BADGE = {
    "strong": "**strong**", "moderate": "moderate", "weak": "weak", "anecdotal": "anecdotal",
    "traditional": "traditional (a lineage's own teaching, not a tested claim)", "none": "none (no evidence)",
}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main(src_dir, out_path):
    lines = ["# Research dossier", "",
             "This dossier is the raw material behind the session design. Six research agents each took one lens, "
             "searched the literature and the traditions, and returned findings with an evidence level. A separate "
             "skeptic agent then fact-checked every lens; its corrections are listed after each lens and take precedence "
             "over the original finding wherever they conflict. Evidence levels: *strong* / *moderate* / *weak* / "
             "*anecdotal* are about scientific support; *traditional* means a lineage's own teaching, reported as such; "
             "*none* means a modern claim with no support.", ""]
    for lens in LENS_ORDER:
        rp = os.path.join(src_dir, f"result_research_{lens}.json")
        vp = os.path.join(src_dir, f"result_verify_{lens}.json")
        if not os.path.exists(rp):
            continue
        r = load(rp)
        lines += [f"## {LENS_TITLES.get(lens, lens)}", "", f"*Lens scope:* {r.get('lens', '')}", ""]
        for f in r.get("findings", []):
            lines += [f"### {f['topic']}", "",
                      f"- **Evidence level:** {LEVEL_BADGE.get(f['evidence_level'], f['evidence_level'])}",
                      f"- **Finding:** {f['claim']}",
                      f"- **Usable parameters:** {f.get('usable_parameters', '')}"]
            if f.get("notes"):
                lines.append(f"- **Notes:** {f['notes']}")
            lines += [f"- **Source:** {f.get('source', '')}", ""]
        if r.get("design_recommendations"):
            lines += ["### Design recommendations from this lens", ""] + [f"- {x}" for x in r["design_recommendations"]] + [""]
        if r.get("cautions"):
            lines += ["### Cautions from this lens", ""] + [f"- {x}" for x in r["cautions"]] + [""]
        if os.path.exists(vp):
            v = load(vp)
            lines += ["### Skeptic fact-check of this lens", "", f"*Overall reliability:* {v.get('overall_reliability', '')}", ""]
            if v.get("corrections"):
                lines += ["| Original claim | Verdict | Corrected statement | Source |", "|---|---|---|---|"]
                for c in v["corrections"]:
                    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| {esc(c['original_claim'])} | **{c['verdict']}** | {esc(c['corrected_statement'])} | {esc(c.get('source', ''))} |")
                lines.append("")
            if v.get("missing_items"):
                lines += ["*Items the lens missed, added by the skeptic:*", ""] + [f"- {x}" for x in v["missing_items"]] + [""]
    # design panel + judges
    designs = sorted(glob.glob(os.path.join(src_dir, "result_design_*.json")))
    judges = sorted(glob.glob(os.path.join(src_dir, "result_judge_*.json")))
    if designs:
        lines += ["## 7. Design panel", "", "Three designers each proposed a full architecture from a different school; two judges scored them. "
                  "The shipped session is a synthesis (see DESIGN.md).", ""]
        for i, dp in enumerate(designs):
            d = load(dp)
            lines += [f"### Design {i}: {d.get('title', os.path.basename(dp))} ({d.get('total_minutes', '?')} min)", "", f"*Rationale:* {d.get('rationale', '')}", "",
                      "| Stage | Minutes | Carrier(s) | Beat Hz | Layers | Sources |", "|---|---|---|---|---|---|"]
            for st in d.get("stages", []):
                lines.append(f"| {st['name']} | {st['start_min']:g}–{st['end_min']:g} | {st['binaural_carrier_hz']} | {st['beat_hz_start']:g} → {st['beat_hz_end']:g} | "
                             f"{'; '.join(st.get('layers', []))} | {'; '.join(st.get('tradition_sources', []))} |".replace("\n", " "))
            lines += ["", f"*Global layers:* {'; '.join(d.get('global_layers', []))}", "", f"*Mastering:* {d.get('mastering', '')}", "",
                      f"*Daily short version:* {d.get('short_daily_version', '')}", ""]
        for jp in judges:
            j = load(jp)
            lines += [f"### Judge: {os.path.basename(jp).replace('result_judge_', '').replace('.json', '')}", ""]
            for s in j.get("scores", []):
                lines += [f"- **Design {s['design_index']}: {s['score_out_of_10']}/10.** Strengths: {s['strengths']} Weaknesses: {s['weaknesses']} Technical errors: {s['technical_errors']}"]
            lines += ["", f"*Best:* design {j.get('best_index')}", "", "*Ideas to graft:*", ""] + [f"- {x}" for x in j.get("graft_ideas", [])]
            lines += ["", "*Must fix before build:*", ""] + [f"- {x}" for x in j.get("must_fix_before_build", [])] + [""]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out_path}: {len(lines)} lines")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
