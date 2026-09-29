"""Governance report (markdown) with evidence citations. Descriptive; does not recommend a policy."""


def build_report(res, top_n=3):
    R = res["retriever"]
    L = ["# CurriculumGuard Governance Report", "",
         "> Descriptive analysis only. It does not declare a correct policy. If the data are synthetic, results demonstrate the method, not real opinion.", ""]
    s = res["schemes"]
    L += ["## 1. Does the outcome depend on the aggregation rule?", "",
          "| scheme | open | constrained | restricted | winner |", "|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.scheme} | {r.open:.1%} | {r.constrained:.1%} | {r.restricted:.1%} | **{r.winner}** |")
    L.append("")
    L.append("Outcome is **robust** to the scheme." if s["winner"].nunique() == 1
             else "Outcome **changes with the aggregation scheme** (" + ", ".join(f"{a}: {b}" for a, b in zip(s.scheme, s.winner)) + ").")
    L += ["", "## 2. Representation audit", "", "| group | n | participation | population | ratio | status |", "|---|---|---|---|---|---|"]
    for _, r in res["audit"].iterrows():
        L.append(f"| {r.group} | {r.n} | {r.participation:.0%} | {r.population:.0%} | {r.ratio:.2f} | {r.status} |")
    L += ["", "## 3. Participation sensitivity (flip thresholds)", ""]
    for _, r in res["flips"].iterrows():
        if r.flips:
            L.append(f"- **{r.group}** ({r.current_share:.0%} of participants): winner `{r.baseline_winner}` changes to `{r.new_winner}` "
                     f"at {r.threshold_share:.0%} ({r.change_pp:+.1f} pp).")
        else:
            L.append(f"- **{r.group}** ({r.current_share:.0%}): no flip at any share; winner stays `{r.baseline_winner}`.")
    L += ["", "## 4. Power-aware risk: high-impact concerns of underrepresented groups", ""]
    hi = res["power"][res["power"]["high_impact"]].head(top_n)
    if hi.empty:
        L.append("None met the high-impact definition (severity >= 4, prevalence >= 25%, underrepresented).")
    for _, r in hi.iterrows():
        L.append(f"- **{r.group} / {r.concern}**: prevalence {r.prevalence:.0%}, severity {r.mean_severity:.1f}/5, risk {r.risk:.2f}.")
        for e in R.search(r.concern.replace("_", " "), k=1):
            L.append(f"  - Evidence [{e['source']}]: {e['text'][:220]}...")
    L += ["", "## 5. Policy trade-offs (assumption-based exposure matrix)", "",
          "| scheme | policy | support | unaddressed-concern burden |", "|---|---|---|---|"]
    for _, r in res["policy"].iterrows():
        L.append(f"| {r.scheme} | {r.policy} | {r.support:.1%} | {r.unaddressed_burden:.3f} |")
    L += ["", "## 6. Limitations", "",
          "- Group shares of the population and policy-exposure values are assumptions; edit them.",
          "- Concern extraction is lexicon-based; validate on hand-labelled real data.",
          "- Evidence notes are starter summaries; replace with vetted sources.",
          "- Small subgroups are suppressed below n=5 and have wide uncertainty."]
    return "\n".join(L)
