"""Cross-configuration tests linking the one-round theory quantities (R6, R6b) to the end-of-training measurements.
(1) median LHS/RHS of the substitution inequality vs the measured within-config interaction (E1 seeds 0-9, E7 10-19);
(2) weighted tilt CV and target displacement/spread vs the measured |ACC(post_footprint_off) - ACC(post_mass_off)|."""
import csv, statistics as st
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"
def avg(path, key):
    d = {}
    for r in csv.DictReader(open(RES / path)): d.setdefault(r["dataset"], []).append(float(r[key]))
    return {k: st.mean(v) for k, v in d.items()}
ratio = avg("t9_theory_link.csv", "ratio_median"); cv = avg("t9_theory_link_b.csv", "tilt_cv_median"); rel = avg("t9_theory_link_b.csv", "rel_median")
def parse(path, marker):
    out = {}
    for line in open(RES / path):
        if "interaction" in line and "p=" in line and (marker is None or marker in line):
            name, rest = line.split("interaction", 1); out[name.strip()] = float(rest.split()[0])
    return out
e1 = parse("t8_objection_analysis.txt", "ceiling"); e7 = parse("t8_e7_confirm_analysis.txt", None)
src = (HERE / "T7_final_analysis.py").read_text().split('print("="*100); print("0.')[0]
ns = {"__file__": str(HERE / "T7_final_analysis.py")}; exec(src, ns); M = ns["M"]
gap = {ds: abs(M(ds, "post_footprint_off") - M(ds, "post_mass_off")) for ds in cv}
ks = sorted(set(ratio) & set(e1) & set(e7) & set(cv))
def show(name, a, b):
    r = spearmanr([a[k] for k in ks], [b[k] for k in ks]); print(f"  {name:<62} rho = {r.correlation:+.3f}  p = {r.pvalue:.4f}  (n = {len(ks)})")
print("CROSS-CONFIGURATION LINK BETWEEN ONE-ROUND THEORY QUANTITIES AND END-OF-TRAINING MEASUREMENTS")
show("median LHS/RHS  vs  interaction, exploratory seeds 0-9", ratio, e1)
show("median LHS/RHS  vs  interaction, held-out seeds 10-19", ratio, e7)
show("weighted tilt CV  vs  |ACC(footprint) - ACC(mass)| at POST, no pers.", cv, gap)
show("target displacement / centre spread  vs  same gap", rel, gap)
