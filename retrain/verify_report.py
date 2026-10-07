"""Check the numbers reported in project_report/ against the result files.

Every check recomputes a value from a result file (October 2026 runs in retrain/,
August 2026 control runs in analysis/), formats it the way the report writes it,
and looks for it in the report: inside one table when a table label is given,
otherwise anywhere in the chapters, abstract and appendices.

    python3 retrain/verify_report.py        # prints "0 mismatched" when all agree
"""
import glob, json, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(HERE)
REP = os.path.join(P, "project_report")


def J(*p):
    return json.load(open(os.path.join(*p)))


pres = J(HERE, "stage2a_presence_test.json")
boot = J(HERE, "stage2a_bootstrap.json")
A = os.path.join(HERE, "analysis_oct2026")
span = J(A, "span_ci.json"); e2e = J(A, "e2e_AND.json"); att = J(A, "e2e_attribution.json")
rt = J(A, "rare_tail.json"); rtc = J(A, "risk_threshold_calib.json"); sm = J(A, "summ_pilot.json")
pool = J(A, "pooling_ablation.json") if os.path.exists(os.path.join(A, "pooling_ablation.json")) else None
cr = J(P, "analysis", "compare_runs.json"); sv = J(P, "analysis", "split_variance.json")

files = sorted(glob.glob(os.path.join(REP, "chapters", "*.tex"))) + \
        [os.path.join(REP, "core", "abstract.tex")] + sorted(glob.glob(os.path.join(REP, "appendix", "*.tex")))
TEX = "\n".join(open(f).read() for f in files)


def norm(s):
    s = s.replace("{,}", ",").replace("\\%", "%").replace("\\ ", " ").replace("\\textbf{", "").replace("$", "")
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{", "", s)
    return " ".join(s.split())


ALL = norm(TEX)


def table(label):
    i = TEX.index("\\label{%s}" % label)
    a = TEX.rfind("\\begin{table}", 0, i); b = TEX.index("\\end{table}", i)
    return norm(TEX[a:b])


def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def pc(x, d=1): return f"{100 * x:.{d}f}%"
def sg(x): return f"{x:+.4f}"
def ci(lo, hi, fmt): return f"[{fmt(lo)}, {fmt(hi)}]"


checks = []  # (description, expected string, table label or None)
def C(desc, s, tab=None): checks.append((desc, s, tab))

# ---- presence -------------------------------------------------------------
et = {r["model"].split()[0] + r["model"].split()[1][:3]: r["micro_F1"] for r in pres["ensemble_table"]}
C("AND micro-F1", f3(pres["and_ensemble"]["micro_f1"]), "tab:ensemble")
C("TF-IDF micro-F1", f3(pres["tfidf"]["micro_f1"]), "tab:ensemble")
C("transformer micro-F1", f3(pres["transformer"]["micro_f1"]), "tab:ensemble")
for k, v in et.items():
    C(f"ensemble table {k}", f3(v), "tab:ensemble")
cm = pres["confusion"]
for k in ("tn", "fp", "fn", "tp"):
    C(f"confusion {k}", f"{k.upper()} = {cm[k]:,}", "tab:cm")
m = pres["metrics"]
C("accuracy", pc(m["accuracy"], 2), "tab:metrics"); C("precision", pc(m["precision"], 2), "tab:metrics")
C("recall", pc(m["recall"], 2), "tab:metrics"); C("specificity", pc(m["specificity"], 2), "tab:metrics")
C("F1", f4(m["f1"]), "tab:metrics")
for k in ("accuracy", "precision", "recall", "specificity"):
    C(f"scorecard {k}", pc(m[k], 2), "tab:scorecard")

# ---- bootstrap ------------------------------------------------------------
for k in ("AND_vs_TFIDF", "TFIDF_vs_transformer", "AND_vs_transformer"):
    b = boot[k]
    C(f"bootstrap {k} diff", sg(b["diff"]), "tab:bootstrap")
    C(f"bootstrap {k} CI", ci(*b["ci95"], sg), "tab:bootstrap")
    C(f"bootstrap {k} P", f3(b["p_gt0"]), "tab:bootstrap")
C("AND alone CI", ci(*boot["AND"]["ci95"], f3)); C("TF-IDF alone CI", ci(*boot["TFIDF"]["ci95"], f3))
C("AND ahead share", pc(boot["AND_vs_TFIDF"]["p_gt0"]))

# ---- span -----------------------------------------------------------------
C("span token-F1", f3(span["token_f1_contract"]["mean"]), "tab:span")
C("span overlap", pc(span["overlap_contract"]["mean"]), "tab:span")
C("span EM", pc(span["em_contract"]["mean"]), "tab:span")
for k, fmt in (("token_f1", f3), ("overlap", pc), ("em", pc)):
    for unit in ("contract", "clause"):
        d = span[f"{k}_{unit}"]
        C(f"span {k} CI by {unit}", ci(d["lo"], d["hi"], fmt), "tab:span")

# ---- summarizer -----------------------------------------------------------
for k in ("ft_vs_template", "ft_vs_diverse", "zs_vs_diverse", "template_vs_diverse"):
    C(f"summ {k}", f3(sm[k]["mean"]), "tab:sumeval")
    C(f"summ {k} CI", ci(*sm[k]["ci"], f3), "tab:sumeval")
C("exact own-template share", pc(sm["exact_template_rate"], 0))

# ---- end to end -----------------------------------------------------------
C("e2e presence", f"{e2e['n_presence']:,}", "tab:e2e"); C("e2e presence %", pc(e2e["presence_recall"]), "tab:e2e")
C("e2e survive", pc(e2e["end_to_end"]), "tab:e2e"); C("e2e count", f"{e2e['n_end_to_end']}", "tab:e2e")
C("e2e CI", ci(*e2e["end_to_end_ci"], pc), "tab:e2e")
C("e2e mean token-F1", f3(e2e["mean_token_f1_given_presence"]))
C("predicted survival", f"{round(100 * e2e['presence_recall'] * span['overlap_contract']['mean'])}%")
C("window correct", f"{att['window_correct']} {pc(att['window_correct_rate'])}", "tab:e2eattr")
C("both", f"{att['both']} {pc(att['span_ok_given_window_correct'])} of {att['window_correct']}", "tab:e2eattr")
C("window wrong", f"{att['window_wrong']} {pc(att['window_wrong'] / att['n'])}", "tab:e2eattr")

# ---- risk bands -----------------------------------------------------------
for band, d in rtc["by_band"].items():
    for mdl in ("TRANS", "TFIDF", "AND", "OR"):
        r = d[mdl]
        for k in ("precision", "recall", "f1"):
            C(f"{band} {mdl} {k}", f3(r[k]), "tab:riskeval")
        C(f"{band} {mdl} found/missed", f"{r['tp']} {r['fn']}", "tab:riskeval")
for mdl, v in rtc["high_risk_miss_rate"].items():
    if mdl != "TFIDF":
        C(f"High miss {mdl}", pc(v))

sw = {m: {r["t"]: r for r in rtc["threshold_sweep"][m]} for m in ("TRANS", "AND")}
C("TRANS F1 at 0.9", f3(sw["TRANS"][0.9]["f1"])); C("TRANS precision at 0.9", f3(sw["TRANS"][0.9]["precision"]))
C("TRANS precision at 0.5", f3(sw["TRANS"][0.5]["precision"]))
C("TRANS gain", f3(sw["TRANS"][0.9]["f1"] - sw["TRANS"][0.5]["f1"]))
C("TRANS High recall at 0.9", f3(sw["TRANS"][0.9]["high_recall"]))
C("AND F1 at 0.7", f3(sw["AND"][0.7]["f1"])); C("AND High recall at 0.7", f3(sw["AND"][0.7]["high_recall"]))
cal = rtc["calibration"]
C("ECE transformer", f3(cal["TRANS (max-pooled)"]["ece"])); C("Brier transformer", f3(cal["TRANS (max-pooled)"]["brier"]))
C("ECE TF-IDF", f3(cal["TFIDF"]["ece"]))
bins = {round(b["bin"][0], 1): b for b in cal["TRANS (max-pooled)"]["bins"]}
C("0.5-0.6 bin", pc(bins[0.5]["acc"])); C("top bin", pc(bins[0.9]["acc"]))

# ---- rare tail ------------------------------------------------------------
for grp in ("rare10", "common31"):
    for mdl in ("TRANS", "TFIDF", "AND"):
        r = rt[grp][mdl]
        C(f"{grp} {mdl}", f"{f3(r['precision'])} {f3(r['recall'])} {f3(r['f1'])} {r['tp']:,} {r['fn']}", "tab:raretail")
    C(f"{grp} precision gain", f"{100 * rt[grp]['precision_gain_vs_trans']:.1f} points of precision")
    C(f"{grp} recall cost", f"{100 * rt[grp]['recall_cost_vs_trans']:.1f} points of recall")

# ---- seeds, budget, split (seed 42 = October, 43/44 = August) ------------
pe = cr["point_estimates"]
T = [pres["transformer"]["micro_f1"], pe["sub24k_seed43:TRANS"]["micro_f1"], pe["sub24k_seed44:TRANS"]["micro_f1"]]
Aa = [pres["and_ensemble"]["micro_f1"], pe["sub24k_seed43:AND"]["micro_f1"], pe["sub24k_seed44:AND"]["micro_f1"]]
for nm, x in (("transformer", T), ("AND", Aa)):
    C(f"seeds {nm}", " ".join(f4(v) for v in x) + f" {f4(st.mean(x))} {f4(st.stdev(x))} {f4(max(x) - min(x))}", "tab:seeds")
C("gap in seed SDs", f"{boot['TFIDF_vs_transformer']['diff'] / st.stdev(T):.1f} times")
for k in ("base24k_seed42:TRANS", "full53k_seed42:TRANS"):
    r = pe[k]; C(f"budget {k}", f"{f4(r['micro_f1'])} {f4(r['precision'])} {f4(r['recall'])}", "tab:budget")
co = cr["contrasts"]
c1 = co["full53k_seed42:TRANS - base24k_seed42:TRANS"]; c2 = co["TFIDF - full53k_seed42:TRANS"]
C("budget contrast 1", f"{sg(c1['diff'])} {ci(c1['lo'], c1['hi'], sg)}", "tab:budget")
C("budget contrast 2", f"{sg(c2['diff'])} {ci(c2['lo'], c2['hi'], sg)}", "tab:budget")
C("split variance", " ".join(f4(sv[k]) for k in ("micro_mean", "micro_sd", "micro_min", "micro_max", "micro_range")), "tab:splitvar")

# ---- aggregation ablation -------------------------------------------------
if pool:
    for name, r in pool.items():
        if isinstance(r, dict):
            C(f"pooling {name}", f"{f4(r['f1_at_05'])} {f3(r['precision_at_05'])} {f3(r['recall_at_05'])} "
                                  f"{f4(r['best_f1'])} {r['best_threshold']:.2f}", "tab:pooling")

# ---- run ------------------------------------------------------------------
def found(s, hay):
    """Values must appear in the given order, separated only by non-numeric table
    markup (cell separators, bold, percent signs), and never as part of a longer number."""
    toks = s.split(" ")
    if len(toks) == 1:
        return s in hay
    pat = r"[^0-9]{0,40}?".join(r"(?<![\d.])" + re.escape(t) + r"(?![\d])" for t in toks)
    return re.search(pat, hay) is not None


bad = 0
for desc, s, tab in checks:
    hay = table(tab) if tab else ALL
    if not found(s, hay):
        bad += 1; print(f"MISMATCH  {desc:34s} expected '{s}'" + (f" in {tab}" if tab else ""))
print(f"{len(checks)} checks, {bad} mismatched")
