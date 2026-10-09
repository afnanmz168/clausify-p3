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
for k in ("accuracy", "precision", "recall"):          # first-setup rows of the scorecard
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
C("both", f"{att['both']} {pc(att['span_ok_given_window_correct'])}", "tab:e2eattr")
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
                                  f"{f4(r['best_f1'])} {r['best_threshold']:.2f} {f4(r['and_f1_at_05'])}", "tab:pooling")
    C("pooling top-3 gain at 0.5", f3(pool["top-3 mean"]["f1_at_05"] - pool["max (thesis)"]["f1_at_05"]))
    C("pooling best gain", f3(pool["max (thesis)"]["best_f1"] - pool["max (thesis)"]["f1_at_05"]))

# ---- app latency (current app, October models) ---------------------------
ab = J(A, "app_bench.json"); ab1 = J(A, "app_bench_first_setup.json")   # current app; 256-token first setup
for r, r1 in zip(ab["rows"], ab1["rows"]):
    C(f"latency {r['bucket']}", f"{r['chars']:,} {r['windows_after_cap']} {r1['seconds']:.1f} {r['seconds']:.1f} {r['peak_rss_gb']:.2f}", "tab:appbench")
C("latency summary", f"median {ab['median']:.1f} mean {ab['mean']:.1f} worst case {ab['max']:.1f}", "tab:appbench")

# ---- gradient sizes logged during training --------------------------------
import statistics as _st
def _g(path):
    rows = json.load(open(path)); rows = rows if isinstance(rows, list) else rows.get("log_history", rows)
    return [r["grad_norm"] for r in rows if "grad_norm" in r]
for nm in ("presence", "span", "summarizer"):
    g = _g(os.path.join(HERE, f"{nm}_log_history.json"))
    C(f"grad {nm} range", f"{min(g):.2f} to {max(g):.2f}")
    C(f"grad {nm} median", f"median {_st.median(g):.2f}")
g = _g(os.path.join(P, "analysis", "run_full53k_log_history.json"))
C("grad full53k", f"{len(g)} logged points the gradient size ranges from {min(g):.2f} to {max(g):.2f} with a median of {_st.median(g):.2f}, and {100 * sum(x > 1 for x in g) / len(g):.1f}%")

# ---- re-run: whole window, both models tuned on validation (Section 5.3.3) --
V = os.path.join(HERE, "v2_fullwindow")
if os.path.exists(os.path.join(V, "runs", "L512", "results.json")):
    vr = J(V, "runs", "L512", "results.json")
    def d3(x): return f"{x:+.3f}"
    def stats(t):                       # recomputed from counts, so no rounding is applied twice
        tp, fp, fn = t["tp"], t["fp"], t["fn"]; tn = 4182 - tp - fp - fn
        return dict(f1=2*tp/(2*tp+fp+fn), p=tp/(tp+fp), r=tp/(tp+fn), f2=5*tp/(5*tp+4*fn+fp),
                    acc=(tp+tn)/4182, spec=tn/(tn+fp))
    rows = list(vr["reference_original"].items()) + [(k, v["test"]) for k, v in vr["configs"].items()]
    for name, t in rows:
        s = stats(t); cis = f" {ci(*t['micro_f1_ci95'], f3)}" if "micro_f1_ci95" in t else ""
        C(f"v2 row {name}", f"{f3(s['f1'])}{cis} {f3(s['p'])} {f3(s['r'])} {f3(s['f2'])} "
                            f"{pc(t['high_risk_recall'])} ({t['high_risk_missed']})", "tab:v2")
    for name in ("transformer_tuned_f1", "ensemble_tuned_f1", "transformer_tuned_f2", "ensemble_tuned_f2"):
        v = vr["configs"][name]["test"]["vs_tuned_tfidf"]
        C(f"v2 lead {name}", f"{d3(v['diff'])} {ci(*v['ci95'], d3)}")
    m = {k: vr["configs"][k]["test"]["macro_f1_ge10pos"] for k in ("transformer_tuned_f1", "tfidf_tuned_f1")}
    C("v2 macro common", f"{f3(m['transformer_tuned_f1'])} against {f3(m['tfidf_tuned_f1'])}")
    e = stats(vr["configs"]["ensemble_tuned_f1"]["test"])
    C("v2 scorecard acc/spec", f"{pc(e['acc'], 2)} / {pc(e['spec'], 2)}", "tab:scorecard")
    C("v2 scorecard P/R", f"{pc(e['p'], 2)} / {pc(e['r'], 2)}", "tab:scorecard")
    th = {k: vr["configs"][k]["chosen_on_validation"]["global_threshold"] for k in ("transformer_tuned_f1", "transformer_tuned_f2")}
    C("v2 shared thresholds", f"{th['transformer_tuned_f1']:.2f} (F1) and {th['transformer_tuned_f2']:.2f} (F2)", "tab:v2")
    ti = J(V, "runs", "L512", "train_info.json")
    ev = [h["eval_loss"] for h in ti["log_history"] if "eval_loss" in h]
    C("v2 epoch losses", f"{ev[0]:.3f}, {ev[1]:.3f} and {ev[2]:.3f}")
    C("v2 train/score minutes", f"took {ti['train_minutes']:.0f} minutes, and scoring every window of the validation and test contracts another {ti['score_minutes']:.0f} minutes")
    C("v2 train minutes (hyper note)", f"Its training took {ti['train_minutes']:.0f} minutes", "tab:hyper")
    ds = J(V, "runs", "L512", "downstream.json")
    E2 = ds["end_to_end"]["RECALL"]
    C("app e2e presence", f"{E2['n_presence']:,} {pc(E2['presence_recall'])}", "tab:e2e")
    C("app e2e survive", f"{E2['n_span_ok']} {pc(E2['end_to_end'])}", "tab:e2e")
    C("app e2e CI", ci(*E2["end_to_end_ci"], pc), "tab:e2e")
    C("app window correct", f"{E2['window_correct']:,} {pc(E2['window_correct_rate'])}", "tab:e2eattr")
    C("app span ok | right window", f"{round(E2['span_ok_given_window_correct'] * E2['window_correct'])} {pc(E2['span_ok_given_window_correct'])}", "tab:e2eattr")
    C("app window wrong", f"{E2['window_wrong']} {pc(E2['window_wrong'] / E2['n_presence'])}", "tab:e2eattr")
    C("app token-F1 in pipeline", f3(E2["mean_token_f1_given_presence"]), "tab:scorecard")
    for band in ("High", "Medium", "Low"):
        for name in ds["setups"]:
            r = ds["by_band"][band][name]; npos = ds["by_band"][band]["n_test_positives"]
            C(f"band {band} {name}", f"{f3(r['precision'])} {f3(r['recall'])} {f3(r['f1'])} {npos - r['fn']} {r['fn']}", "tab:riskeval2")
    for grp in ("rare10", "common31"):
        for name in ("RECALL", "RECALL_ENS", "BALANCED"):
            r = ds["rare_tail"][grp][name]
            C(f"rare {grp} {name}", f"{f3(r['precision'])} {f3(r['recall'])} {f3(r['f1'])} {r['tp']:,} {r['fn']}", "tab:raretail2")
    cal = ds["calibration"]
    C("calibration raw ECE", f"{cal['transformer_raw']['ece']:.3f}"); C("calibration iso ECE", f"{cal['transformer_isotonic']['ece']:.3f}")
    C("calibration Brier", f"{cal['transformer_raw']['brier']:.3f} to {cal['transformer_isotonic']['brier']:.3f}")
    C("calibration Platt", f"Platt scaling, which fits a single logistic curve, reaches an Expected Calibration Error of {cal['transformer_platt']['ece']:.3f} and a Brier score of {cal['transformer_platt']['brier']:.3f}")
    for nm, lab in (("BALANCED", "for the balanced ensemble"), ("RECALL", "for the app's recall-first default")):
        r = ds["overall"][nm]
        C(f"final cm {nm}", f"TP {r['tp']:,}, FP {r['fp']}, FN {r['fn']} and TN {4182 - r['tp'] - r['fp'] - r['fn']:,} {lab}")
    C("app errors", f"{ds['errors']['n_false_positives']} false positives and only {ds['errors']['n_false_negatives']} misses")
    if os.path.exists(os.path.join(V, "truncation_256.json")):
        tr = J(V, "truncation_256.json")
        C("truncation share read", f"{pc(tr['share_read_median'])} of each window (about {tr['chars_read_median']:,}")
        C("truncation middle 80%", f"{pc(tr['share_read_p10'], 0)} to {pc(tr['share_read_p90'], 0)}")
        C("truncation label noise", f"{pc(tr['share_label_noise'])} of positive training windows "
                                    f"({tr['positive_train_windows_clause_after_cut']:,} of {tr['positive_train_windows']:,})")
    if os.path.exists(os.path.join(V, "runs", "L256", "results.json")):
        cr2 = J(V, "runs", "L256", "results.json")["configs"]; c5 = vr["configs"]
        for name in ("transformer_untuned", "transformer_tuned_f1", "ensemble_tuned_f1", "transformer_tuned_f2", "ensemble_tuned_f2"):
            a, b = stats(cr2[name]["test"]), stats(c5[name]["test"])
            C(f"control row {name}", f"{f3(a['f1'])} / {f3(a['f2'])} {pc(cr2[name]['test']['high_risk_recall'])} "
                                     f"({cr2[name]['test']['high_risk_missed']}) {f3(b['f1'])} / {f3(b['f2'])} "
                                     f"{pc(c5[name]['test']['high_risk_recall'])} ({c5[name]['test']['high_risk_missed']})", "tab:v2ctrl")
        v = cr2["transformer_tuned_f1"]["test"]["vs_tuned_tfidf"]
        C("control 256 tuned loses", f"by {abs(v['diff']):.3f} (interval {ci(*v['ci95'], d3)}")
        cw = J(V, "compare_window.json")
        for k in ("untuned", "f1", "f2"):
            C(f"window gain {k}", f"{d3(cw[k]['diff'])}" + (" micro-F1" if k != "f2" else " micro-F2"))
            C(f"window gain CI {k}", ci(*cw[k]["ci95"], d3))
        C("window gain high-risk", f"{100 * cw['f2']['high_recall_diff']:.1f} points ([{100 * cw['f2']['high_recall_ci95'][0]:+.1f}, "
                                   f"{100 * cw['f2']['high_recall_ci95'][1]:+.1f}])")

    sp2 = J(V, "runs", "L512", "span_position.json")
    C("span position share", f"only {pc(sp2['first_setup']['share_clause_in_second_half'])} of the time. With the re-run model it is {pc(sp2['app']['share_clause_in_second_half'])}")
    C("span position success", f"({pc(sp2['first_setup']['span_ok_first_half'])} and {pc(sp2['app']['span_ok_first_half'])}); in the second half it succeeds only {pc(sp2['first_setup']['span_ok_second_half'])} and {pc(sp2['app']['span_ok_second_half'])}")

    if os.path.exists(os.path.join(V, "runs", "L512", "cards.json")):
        cd = J(V, "runs", "L512", "cards.json"); b5, b2 = cd["below_50"], cd["below_20"]
        C("cards total", f"the default setting shows {cd['n_cards']:,} cards. Of these, {b5['n']} ({pc(b5['share'])})")
        C("cards below 20%", f"{b2['n']} ({pc(b2['share'])}) below 20")
        C("cards real share", f"{pc(b5['real_share'])} of the {b5['n']} are real clauses, and they include {b5['high_real']} of the {cd['high_real_found']} High-risk")
        C("cards recall without", f"from 90.3% to {pc(cd['high_recall_without_possible'])}")
        C("cards >= 50%", f"Of the {cd['at_least_50']['n']:,} cards at 50% or more, {pc(cd['at_least_50']['real_share'])}")
        C("cards non-compete", f"with a chance of about {100 * cd['lowest_non_compete_chance']:.0f}%")

AB = os.path.join(A, "extra_windows_ab.json")
if os.path.exists(AB):
    ab2 = J(AB); m = lambda k, c: st.mean(r["seconds"] for r in ab2 if r["extra"] == k and r["contract"] == c)
    C("extra windows A/B", f"the 75th-percentile contract took {m(5, '75th'):.1f} against {m(0, '75th'):.1f} seconds on average, and the longest {m(5, 'longest'):.1f} against {m(0, 'longest'):.1f} seconds")

# ---- what the app reads in long contracts (window_cap.py, app_presence_test.py) ----
WC = os.path.join(V, "runs", "L512", "window_cap.json")
if os.path.exists(WC):
    wc = J(WC)["test"]; ap = J(V, "runs", "L512", "app_presence_test.json")
    for row, r in (("Every window", wc["whole"]), ("First 30 windows", wc["first30"])):
        C(f"window cap {row}", f"{pc(r['high_found'])} ({r['high_missed']}) {f3(r['micro_f1'])} {f3(r['micro_f2'])} {pc(r['high_found_long'])}", "tab:windowcap")
    C("window cap app (measured)", f"{pc(ap['high_found'])} ({ap['high_missed']}) {f3(ap['micro_f1'])} {f3(ap['micro_f2'])} {pc(ap['high_found_long'])}", "tab:windowcap")
    C("window cap replay = app", f"{pc(wc['chosen']['high_found'])} ({wc['chosen']['high_missed']}) {f3(wc['chosen']['micro_f1'])}", "tab:windowcap")
    C("window cap k", f"The number {J(WC)['chosen']['k']} was chosen on the 81 validation contracts")
    C("window cap long n", f"on the {ap['n_long']} test contracts that are longer than the limit")

# ---- retrained span model (Section 5.2.1, retrain/span_v2) ----------------
SV = os.path.join(HERE, "span_v2", "runs", "chunks")
if os.path.exists(os.path.join(SV, "evaluation.json")):
    ev = J(SV, "evaluation.json"); tv = J(SV, "train_info.json"); va = ev["validation"]
    R1, R2 = ev["test"]["RECALL"]["first_model"], ev["test"]["RECALL"]["new_model"]
    B1, B2 = ev["test"]["BALANCED"]["first_model"], ev["test"]["BALANCED"]["new_model"]
    C("spanv2 e2e", f"{R1['n_ok']} {pc(R1['end_to_end'])} {R2['n_ok']} {pc(R2['end_to_end'])}", "tab:spanv2")
    C("spanv2 e2e CI", f"{ci(*R1['end_to_end_ci'], pc)} {ci(*R2['end_to_end_ci'], pc)}", "tab:spanv2")
    C("spanv2 right window", f"{pc(R1['ok_given_right_window'])} {pc(R2['ok_given_right_window'])}", "tab:spanv2")
    C("spanv2 first half", f"{pc(R1['ok_first_half'])} {pc(R2['ok_first_half'])}", "tab:spanv2")
    C("spanv2 second half", f"{pc(R1['ok_second_half'])} {pc(R2['ok_second_half'])}", "tab:spanv2")
    C("spanv2 mean F1", f"{f3(ev['test']['RECALL']['mean_token_f1_first'])} {f3(ev['test']['RECALL']['mean_token_f1_new'])}", "tab:spanv2")
    C("spanv2 balanced", f"{B1['n_ok']} {pc(B1['end_to_end'])} {B2['n_ok']} {pc(B2['end_to_end'])}", "tab:spanv2")
    rw = ev["test"]["right_window"]
    C("spanv2 plain window F1", f"{f3(rw['first_model_mean_f1'])} {f3(rw['new_model_mean_f1'])}", "tab:spanv2")
    C("spanv2 plain window ok", f"{pc(rw['first_model_ok'])} {pc(rw['new_model_ok'])}", "tab:spanv2")
    C("spanv2 validation", f"{pc(va['grid']['epoch2_minus_null_cap120'])} of the detected clauses at token-F1 >= 0.5, against {pc(va['old_model_ok'])}".replace(">=", "\\geq"))
    C("spanv2 training data", f"{tv['answers']:,} clauses from the {tv['fit_contracts']} contracts")
    C("spanv2 chunks", f"{tv['chunk_examples']:,} chunks ({pc(tv['positive_chunks'] / tv['chunk_examples'])} with an answer). Training took {tv['train_minutes']:.1f} minutes")
    q = ev["test"]["app_quote"]; qp, qs = q["presence"], q["span"]
    C("quote presence", f"{pc(qp['covers_half_of_clause']['share_of_detected'])} of the detected clauses are shown this way (median quote {qp['median_quote_chars']:.0f} characters)")
    C("quote span", f"{pc(qs['covers_half_of_clause']['share_of_detected'])} are ({qs['median_quote_chars']:.0f} characters)")
    C("quote validation", f"({pc(va['quote_cov_span_paragraph'])} against {pc(va['quote_cov_presence_paragraph'])})")
    C("quote overall", f"Over all real clauses, {pc(qs['covers_half_of_clause']['end_to_end'])} get a card whose quote contains at least half of the clause (95% CI {ci(*qs['covers_half_of_clause']['end_to_end_ci'], pc)})")
    C("quote strict", f"({pc(qs['token_f1_ok']['share_of_detected'])} and {pc(qp['token_f1_ok']['share_of_detected'])} of detected clauses)")
    E2 = J(V, "runs", "L512", "downstream.json")["end_to_end"]["RECALL"]
    prod = E2["presence_recall"] * E2["window_correct_rate"] * R2["ok_given_right_window"]
    C("spanv2 product", f"which multiplies to {pc(prod)}; another {R2['n_ok'] - round(R2['ok_given_right_window'] * R2['right_window'])} clauses")
    lost_window = E2["window_wrong"] - (R2["n_ok"] - round(R2["ok_given_right_window"] * R2["right_window"]))
    lost_span = R2["right_window"] - round(R2["ok_given_right_window"] * R2["right_window"])
    C("spanv2 losses", f"The {ev['n_gold_test'] - R2['n_ok']} clauses still lost split three ways: {ev['n_gold_test'] - R2['n_detected']} are never detected, {lost_window} are in a window the presence model did not choose, and {lost_span} are in the right window")
    C("spanv2 future", f"{lost_window} clauses against the {lost_span} the span model misses")
    C("spanv2 miss right window", f"still misses {pc(1 - R2['ok_given_right_window'])} of clauses in the right window")
    C("quote rest", f"The other {pc(1 - qs['covers_half_of_clause']['end_to_end'])} are clauses")
    import hashlib
    h = hashlib.sha256(open(os.path.join(P, "notebooks", "outputs", "span_v2", "final", "model.safetensors"), "rb").read()).hexdigest()[:16]
    C("span_v2 fingerprint", h)

# ---- clause classifier (retrain/clause_v2) ----------------------------------
CE = os.path.join(HERE, "clause_v2", "runs", "evaluation.json")
if os.path.exists(CE):
    ce = J(CE); tc = ce["test_chosen"]; vv = ce["validation"]; bv = J(HERE, "clause_v2", "runs", "bert_val.json")
    C("clause v2 test", f"it gives the right category {pc(tc['acc41'])} of the time, against {pc(ce['old_presence_zscore']['acc41'])} before, the right one in its top three {pc(tc['top3'])} of the time, and the right risk level {pc(tc['risk_level_acc'])}")
    C("clause v2 any label", f"it is right {pc(tc['acc_any_label'])} of the time")
    C("clause v2 none", f"It calls only {pc(tc['clauses_called_none'])} of real clauses")
    C("clause v2 none lines", f"of {tc['n_none']} test lines that belong to no clause, it recognizes {pc(tc['none_acc'], 0)}")
    C("clause v2 validation", f"({pc(vv['tfidf']['acc41'])}, against {pc(vv['bert']['acc41'])} for DistilBERT and {pc(vv['average']['acc41'])} for the average)")
    C("clause v2 bert epochs", ", ".join(pc(h["eval_acc41"]) for h in bv["epochs_eval"][:-1]) + f" and {pc(bv['epochs_eval'][-1]['eval_acc41'])}")
    C("clause v2 bert minutes", f"three epochs, {bv['train_minutes']:.1f} minutes")
    C("clause v2 n test", f"On the same {tc['n_clauses']} test clauses")

# ---- larger models for the plain-English line (retrain/plain_v2) -----------
PE = os.path.join(HERE, "plain_v2", "runs", "plain_english.json")
if os.path.exists(PE):
    pe = J(PE); tcp = pe["test_chosen"]; fs = J(HERE, "plain_v2", "runs", "fewshot.json")["result"]
    rows = J(HERE, "plain_v2", "runs", "rows_" + pe["chosen"].replace("|", "_") + ".json")
    test_i = [i for i in range(150) if i not in set(pe["dev_idx"])]
    near = sum(rows[i]["copy"] >= 0.9 for i in test_i) / len(test_i)
    C("plain v2 scores", f"it scores {f3(tcp['rouge_l'])}, against {f3(pe['baselines_on_test_100']['template_vs_diverse'])} for the category template and {f3(pe['baselines_on_test_100']['finetuned_vs_diverse'])} for our fine-tuned model")
    C("plain v2 copy", f"On average {pc(tcp['copy_share'])} of an output's three-word sequences appear word for word in the clause, and {pc(near, 0)} of the outputs")
    C("plain v2 fewshot", f"ROUGE-L {f3(fs['rouge_l'])}, with {pc(fs['near_verbatim'], 0)} of outputs near-verbatim")
    C("plain v2 split", f"into {pe['dev_n']} for choosing the model and instruction and {pe['test_n']} for testing")

# ---- clause mode (app repository) -----------------------------------------
CAL2 = os.path.join(os.path.dirname(P), "final project app p3", "clause_calibration_v2.json")
if os.path.exists(CAL2):
    cv = J(CAL2)["eval"]; c1 = J(os.path.dirname(CAL2), "clause_calibration.json")["eval"]
    C("clause mode first version", f"the right category out of 41 for {pc(cv['calibrated_acc'])} of clauses ({pc(cv['raw_argmax_acc'])} with the raw scores)")
    C("clause mode top3/risk", f"the right one in the top three {pc(cv['calibrated_top3'])} of the time and the right risk level {pc(cv['risk_level_acc'])} of the time")
    C("clause mode unrecognized", f"it left {pc(cv['share_test_max_z_below_2'])} of clauses unrecognized")

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
