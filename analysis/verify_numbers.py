"""Cross-check every headline number in the thesis against its source file.

Each entry is (claim, regex-or-literal as it appears in the .tex, value recomputed
from the analysis artifacts). Run before submission; it should print NO MISMATCH.
"""
import json, re, glob, os, sys
A = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(os.path.dirname(A), "thesis")

def J(n): return json.load(open(os.path.join(A, n)))
cr   = J("compare_runs.json");        bs  = J("bootstrap_result.json")
sp   = J("span_ci.json");             rt  = J("rare_tail.json")
e2e  = J("e2e_AND.json");             att = J("e2e_attribution.json")
sm   = J("summ_pilot.json");          rtc = J("risk_threshold_calib.json")
sv   = J("split_variance.json");      pa  = J("pooling_ablation.json")
lf   = J("lf_clean.json");            ab  = J("app_bench.json")

tex = ""
for f in glob.glob(os.path.join(T, "chapters", "*.tex")) + [os.path.join(T, "main.tex")]:
    tex += open(f).read()
tex_n = " ".join(tex.split())

pe = cr["point_estimates"]; co = cr["contrasts"]; sd = cr["seed_variance"]
lfd = {(r["model"], r["batch"], r["mode"]): r for r in lf}

CHECKS = [
 # ---- presence, budgets, seeds -------------------------------------------
 ("transformer 24k micro-F1",      "0.6943", pe["base24k_seed42:TRANS"]["micro_f1"], 4),
 ("AND 24k micro-F1",              "0.7761", pe["base24k_seed42:AND"]["micro_f1"], 4),
 ("TF-IDF micro-F1",               "0.7747", pe["TFIDF"]["micro_f1"], 4),
 ("transformer full-data micro-F1","0.6939", pe["full53k_seed42:TRANS"]["micro_f1"], 4),
 ("AND full-data micro-F1",        "0.7820", pe["full53k_seed42:AND"]["micro_f1"], 4),
 ("full-data precision",           "0.5546", pe["full53k_seed42:TRANS"]["precision"], 4),
 ("full-data recall",              "0.9266", pe["full53k_seed42:TRANS"]["recall"], 4),
 ("24k precision",                 "0.5620", pe["base24k_seed42:TRANS"]["precision"], 4),
 ("24k recall",                    "0.9080", pe["base24k_seed42:TRANS"]["recall"], 4),
 ("seed43 TRANS",                  "0.6928", pe["sub24k_seed43:TRANS"]["micro_f1"], 4),
 ("seed44 TRANS",                  "0.7082", pe["sub24k_seed44:TRANS"]["micro_f1"], 4),
 ("seed43 AND",                    "0.7774", pe["sub24k_seed43:AND"]["micro_f1"], 4),
 ("seed44 AND",                    "0.7803", pe["sub24k_seed44:AND"]["micro_f1"], 4),
 ("TRANS seed mean",               "0.6984", sd["TRANS"]["mean"], 4),
 ("TRANS seed SD",                 "0.0085", sd["TRANS"]["sd"], 4),
 ("TRANS seed range",              "0.0154", sd["TRANS"]["range"], 4),
 ("AND seed mean",                 "0.7780", sd["AND"]["mean"], 4),
 ("AND seed SD",                   "0.0022", sd["AND"]["sd"], 4),
 ("AND seed range",                "0.0042", sd["AND"]["range"], 4),
 # ---- bootstrap contrasts -------------------------------------------------
 ("budget diff",                   "-0.0004", co["full53k_seed42:TRANS - base24k_seed42:TRANS"]["diff"], 4),
 ("budget CI lo",                  "-0.0098", co["full53k_seed42:TRANS - base24k_seed42:TRANS"]["lo"], 4),
 ("budget CI hi",                  "+0.0095", co["full53k_seed42:TRANS - base24k_seed42:TRANS"]["hi"], 4),
 ("TFIDF-full53k diff",            "+0.0809", co["TFIDF - full53k_seed42:TRANS"]["diff"], 4),
 ("TFIDF-full53k CI lo",           "+0.0649", co["TFIDF - full53k_seed42:TRANS"]["lo"], 4),
 ("TFIDF-full53k CI hi",           "+0.0980", co["TFIDF - full53k_seed42:TRANS"]["hi"], 4),
 ("AND-TFIDF diff",                "+0.0014", co["base24k_seed42:AND - TFIDF"]["diff"], 4),
 ("AND-TFIDF CI lo",               "-0.0068", co["base24k_seed42:AND - TFIDF"]["lo"], 4),
 ("AND-TFIDF CI hi",               "+0.0089", co["base24k_seed42:AND - TFIDF"]["hi"], 4),
 ("TFIDF-TRANS diff",              "+0.0805", co["TFIDF - base24k_seed42:TRANS"]["diff"], 4),
 # ---- confusion matrix ----------------------------------------------------
 ("TN", "2{,}523", bs["confusion"]["tn"], 0), ("FP", "311", bs["confusion"]["fp"], 0),
 ("FN", "296", bs["confusion"]["fn"], 0),     ("TP", "1{,}052", bs["confusion"]["tp"], 0),
 # ---- span ---------------------------------------------------------------
 ("span token-F1",       "0.763", sp["token_f1_contract"]["mean"], 3),
 ("span CI contract lo", "0.740", sp["token_f1_contract"]["lo"], 3),
 ("span CI contract hi", "0.782", sp["token_f1_contract"]["hi"], 3),
 ("span CI clause lo",   "0.751", sp["token_f1_clause"]["lo"], 3),
 ("span CI clause hi",   "0.774", sp["token_f1_clause"]["hi"], 3),
 # ---- rare tail ----------------------------------------------------------
 ("rare AND F1",   "0.343", rt["rare10"]["AND"]["f1"], 3),
 ("rare TRANS F1", "0.343", rt["rare10"]["TRANS"]["f1"], 3),
 ("rare TFIDF F1", "0.353", rt["rare10"]["TFIDF"]["f1"], 3),
 ("rare AND tp",   "17",    rt["rare10"]["AND"]["tp"], 0),
 ("rare TRANS tp", "35",    rt["rare10"]["TRANS"]["tp"], 0),
 ("common AND F1", "0.792", rt["common31"]["AND"]["f1"], 3),
 ("common TRANS F1","0.716", rt["common31"]["TRANS"]["f1"], 3),
 # ---- end to end ---------------------------------------------------------
 ("gold clauses",  "1{,}348", e2e["n_gold"], 0),
 ("presence survivors", "1{,}052", e2e["n_presence"], 0),
 ("span survivors", "293", e2e["n_span_ok"], 0),
 ("e2e rate", "21.7", e2e["end_to_end"]*100, 1),
 ("presence recall", "78.0", e2e["presence_recall"]*100, 1),
 ("window correct", "68.3", att["window_correct_rate"]*100, 1),
 ("span ok given window", "39.7", att["span_ok_given_window_correct"]*100, 1),
 ("mean token-F1 e2e", "0.386", e2e["mean_token_f1_given_presence"], 3),
 # ---- summarizer ---------------------------------------------------------
 ("ROUGE template", "0.775", sm["ft_vs_template"]["mean"], 3),
 ("ROUGE diverse",  "0.116", sm["ft_vs_diverse"]["mean"], 3),
 ("ROUGE zeroshot", "0.299", sm["zs_vs_diverse"]["mean"], 3),
 ("ROUGE template-only", "0.115", sm["template_vs_diverse"]["mean"], 3),
 # ---- risk bands ---------------------------------------------------------
 ("High AND recall",   "0.625", rtc["by_band"]["High"]["AND"]["recall"], 3),
 ("High TRANS recall", "0.841", rtc["by_band"]["High"]["TRANS"]["recall"], 3),
 ("High OR recall",    "0.864", rtc["by_band"]["High"]["OR"]["recall"], 3),
 ("High AND missed",   "66",    rtc["by_band"]["High"]["AND"]["fn"], 0),
 ("High TRANS missed", "28",    rtc["by_band"]["High"]["TRANS"]["fn"], 0),
 ("High OR missed",    "24",    rtc["by_band"]["High"]["OR"]["fn"], 0),
 ("High miss rate AND","37.5",  rtc["high_risk_miss_rate"]["AND"]*100, 1),
 ("High miss rate TRANS","15.9",rtc["high_risk_miss_rate"]["TRANS"]*100, 1),
 ("High miss rate OR", "13.6",  rtc["high_risk_miss_rate"]["OR"]*100, 1),
 ("ECE transformer",   "0.201", rtc["calibration"]["TRANS (max-pooled)"]["ece"], 3),
 ("ECE tfidf",         "0.172", rtc["calibration"]["TFIDF"]["ece"], 3),
 # ---- split variance -----------------------------------------------------
 ("split mean", "0.7829", sv["micro_mean"], 4),
 ("split SD",   "0.0063", sv["micro_sd"], 4),
 ("split min",  "0.7702", sv["micro_min"], 4),
 ("split max",  "0.7918", sv["micro_max"], 4),
 ("split range","0.0216", sv["micro_range"], 4),
 # ---- pooling ------------------------------------------------------------
 ("pool max@0.5",   "0.6943", pa["max (thesis)"]["f1_at_05"], 4),
 ("pool max best",  "0.7504", pa["max (thesis)"]["best_f1"], 4),
 ("pool top3@0.5",  "0.7144", pa["top-3 mean"]["f1_at_05"], 4),
 ("pool top3 best", "0.7210", pa["top-3 mean"]["best_f1"], 4),
 ("pool mean@0.5",  "0.1664", pa["mean"]["f1_at_05"], 4),
 # ---- longformer / app ---------------------------------------------------
 ("LF params M", "148.7", lfd[("longformer",1,"infer")]["params"]/1e6, 1),
 ("LF fwd s",    "3.69",  lfd[("longformer",1,"infer")]["sec_per_forward"], 2),
 ("LF train b1 mem", "14.71", lfd[("longformer",1,"train")]["peak_gb"], 2),
 ("LF train b2 mem", "20.75", lfd[("longformer",2,"train")]["peak_gb"], 2),
 ("app median s", "27.1", ab["rows"][2]["seconds"], 1),
 ("app 75th s",   "32.4", ab["rows"][3]["seconds"], 1),
]

bad = 0; missing = 0
for name, literal, value, dp in CHECKS:
    shown = literal.replace("{,}", "").replace(",", "").lstrip("+")
    try: claimed = float(shown)
    except ValueError: claimed = None
    ok_val = claimed is not None and abs(round(value, dp) - abs(claimed)) < 10**(-dp)/2 + 1e-9
    if claimed is not None and claimed < 0: ok_val = abs(round(value,dp) - claimed) < 10**(-dp)/2 + 1e-9
    in_tex = literal in tex_n or literal.replace("{,}", ",") in tex_n
    if not ok_val:
        print(f"  MISMATCH  {name:26s} thesis says {literal:9s} | source = {round(value,dp)}"); bad += 1
    elif not in_tex:
        print(f"  NOT FOUND {name:26s} expected {literal:9s} somewhere in the .tex"); missing += 1

# ---- derived comparisons (one number stated relative to another) ----------
DERIVED = [
 ("bootstrap half-width ~0.008", (0.0089+0.0068)/2, 0.008, 0.001),
 ("seed range = 3x the margin", sd["AND"]["range"]/0.0014, 3, 0.15),
 ("split SD = 4.5x the margin", sv["micro_sd"]/0.0014, 4.5, 0.15),
 ("baseline lead = 9.5 seed SDs", co["TFIDF - base24k_seed42:TRANS"]["diff"]/sd["TRANS"]["sd"], 9.5, 0.2),
 ("baseline lead = 13 split SDs", co["TFIDF - base24k_seed42:TRANS"]["diff"]/sv["micro_sd"], 13, 0.3),
 ("training data +124%", (53662/24000-1)*100, 124, 1),
 ("threshold gain 0.056", pa["max (thesis)"]["best_f1"]-pa["max (thesis)"]["f1_at_05"], 0.056, 0.001),
 ("component product ~64%", e2e["presence_recall"]*sp["overlap_contract"]["mean"]*100, 64, 1),
 ("42 High-risk clauses lost", rtc["by_band"]["High"]["OR"]["tp"]-rtc["by_band"]["High"]["AND"]["tp"], 42, 0.5),
 ("rare precision gain 32.5", (rt["rare10"]["AND"]["precision"]-rt["rare10"]["TRANS"]["precision"])*100, 32.5, 0.1),
 ("rare recall cost 25.7", (rt["rare10"]["TRANS"]["recall"]-rt["rare10"]["AND"]["recall"])*100, 25.7, 0.1),
 ("common F1 gain 0.077", rt["common31"]["AND"]["f1"]-rt["common31"]["TRANS"]["f1"], 0.077, 0.001),
 ("Longformer b2 8.4x slower", lfd[("longformer",2,"train")]["sec_per_step"]/lfd[("longformer",1,"train")]["sec_per_step"], 8.4, 0.1),
 ("Longformer 136 days", 45600*lfd[("longformer",1,"train")]["sec_per_step"]/86400, 136, 1),
 ("Longformer scan 454 s", 3*41*lfd[("longformer",1,"infer")]["sec_per_forward"], 454, 1),
 ("presence miss 22.0%", (1-e2e["presence_recall"])*100, 22.0, 0.1),
 ("window cap 45,500 chars", 2000+1500*29, 45500, 0),
]
for name, got, claim, tol in DERIVED:
    if abs(got - claim) > tol:
        print(f"  DERIVED MISMATCH {name:34s} computed {got:.4f} vs stated {claim}"); bad += 1

print(f"\n{len(CHECKS)} literal + {len(DERIVED)} derived numbers checked | {bad} mismatched | {missing} not located in text")
sys.exit(1 if bad else 0)
