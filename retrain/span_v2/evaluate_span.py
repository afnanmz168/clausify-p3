"""Choose and test the retrained span model (span_chunks.py), and measure the app's quoted paragraph.

Every clause scored here is a gold clause whose category the presence step detected, read from the
window the presence model chose (its highest-scoring window), exactly as in the end-to-end test of
the report (retrain/v2_fullwindow/downstream_v2.py).

1. Validation (the 81 contracts held out from training). For the clauses the app's default setting
   (recall-first transformer) detects, score
     * the first span model, exactly as in the report (chunks at 0 and 800, 320 tokens, best
       start + end logit, at most 60 tokens);
     * the new model after each epoch, decoded four ways: the best start + end logit, or that minus
       the "no answer" score of the chunk, with answers of at most 60 or 120 tokens.
   The epoch and decoding with the most extractions at token-F1 >= 0.5 are chosen. The app's quote
   is chosen here as well: the paragraph the presence model scores highest (the current app), or
   the paragraph that contains the new model's answer.
2. Test (102 contracts), scored once with those choices: end-to-end survival for the app setting
   and the balanced setting, where the clause sits in the window, a "right window" test, and the
   app's quote.

Writes runs/chunks/evaluation.json.

    cd ~/Desktop/"final project p3" && python3 retrain/span_v2/evaluate_span.py
"""
import os, sys, re, json, string, glob
from collections import Counter
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
APP = os.path.join(os.path.dirname(P), "final project app p3")
sys.path.insert(0, APP)
import model_utils as mu                                   # the app's own paragraph and quote code
DEV = os.environ.get("SPAN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
mu.DEVICE = DEV
RUN = os.path.join(HERE, "runs", "chunks")
L512 = os.path.join(P, "retrain", "v2_fullwindow", "runs", "L512")
SMOKE = "--smoke" in sys.argv

D = common.load_all(); cats = D["cats"]; C = len(cats)
sp = json.load(open(os.path.join(L512, "val_split.json"))); val_t, test_t = sp["val"], sp["test"]
dec = json.load(open(os.path.join(L512, "decision.json")))["objectives"]
wins = {t: common.make_windows(D["ctx"][t]) for t in val_t + test_t}

# ------------------------------------------------------------ detected gold clauses and their window
z = np.load(os.path.join(L512, "window_scores.npz")); titles = list(z["titles"])
df = pd.DataFrame({"t": z["title_idx"], "c": z["cat_idx"], "w": z["win_idx"], "p": z["p"]})
best = df.loc[df.groupby(["t", "c"]).p.idxmax()]
B = np.zeros((len(titles), C)); W = np.zeros((len(titles), C), dtype=int)
B[best.t.values, best.c.values] = best.p.values; W[best.t.values, best.c.values] = best.w.values
thr_f2 = np.array([dec["f2"]["t_transformer"][c] for c in cats])


def gold_of(t, c):
    return D["gold_spans"].get((t, c), [])


def detected_rows(ts):
    """Gold clauses the recall-first transformer detects, with the window it chose."""
    rows = []
    for i, t in enumerate(ts):
        ti = titles.index(t)
        for j, c in enumerate(cats):
            if gold_of(t, c) and B[ti, j] >= thr_f2[j]:
                rows.append(dict(i=i, title=t, category=c, window=int(W[ti, j])))
    return rows


val_rows = detected_rows(val_t)
test_rows = {n: json.load(open(os.path.join(L512, f"e2e_rows_{n}.json"))) for n in ("RECALL", "BALANCED")}
assert len(detected_rows(test_t)) == len(test_rows["RECALL"]) == 1253     # same clauses as the report
n_gold_val = sum(1 for t in val_t for c in cats if gold_of(t, c))
n_gold_test = sum(1 for t in test_t for c in cats if gold_of(t, c))
if SMOKE:
    val_rows, test_rows = val_rows[:40], {k: v[:40] for k, v in test_rows.items()}

# ------------------------------------------------------------ scoring helpers (as downstream_v2.py)
norm = lambda s: " ".join("".join(ch for ch in re.sub(r"\b(a|an|the)\b", " ", s.lower())
                                  if ch not in string.punctuation).split())


def token_f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    if not p or not g:
        return float(p == g)
    n = sum((Counter(p) & Counter(g)).values())
    return 0.0 if n == 0 else 2 * (n / len(p)) * (n / len(g)) / (n / len(p) + n / len(g))


def coverage(q, g):
    """Share of the gold clause's words that appear in the quote."""
    q, g = Counter(norm(q).split()), Counter(norm(g).split())
    return sum((q & g).values()) / max(1, sum(g.values()))


def clause_start(t, c, k):
    w = wins[t][k]; p = [w.find(g[:80]) for g in gold_of(t, c) if g[:80] in w]
    return min(p) if p else -1


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def old_extract(q, w, _cache={}):
    """The first span model exactly as scored in the report (downstream_v2.extract)."""
    if "m" not in _cache:
        d = os.path.join(common.NB, "outputs", "span", "final")
        _cache["t"] = AutoTokenizer.from_pretrained(d)
        _cache["m"] = AutoModelForQuestionAnswering.from_pretrained(d).to(DEV).eval()
    qt, qa = _cache["t"], _cache["m"]
    chunks = [w[s:s + 1200] for s in range(0, max(1, len(w) - 1200 + 1), 800)] or [w]
    e = qt([q] * len(chunks), chunks, truncation="only_second", max_length=320, padding=True, return_tensors="pt").to(DEV)
    with torch.no_grad():
        o = qa(**e)
    sl, el, ids = o.start_logits.cpu().numpy(), o.end_logits.cpu().numpy(), e["input_ids"].cpu().numpy()
    bc, bt = -1e18, ""
    for k in range(len(chunks)):
        row = list(ids[k]); L = int(e["attention_mask"][k].sum()); sep = row.index(qt.sep_token_id)
        m = np.full(len(row), -1e9); m[sep + 1:L] = 0.0
        s_, e_ = int(np.argmax(sl[k] + m)), int(np.argmax(el[k] + m)); e_ = max(e_, s_); e_ = min(e_, s_ + 60)
        if float(sl[k][s_] + el[k][e_]) > bc:
            bc, bt = float(sl[k][s_] + el[k][e_]), qt.decode(row[s_:e_ + 1], skip_special_tokens=True)
    return bt


CHUNK, STRIDE = 1200, 800
DECODINGS = [(rule, cap) for rule in ("raw", "minus_null") for cap in (60, 120)]


def chunk_starts(n):
    s = list(range(0, max(1, n - CHUNK + 1), STRIDE))
    if s[-1] + CHUNK < n:
        s.append(n - CHUNK)
    return s


def new_extract(model, q, w, decodings=DECODINGS):
    """New model: same chunks as in training, 384 tokens. Returns {decoding: (text, start, end)},
    start/end as character offsets in the window, so the answer can be quoted in its original form."""
    cs = chunk_starts(len(w)); chunks = [w[s:s + CHUNK] for s in cs]
    e = tok([q] * len(chunks), chunks, truncation="only_second", max_length=384, padding=True,
            return_offsets_mapping=True, return_tensors="pt")
    off = e.pop("offset_mapping").numpy()
    with torch.no_grad():
        o = model(**e.to(DEV))
    sl, el = o.start_logits.cpu().numpy(), o.end_logits.cpu().numpy()
    out = {}
    for rule, cap in decodings:
        best = (-1e18, "", -1, -1)
        for k in range(len(chunks)):
            seq = e.sequence_ids(k); ctx = np.array([s == 1 for s in seq])
            m = np.where(ctx, 0.0, -1e9)
            s_ = int(np.argmax(sl[k] + m))
            ends = np.arange(len(seq)); ok = ctx & (ends >= s_) & (ends <= s_ + cap)
            e_ = int(np.argmax(np.where(ok, el[k], -1e18)))
            sc = float(sl[k][s_] + el[k][e_]) - (float(sl[k][0] + el[k][0]) if rule == "minus_null" else 0.0)
            if sc > best[0]:
                a, b = int(off[k][s_][0]) + cs[k], int(off[k][e_][1]) + cs[k]
                best = (sc, w[a:b], a, b)
        out[(rule, cap)] = best[1:]
    return out


def para_of_span(t, k, a):
    """The app's paragraph (same candidates as mu._paragraphs) that holds window offset a."""
    w = wins[t][k]; paras = mu._paragraphs(w); pos = [w.find(p) for p in paras]
    hit = [p for p, s in zip(paras, pos) if 0 <= s <= a < s + len(p)]
    if not hit:                                                   # answer in a dropped short piece
        hit = [min(zip(paras, pos), key=lambda x: abs(x[1] - a))[0]]
    return mu._full_paragraph(D["ctx"][t], hit[0])[0]


def app_quote(t, c, k):
    """The current app: the presence model picks the paragraph, then it is widened to the whole
    paragraph of the contract (model_utils.locate_clause and _full_paragraph)."""
    para, _ = mu.locate_clause(c, wins[t][k], "v2")
    return mu._full_paragraph(D["ctx"][t], para)[0]


def best_over_gold(fn, t, c):
    return max(fn(g) for g in gold_of(t, c))


def summarize(rows, key, n_gold, ts, ok_key=None):
    """Survival and breakdowns for rows that carry a per-clause success flag `key`."""
    r = pd.DataFrame(rows); ok = r[key].astype(int)
    right = r[r.clause_start >= 0]; first, second = right[right.clause_start < 1000], right[right.clause_start >= 1000]
    per = np.zeros(len(ts)); tot = np.zeros(len(ts))
    for t in ts:
        tot[ts.index(t)] = sum(1 for c in cats if gold_of(t, c))
    for x, o in zip(rows, ok):
        per[ts.index(x["title"])] += o
    rng = np.random.default_rng(42); bs = []
    for _ in range(10000):
        b = rng.integers(0, len(ts), len(ts)); bs.append(per[b].sum() / tot[b].sum())
    return dict(n_detected=len(r), n_ok=int(ok.sum()), share_of_detected=float(ok.mean()),
                end_to_end=float(ok.sum() / n_gold),
                end_to_end_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                right_window=int(len(right)), ok_given_right_window=float(right[key].mean()),
                ok_first_half=float(first[key].mean()), ok_second_half=float(second[key].mean()),
                n_first_half=int(len(first)), n_second_half=int(len(second)))


out = {"n_val_contracts": len(val_t), "n_test_contracts": len(test_t), "n_gold_val": n_gold_val,
       "n_gold_test": n_gold_test}

# ============================================================ 1. validation: choose
for x in val_rows:
    x["clause_start"] = clause_start(x["title"], x["category"], x["window"])
q_of = lambda c: D["cat_question"][c]
for x in val_rows:
    t, c, k = x["title"], x["category"], x["window"]
    x["old_ok"] = int(best_over_gold(lambda g: token_f1(old_extract(q_of(c), wins[t][k]), g), t, c) >= 0.5)
ckpts = sorted(glob.glob(os.path.join(RUN, "ckpt", "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
if SMOKE:                                    # plumbing check only: stand in the first model for a checkpoint
    ckpts = [os.path.join(common.NB, "outputs", "span", "final")]
val_grid = {}
for ep, ck in enumerate(ckpts, 1):
    model = AutoModelForQuestionAnswering.from_pretrained(ck).to(DEV).eval()
    for x in val_rows:
        t, c, k = x["title"], x["category"], x["window"]
        res = new_extract(model, q_of(c), wins[t][k])
        for d, (txt, a, b) in res.items():
            x[f"e{ep}_{d[0]}_{d[1]}"] = int(best_over_gold(lambda g: token_f1(txt, g), t, c) >= 0.5)
            x[f"e{ep}_{d[0]}_{d[1]}_a"] = a
    for d in DECODINGS:
        val_grid[f"epoch{ep}_{d[0]}_cap{d[1]}"] = float(np.mean([x[f"e{ep}_{d[0]}_{d[1]}"] for x in val_rows]))
    del model
    if DEV == "mps":
        torch.mps.empty_cache()
choice = max(val_grid, key=val_grid.get)
ep = int(choice.split("_")[0][5:]); rule = "minus_null" if "minus_null" in choice else "raw"; cap = int(choice[-3:].lstrip("p"))
key = f"e{ep}_{rule}_{cap}"
print("validation grid:", {k: round(v, 4) for k, v in val_grid.items()}, "-> chosen", choice, flush=True)

# the app's quote on validation: presence-chosen paragraph vs paragraph around the new answer
for x in val_rows:
    t, c, k = x["title"], x["category"], x["window"]
    qa_, qb = app_quote(t, c, k), para_of_span(t, k, x[key + "_a"])
    x["quote_presence_cov"] = int(best_over_gold(lambda g: coverage(qa_, g), t, c) >= 0.5)
    x["quote_span_cov"] = int(best_over_gold(lambda g: coverage(qb, g), t, c) >= 0.5)
quote_choice = ("span_paragraph" if np.mean([x["quote_span_cov"] for x in val_rows])
                > np.mean([x["quote_presence_cov"] for x in val_rows]) else "presence_paragraph")
out["validation"] = dict(
    n_detected=len(val_rows), old_model_ok=float(np.mean([x["old_ok"] for x in val_rows])), grid=val_grid,
    chosen=dict(epoch=ep, rule=rule, max_answer_tokens=cap, checkpoint=os.path.basename(ckpts[ep - 1])),
    quote_cov_presence_paragraph=float(np.mean([x["quote_presence_cov"] for x in val_rows])),
    quote_cov_span_paragraph=float(np.mean([x["quote_span_cov"] for x in val_rows])), quote_chosen=quote_choice)
print("validation:", {k: v for k, v in out["validation"].items() if k != "grid"}, flush=True)

# ============================================================ 2. test: scored once
model = AutoModelForQuestionAnswering.from_pretrained(ckpts[ep - 1]).to(DEV).eval()
dsel = [(rule, cap)]
out["test"] = {}
for name, rows in test_rows.items():
    for x in rows:
        t, c, k = x["title"], x["category"], x["window"]
        txt, a, b = new_extract(model, q_of(c), wins[t][k], dsel)[(rule, cap)]
        x["new_f1"] = best_over_gold(lambda g: token_f1(txt, g), t, c); x["new_ok"] = int(x["new_f1"] >= 0.5)
        x["old_ok"] = int(x["span_ok"]); x["new_a"] = a
    n = len(test_t)
    out["test"][name] = dict(first_model=summarize(rows, "old_ok", n_gold_test, test_t),
                             new_model=summarize(rows, "new_ok", n_gold_test, test_t),
                             mean_token_f1_first=float(np.mean([x["token_f1"] for x in rows])),
                             mean_token_f1_new=float(np.mean([x["new_f1"] for x in rows])))
    print(name, {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["test"][name][m].items()
                     if k in ("n_ok", "end_to_end", "ok_given_right_window", "ok_first_half", "ok_second_half")}
                 for m in ("first_model", "new_model")}, flush=True)

# the app's quote on the test set (app setting)
rows = test_rows["RECALL"]
for x in rows:
    t, c, k = x["title"], x["category"], x["window"]
    qa_, qb = app_quote(t, c, k), para_of_span(t, k, x["new_a"])
    for tag, q in (("presence", qa_), ("span", qb)):
        x[f"q_{tag}_cov"] = int(best_over_gold(lambda g: coverage(q, g), t, c) >= 0.5)
        x[f"q_{tag}_f1"] = int(best_over_gold(lambda g: token_f1(q, g), t, c) >= 0.5)
        x[f"q_{tag}_len"] = len(q)
out["test"]["app_quote"] = {
    tag: dict(covers_half_of_clause=summarize(rows, f"q_{tag}_cov", n_gold_test, test_t),
              token_f1_ok=summarize(rows, f"q_{tag}_f1", n_gold_test, test_t),
              median_quote_chars=float(np.median([x[f"q_{tag}_len"] for x in rows])))
    for tag in ("presence", "span")}
print("app quote:", {tag: (round(v["covers_half_of_clause"]["share_of_detected"], 4),
                           round(v["token_f1_ok"]["share_of_detected"], 4), v["median_quote_chars"])
                     for tag, v in out["test"]["app_quote"].items()}, flush=True)

# a window that contains the clause (the first one holding its start), clause anywhere in it
iso = []
for t in test_t:
    for c in cats:
        gs = gold_of(t, c)
        if not gs:
            continue
        k = next((k for k, w in enumerate(wins[t]) if any(g[:80] in w for g in gs)), None)
        if k is None:
            continue
        w = wins[t][k]; txt = new_extract(model, q_of(c), w, dsel)[(rule, cap)][0]
        iso.append(dict(new=best_over_gold(lambda g: token_f1(txt, g), t, c),
                        old=best_over_gold(lambda g: token_f1(old_extract(q_of(c), w), g), t, c)))
        if SMOKE and len(iso) >= 30:
            break
iso = pd.DataFrame(iso)
out["test"]["right_window"] = dict(n=len(iso), first_model_mean_f1=float(iso.old.mean()),
                                   first_model_ok=float((iso.old >= 0.5).mean()),
                                   new_model_mean_f1=float(iso.new.mean()), new_model_ok=float((iso.new >= 0.5).mean()))
print("right window:", out["test"]["right_window"], flush=True)
for name, rows in test_rows.items():
    pd.DataFrame(rows).to_json(os.path.join(RUN, f"test_rows_{name}.json"), orient="records")
json.dump(out, open(os.path.join(RUN, "evaluation_smoke.json" if SMOKE else "evaluation.json"), "w"), indent=1)
print("wrote", os.path.join(RUN, "evaluation.json"))
