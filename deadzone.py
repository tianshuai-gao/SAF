import json, statistics

LANGS = ["bo", "ug", "mn", "kk"]
TASKS = ["reading_comprehension", "response_selection"]

def analyze(task, lang):
    recs = json.load(open(f"records_{task}_{lang}.json"))
    n = len(recs)
    fixed_ins = sum(r["ok_ins"] for r in recs) / n
    fixed_cpt = sum(r["ok_cpt"] for r in recs) / n
    best_fixed = max(fixed_ins, fixed_cpt)
    default_host = "ins" if fixed_ins >= fixed_cpt else "cpt"

    # 中心校准: 减掉这一组 d 的 median
    diffs = [r["e_ins"] - r["e_cpt"] for r in recs]
    med = statistics.median(diffs)

    row = {}
    for thr in [0.0, 0.02, 0.03, 0.05, 0.08, 0.10]:
        correct = 0
        for r in recs:
            d = (r["e_ins"] - r["e_cpt"]) - med   # 校准后
            if d > thr:      use = "ins"
            elif -d > thr:   use = "cpt"
            else:            use = default_host
            correct += r["ok_ins"] if use == "ins" else r["ok_cpt"]
        row[thr] = correct / n - best_fixed


    return best_fixed, default_host, med, row

thrs = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10]
print("中心校准死区: d 先减本组 median, 再比阈值")
print("vs_fixed (>=0 = 不下降). 每格 acc-best_fixed")
print(f"{'task/lang':<26}{'def':>4}{'med':>8} " + "".join(f"{t:>8.2f}" for t in thrs))
print("-" * 88)
worst = {t: 1.0 for t in thrs}
for task in TASKS:
    for lang in LANGS:
        bf, dh, med, row = analyze(task, lang)
        cells = ""
        for t in thrs:
            v = row[t]
            mark = "" if v >= -1e-9 else "*"
            cells += f"{v:>+7.3f}{mark}"
            worst[t] = min(worst[t], v)
        print(f"{task[:18]+'/'+lang:<26}{dh:>4}{med:>+8.3f} {cells}")
print("-" * 88)
print(f"{'WORST across all 8':<26}{'':>12} " + "".join(f"{worst[t]:>+8.3f}" for t in thrs))
print("\n* = 该组合下降. 找一列 WORST >= 0 的 thr = 8 组合全不下降的通用阈值")
