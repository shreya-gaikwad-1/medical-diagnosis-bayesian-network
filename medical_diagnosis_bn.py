"""
Advanced AI Mini Project: Medical Diagnosis with Bayesian Networks + Decision Theory
Units covered: 1-2 (Bayes nets, Variable Elimination), 3 (Sampling), 5 (Expected Utility, VOI)
Pure Python, no external libraries.
"""
import random
from itertools import product

# ---------- 1. Bayesian Network (all variables binary: 0/1) ----------
# name: (parents, {parent_values: P(var=1 | parents)}). Order is topological.
BN = {
    "Smoker":    ((), {(): 0.30}),
    "Pollution": ((), {(): 0.20}),
    "Cancer":    (("Smoker", "Pollution"),
                  {(0, 0): 0.001, (0, 1): 0.02, (1, 0): 0.03, (1, 1): 0.05}),
    "XRay":      (("Cancer",), {(0,): 0.05, (1,): 0.90}),
    "Dyspnea":   (("Cancer",), {(0,): 0.20, (1,): 0.65}),
}

# ---------- 2. Exact inference: Variable Elimination ----------
# A factor is (variables_tuple, {assignment_tuple: value})

def make_factor(var):
    parents, cpt = BN[var]
    vs = parents + (var,)
    table = {}
    for vals in product([0, 1], repeat=len(vs)):
        p1 = cpt[vals[:-1]]
        table[vals] = p1 if vals[-1] == 1 else 1 - p1
    return vs, table

def restrict(f, var, val):
    vs, t = f
    if var not in vs:
        return f
    i = vs.index(var)
    return (vs[:i] + vs[i + 1:],
            {k[:i] + k[i + 1:]: v for k, v in t.items() if k[i] == val})

def multiply(f, g):
    fv, ft = f
    gv, gt = g
    vs = fv + tuple(v for v in gv if v not in fv)
    table = {}
    for vals in product([0, 1], repeat=len(vs)):
        a = dict(zip(vs, vals))
        table[vals] = ft[tuple(a[v] for v in fv)] * gt[tuple(a[v] for v in gv)]
    return vs, table

def sum_out(f, var):
    vs, t = f
    i = vs.index(var)
    new = {}
    for k, v in t.items():
        key = k[:i] + k[i + 1:]
        new[key] = new.get(key, 0) + v
    return vs[:i] + vs[i + 1:], new

def variable_elimination(query, evidence):
    """Return P(query | evidence) as {0: p0, 1: p1}."""
    factors = [make_factor(v) for v in BN]
    for var, val in evidence.items():
        factors = [restrict(f, var, val) for f in factors]
    hidden = [v for v in BN if v != query and v not in evidence]
    for h in hidden:
        related = [f for f in factors if h in f[0]]
        factors = [f for f in factors if h not in f[0]]
        prod = related[0]
        for f in related[1:]:
            prod = multiply(prod, f)
        factors.append(sum_out(prod, h))
    result = factors[0]
    for f in factors[1:]:
        result = multiply(result, f)
    vs, t = result
    z = sum(t.values())
    return {val: t[(val,)] / z for val in (0, 1)}

# ---------- 3. Approximate inference: Likelihood Weighting ----------

def likelihood_weighting(query, evidence, n=20000, seed=0):
    rng = random.Random(seed)
    w_total = {0: 0.0, 1: 0.0}
    for _ in range(n):
        sample, w = {}, 1.0
        for var, (parents, cpt) in BN.items():
            p1 = cpt[tuple(sample[p] for p in parents)]
            if var in evidence:
                sample[var] = evidence[var]
                w *= p1 if evidence[var] == 1 else 1 - p1
            else:
                sample[var] = 1 if rng.random() < p1 else 0
        w_total[sample[query]] += w
    z = sum(w_total.values())
    return {k: v / z for k, v in w_total.items()}

# ---------- 4. Decision theory: Expected Utility & Value of Information ----------
ACTIONS = ["Treat", "Don't treat"]
# U(action, has_cancer)
UTILITY = {("Treat", 1): 80, ("Treat", 0): -20,
           ("Don't treat", 1): -100, ("Don't treat", 0): 0}

def expected_utility(action, evidence):
    p = variable_elimination("Cancer", evidence)
    return sum(p[c] * UTILITY[(action, c)] for c in (0, 1))

def best_action(evidence):
    eus = {a: expected_utility(a, evidence) for a in ACTIONS}
    a = max(eus, key=eus.get)
    return a, eus[a], eus

def value_of_information(test_var, evidence):
    """EVSI of observing test_var before deciding."""
    _, meu_now, _ = best_action(evidence)
    p_test = variable_elimination(test_var, evidence)
    meu_with = 0.0
    for x in (0, 1):
        ev = dict(evidence, **{test_var: x})
        meu_with += p_test[x] * best_action(ev)[1]
    return meu_with - meu_now

# ---------- 5. Demo ----------

def main():
    scenarios = {
        "Smoker, no other info": {"Smoker": 1},
        "Smoker with Dyspnea": {"Smoker": 1, "Dyspnea": 1},
        "Smoker, Dyspnea, positive X-Ray": {"Smoker": 1, "Dyspnea": 1, "XRay": 1},
        "Non-smoker, negative X-Ray": {"Smoker": 0, "XRay": 0},
    }

    print("=" * 66)
    print("PART A/B: P(Cancer=1 | evidence)  -  Exact vs Likelihood Weighting")
    print("=" * 66)
    print(f"{'Scenario':36s}{'Exact':>9s}{'Approx':>9s}{'Error':>9s}")
    for name, ev in scenarios.items():
        ex = variable_elimination("Cancer", ev)[1]
        ap = likelihood_weighting("Cancer", ev)[1]
        print(f"{name:36s}{ex:9.4f}{ap:9.4f}{abs(ex - ap):9.4f}")

    print("\nConvergence (Smoker with Dyspnea): error vs number of samples")
    ev = scenarios["Smoker with Dyspnea"]
    ex = variable_elimination("Cancer", ev)[1]
    for n in (100, 1000, 10000, 100000):
        ap = likelihood_weighting("Cancer", ev, n=n)[1]
        print(f"  n={n:>6d}  approx={ap:.4f}  error={abs(ap - ex):.4f}")

    print("\n" + "=" * 66)
    print("PART C: Decision making (maximize expected utility)")
    print("=" * 66)
    for name, ev in scenarios.items():
        a, eu, eus = best_action(ev)
        print(f"{name:36s} -> {a:12s} (EU={eu:7.2f})")

    print("\nValue of ordering an X-Ray before deciding:")
    for name in ("Smoker, no other info", "Smoker with Dyspnea"):
        v = value_of_information("XRay", scenarios[name])
        print(f"  {name:28s} VOI = {v:6.2f}")

if __name__ == "__main__":
    main()
