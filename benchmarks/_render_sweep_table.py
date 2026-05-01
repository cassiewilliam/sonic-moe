"""Render sweep_compare.sh log lines into a comparison table.

Reads pipe-delimited lines like:
    OLMoE|T=32768|sonic-bf16|5.68ms|517.0TF
    OLMoE|T=32768|sonic-bf16|5.68ms|517.0TF|vmax=1.50e-04|vrel=2.30e-03|verify=PASS
    OLMoE|T=32768|te-bf16|FAIL|OOM
"""

import sys
import re
from collections import defaultdict


LOGFILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep_compare.log"

# parsed[(name, T)][mode] = dict {"t_ms": ..., "tf": ..., "verify": (vmax, vrel, vpass)} or {"fail": reason}
parsed = defaultdict(dict)
order_names = []
order_t = set()
modes_seen = []


def parse_pairs(rest):
    """Parse vmax=..., vrel=..., verify=... from trailing fields."""
    out = {}
    for f in rest:
        if "=" in f:
            k, v = f.split("=", 1)
            out[k] = v
    return out


with open(LOGFILE) as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name, t_str, mode, val, *rest = parts
        try:
            T = int(t_str.split("=")[1])
        except (IndexError, ValueError):
            continue
        if name not in order_names:
            order_names.append(name)
        order_t.add(T)
        if mode not in modes_seen:
            modes_seen.append(mode)

        rec = {}
        if val == "FAIL":
            rec["fail"] = rest[0] if rest else "?"
        else:
            m1 = re.match(r"([\d.]+)ms", val)
            rec["t_ms"] = float(m1.group(1)) if m1 else None
            if rest:
                m2 = re.match(r"([\d.]+)TF", rest[0])
                rec["tf"] = float(m2.group(1)) if m2 else None
                kv = parse_pairs(rest[1:])
                # accept both old (vrel) and new (head) field names
                hkey = "head" if "head" in kv else ("vrel" if "vrel" in kv else None)
                if "vmax" in kv and hkey and "verify" in kv:
                    try:
                        rec["verify"] = (float(kv["vmax"]), float(kv[hkey]), kv["verify"] == "PASS")
                    except ValueError:
                        pass
        parsed[(name, T)][mode] = rec

t_sorted = sorted(order_t)


def fmt_cell(rec, with_verify):
    if rec is None:
        return f"{'-':>20}"
    if "fail" in rec:
        msg = rec["fail"][:14]
        return f"{'⨯ ' + msg:>20}"
    t = rec.get("t_ms"); tf = rec.get("tf")
    base = f"{t:>6.2f}ms · {tf:>5.0f}TF"
    if with_verify and "verify" in rec:
        _, _, vp = rec["verify"]
        base += "  ✓" if vp else "  ✗"
    return f"{base:>20}"


# Determine if any line had verify info
HAS_VERIFY = any("verify" in cells.get(m, {}) for cells in parsed.values() for m in modes_seen)

for name in order_names:
    print(f"\n=== {name} ===")
    header = f"{'T':>7}"
    for m in modes_seen:
        header += f" | {m:>20}"
    if "sonic-bf16" in modes_seen:
        for ref_mode in ("te-bf16", "te-fp16", "te-fp8"):
            if ref_mode in modes_seen:
                header += f" | s/{ref_mode[3:]:<6}"
    print(header)
    print("-" * len(header))

    for T in t_sorted:
        cells = parsed.get((name, T), {})
        if not cells:
            continue
        row = f"{T:>7}"
        for m in modes_seen:
            row += " | " + fmt_cell(cells.get(m), with_verify=HAS_VERIFY)
        sonic = cells.get("sonic-bf16", {})
        sonic_t = sonic.get("t_ms")
        for ref_mode in ("te-bf16", "te-fp16", "te-fp8"):
            if ref_mode in modes_seen:
                te = cells.get(ref_mode, {})
                te_t = te.get("t_ms")
                if sonic_t and te_t:
                    row += f" | {te_t/sonic_t:>6.2f}x"
                else:
                    row += f" | {'-':>7}"
        print(row)

    # Verify summary line per model
    if HAS_VERIFY:
        ver = []
        for T in t_sorted:
            cells = parsed.get((name, T), {})
            for m in modes_seen:
                rec = cells.get(m, {})
                if "verify" in rec:
                    vmax, vrel, vp = rec["verify"]
                    if not vp:
                        ver.append(f"{m}@T={T}: max={vmax:.2e}, rel={vrel:.2e}")
        if ver:
            print("    ⚠ verify FAIL:")
            for v in ver:
                print(f"      {v}")
        else:
            ran = sum(1 for T in t_sorted for m in modes_seen
                      if "verify" in parsed.get((name, T), {}).get(m, {}))
            if ran:
                print(f"    ✓ verify PASS on all {ran} runs")

if HAS_VERIFY:
    print()
    print("Legend:  ✓ = output matches torch reference within tolerance · ✗ = above tolerance")
