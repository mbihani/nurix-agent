"""
Live SSE probe: POST /chat against a running nurix-agent (local or deployed) and
report the full ordered event sequence, the observed chart_index/chart_total values,
and the window.CHART_DATA rows parsed out of each chart.

Verification aid, not a unit test — it needs a live endpoint.

Usage:
  uv run python tests/probe_sse.py <base_url> <question> [--deep] [--token <tok>]
  uv run python tests/probe_sse.py <base_url> <instruction> --refine [--token <tok>]

--refine first runs a plain /chat to obtain a real chart + its SQL, then POSTs both
to /refine so the refined chart event can be inspected on the wire.
"""
import json
import re
import sys
import time

import httpx

_DATA_RE = re.compile(r"window\.CHART_DATA\s*=\s*(\{.*?\});</script>", re.DOTALL)


def parse_chart_data(html: str):
    """Pull the injected window.CHART_DATA payload back out of a chart's HTML."""
    m = _DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def sentiment_totals(payload):
    """Aggregate {sentiment -> total} from a CHART_DATA payload, or None."""
    if not payload:
        return None
    # `columns` are OBJECTS {"name","type"}, not bare strings — index by ['name'].
    names = [
        (c.get("name", "") if isinstance(c, dict) else str(c)).lower()
        for c in payload.get("columns", [])
    ]
    rows = payload.get("rows", [])
    sent_i = next((i for i, n in enumerate(names) if "sentiment" in n), None)
    cnt_i = next((i for i, n in enumerate(names)
                  if "count" in n or "review" in n or "num" in n or "total" in n), None)
    if sent_i is None or cnt_i is None:
        return None
    totals = {}
    for r in rows:
        try:
            totals[str(r[sent_i]).lower()] = totals.get(str(r[sent_i]).lower(), 0) + int(r[cnt_i])
        except (ValueError, TypeError, IndexError):
            return None
    return totals


def stream(base, path, payload, headers, quiet=False):
    """POST one SSE request and return (events, charts, counts, seq, elapsed)."""
    t0 = time.time()
    charts, seq, counts, events = [], [], {}, []
    with httpx.Client(timeout=httpx.Timeout(420.0), follow_redirects=True) as client:
        with client.stream("POST", f"{base}{path}", json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {resp.read()[:600]!r}")
                return None, None, None, None, None
            buf = []
            for line in resp.iter_lines():
                line = line.rstrip("\r\n")
                if line.startswith("data:"):
                    buf.append(line[5:].lstrip(" "))
                    continue
                if line or not buf:
                    continue
                try:
                    ev = json.loads("\n".join(buf))
                except json.JSONDecodeError:
                    buf = []
                    continue
                buf = []
                events.append(ev)
                t = ev.get("type", "?")
                counts[t] = counts.get(t, 0) + 1
                el = time.time() - t0
                seq.append((round(el, 1), t))
                if t == "chart":
                    charts.append(ev)
                if not quiet:
                    if t == "chart":
                        print(f"  +{el:6.1f}s  chart      "
                              f"chart_index={ev.get('chart_index')!r} "
                              f"chart_total={ev.get('chart_total')!r} "
                              f"index={ev.get('index')!r} total={ev.get('total')!r} "
                              f"sql={'YES' if ev.get('sql') else 'MISSING':<7} "
                              f"html={len(ev.get('html') or '')}B")
                    elif t == "sql":
                        print(f"  +{el:6.1f}s  sql        chart_index={ev.get('chart_index')!r} "
                              f"index={ev.get('index')!r}  {(ev.get('sql') or '')[:80]!r}")
                    else:
                        print(f"  +{el:6.1f}s  {t:<10} "
                              f"{str(ev.get('text') or ev.get('message') or '')[:150]}")
                if t in ("done", "error", "rejected"):
                    break
    return events, charts, counts, seq, time.time() - t0


def refine_probe(base, instruction, headers):
    """
    Exercise /refine end to end: get a real chart + SQL from /chat, then refine it.

    Verifies the refined chart event carries chart_index/chart_total AND a real,
    non-empty sql — the path WARNING 1 was about.
    """
    print("STEP 1 — /chat to obtain a real chart and its SQL\n")
    _, charts, _, _, _ = stream(
        base, "/chat",
        {"question": "Show me the sentiment breakdown across all reviews",
         "session_id": "refine-probe", "deep_research": False},
        headers,
    )
    if not charts:
        print("REFINE FAIL: no seed chart obtained")
        return 1
    seed = charts[0]
    seed_sql = seed.get("sql") or ""
    print(f"\nseed chart: chart_index={seed.get('chart_index')} "
          f"chart_total={seed.get('chart_total')} sql_len={len(seed_sql)}")
    if not seed_sql.strip():
        print("REFINE FAIL: seed chart carried no sql, cannot test the echo")
        return 1

    print(f"\nSTEP 2 — /refine  instruction={instruction!r}\n")
    _, rcharts, counts, seq, elapsed = stream(
        base, "/refine",
        {"chart_html": seed["html"], "instruction": instruction,
         "session_id": "refine-probe", "sql": seed_sql},
        headers,
    )
    print(f"\n{'=' * 72}\nwall clock: {elapsed:.1f}s\nevent counts: {counts}")
    print(f"ordered sequence: {[t for _, t in seq]}")
    if not rcharts:
        print("REFINE FAIL: no chart event from /refine")
        return 1
    rc = rcharts[0]
    print(f"refined chart_index={rc.get('chart_index')!r} chart_total={rc.get('chart_total')!r} "
          f"index={rc.get('index')!r} total={rc.get('total')!r}")
    print(f"refined sql present={bool(rc.get('sql'))} matches_seed={rc.get('sql') == seed_sql}")
    payload = parse_chart_data(rc.get("html") or "")
    print(f"refined chart data preserved: rows={len(payload.get('rows', [])) if payload else None} "
          f"sentiment_totals={sentiment_totals(payload)}")
    ok = (rc.get("chart_index") == 0 and rc.get("chart_total") == 1
          and rc.get("index") == 0 and rc.get("total") == 1
          and bool((rc.get("sql") or "").strip()) and rc.get("sql") == seed_sql)
    print("REFINE " + ("PASS: aliases present and sql echoed intact" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    args = [a for a in sys.argv[1:]]
    deep = "--deep" in args
    is_refine = "--refine" in args
    args = [a for a in args if a not in ("--deep", "--refine")]
    token = None
    if "--token" in args:
        i = args.index("--token")
        token = args[i + 1]
        args = args[:i] + args[i + 2:]
    base, question = args[0].rstrip("/"), args[1]

    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if is_refine:
        return refine_probe(base, question, headers)

    print(f"POST {base}/chat  deep_research={deep}\nquestion={question!r}\n")
    t0 = time.time()
    charts, seq, counts = [], [], {}

    with httpx.Client(timeout=httpx.Timeout(420.0), follow_redirects=True) as client:
        with client.stream(
            "POST", f"{base}/chat",
            json={"question": question, "session_id": "probe", "deep_research": deep},
            headers=headers,
        ) as resp:
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {resp.read()[:600]!r}")
                return 1
            buf = []
            for line in resp.iter_lines():
                line = line.rstrip("\r\n")
                if line.startswith("data:"):
                    buf.append(line[5:].lstrip(" "))
                    continue
                if line or not buf:
                    continue
                try:
                    ev = json.loads("\n".join(buf))
                except json.JSONDecodeError:
                    buf = []
                    continue
                buf = []
                t = ev.get("type", "?")
                counts[t] = counts.get(t, 0) + 1
                el = time.time() - t0
                seq.append((round(el, 1), t))
                if t == "chart":
                    charts.append(ev)
                    print(f"  +{el:6.1f}s  chart      "
                          f"chart_index={ev.get('chart_index')!r} "
                          f"chart_total={ev.get('chart_total')!r} "
                          f"index={ev.get('index')!r} total={ev.get('total')!r} "
                          f"sql={'YES' if ev.get('sql') else 'MISSING':<7} "
                          f"html={len(ev.get('html') or '')}B")
                elif t == "sql":
                    print(f"  +{el:6.1f}s  sql        chart_index={ev.get('chart_index')!r} "
                          f"index={ev.get('index')!r}  {(ev.get('sql') or '')[:80]!r}")
                else:
                    print(f"  +{el:6.1f}s  {t:<10} {str(ev.get('text') or ev.get('message') or '')[:150]}")
                if t in ("done", "error", "rejected"):
                    break

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}\nwall clock: {elapsed:.1f}s")
    print(f"event counts: {counts}")
    print(f"ordered sequence: {[t for _, t in seq]}")
    print(f"observed chart_index: {[c.get('chart_index') for c in charts]}")
    print(f"observed chart_total: {[c.get('chart_total') for c in charts]}")
    print(f"observed legacy index/total: "
          f"{[(c.get('index'), c.get('total')) for c in charts]}")

    for c in charts:
        payload = parse_chart_data(c.get("html") or "")
        n_rows = len(payload.get("rows", [])) if payload else "NO CHART_DATA"
        cols = [cc.get("name") for cc in payload.get("columns", [])] if payload else []
        tot = sentiment_totals(payload)
        print(f"  chart {c.get('chart_index')}: rows={n_rows} columns={cols}"
              + (f" sentiment_totals={tot}" if tot else ""))

    # Contract checks the reviewer asked to see on the wire.
    idxs = sorted(c.get("chart_index") for c in charts)
    n_sql, n_chart = counts.get("sql", 0), counts.get("chart", 0)
    print(f"\nCONTRACT sql=={n_sql} chart=={n_chart} equal={n_sql == n_chart}")
    print(f"CONTRACT dense 0..{len(charts) - 1}: {idxs == list(range(len(charts)))}")
    print(f"CONTRACT chart_total==count: "
          f"{all(c.get('chart_total') == len(charts) for c in charts)}")
    print(f"CONTRACT every chart has non-empty sql: "
          f"{all((c.get('sql') or '').strip() for c in charts)}")
    if not deep:
        # A duplicate sql event on the plain path was a real regression once; pin it.
        print(f"CONTRACT plain-path sql event count == 1: {n_sql == 1} (got {n_sql})")
    # No column that reads as an identifier/calendar part may be charted as a measure.
    suspects = ("_id", "id", "zip", "postal", "phone", "year", "month", "uuid", "sku")
    for c in charts:
        payload = parse_chart_data(c.get("html") or "")
        names = [(cc.get("name") or "").lower() for cc in (payload or {}).get("columns", [])]
        flagged = [n for n in names if n in suspects or n.endswith("_id")]
        if flagged:
            print(f"  NOTE chart {c.get('chart_index')} carries id/date-like columns "
                  f"{flagged} — check they are dimensions, not the plotted measure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
