"""
Live SSE probe: POST /chat against a running nurix-agent (local or deployed) and
report the full ordered event sequence, the observed chart_index/chart_total values,
and the window.CHART_DATA rows parsed out of each chart.

Verification aid, not a unit test — it needs a live endpoint.

Usage:
  uv run python tests/probe_sse.py <base_url> <question> [--deep] [--token <tok>]
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


def main():
    args = [a for a in sys.argv[1:]]
    deep = "--deep" in args
    args = [a for a in args if a != "--deep"]
    token = None
    if "--token" in args:
        i = args.index("--token")
        token = args[i + 1]
        args = args[:i] + args[i + 2:]
    base, question = args[0].rstrip("/"), args[1]

    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

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
                          f"sql={'YES' if ev.get('sql') else 'NO':<3} "
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
