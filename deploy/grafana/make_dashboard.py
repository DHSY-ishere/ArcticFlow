"""Regenerates dashboards/arcticflow.json (dark theme). Run: python deploy/grafana/make_dashboard.py"""
import json
import pathlib

DS = {"type": "prometheus", "uid": "prometheus"}
panels = []
_id = [0]


def target(expr, legend="", ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def panel(kind, title, x, y, w, h, targets, unit=None, **extra):
    _id[0] += 1
    p = {"id": _id[0], "type": kind, "title": title, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
         "fieldConfig": {"defaults": {"unit": unit} if unit else {}, "overrides": []},
         "options": {}}
    p.update(extra)
    panels.append(p)
    return p


def stat(title, expr, x, unit, thresholds=None, minv=None, maxv=None):
    d = {"thresholds": {"mode": "absolute", "steps": thresholds or [{"color": "#5794F2", "value": None}]},
         "color": {"mode": "thresholds"}}
    if minv is not None:
        d["min"], d["max"] = minv, maxv
    p = panel("stat", title, x, 0, 6, 5, [target(expr)], unit)
    p["fieldConfig"]["defaults"].update(d)
    p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background", "graphMode": "area"}


green = [{"color": "#F2495C", "value": None}, {"color": "#FF9830", "value": 0.4}, {"color": "#73BF69", "value": 0.7}]
stat("KV cache hit ratio", "arctic_kv_cache_hit_ratio", 0, "percentunit", green, 0, 1)
stat("TTFT p99 (5m window)", "histogram_quantile(0.99, sum by (le) (rate(arctic_ttft_latency_ms_bucket[1m])))", 6, "ms",
     [{"color": "#73BF69", "value": None}, {"color": "#FF9830", "value": 200}, {"color": "#F2495C", "value": 1000}])
stat("In-flight tokens (all workers)", "sum(arctic_active_in_flight_tokens)", 12, "short")
stat("Speculative acceptance (fleet)", "sum(arctic_speculative_acceptance_rate)/count(arctic_speculative_acceptance_rate)", 18,
     "percentunit", green, 0, 1)

panel("timeseries", "Time to first token (ms)", 0, 5, 12, 9, [
    target("histogram_quantile(0.50, sum by (le) (rate(arctic_ttft_latency_ms_bucket[30s])))", "p50", "A"),
    target("histogram_quantile(0.95, sum by (le) (rate(arctic_ttft_latency_ms_bucket[30s])))", "p95", "B"),
    target("histogram_quantile(0.99, sum by (le) (rate(arctic_ttft_latency_ms_bucket[30s])))", "p99", "C")], "ms")
panel("timeseries", "KV hit ratio: cumulative vs live (rate)", 12, 5, 12, 9, [
    target("arctic_kv_cache_hit_ratio", "cumulative", "A"),
    target("sum(rate(arctic_cached_tokens_total[30s])) / sum(rate(arctic_prompt_tokens_total[30s]))", "live 30s", "B"),
    target("histogram_quantile(0.5, sum by (le) (rate(arctic_route_predicted_overlap_ratio_bucket[30s])))", "radix predicted overlap p50", "C")],
    "percentunit")
panel("timeseries", "Active in-flight tokens by worker", 0, 14, 12, 8,
      [target("arctic_active_in_flight_tokens", "{{worker}}")], "short", fieldConfig={
          "defaults": {"custom": {"fillOpacity": 25, "stacking": {"mode": "normal"}}}, "overrides": []})
panel("timeseries", "Batch queue depth by worker (hot-spot check)", 12, 14, 12, 8,
      [target("arctic_queue_depth", "{{worker}}")], "short")
panel("timeseries", "Speculative acceptance rate by worker", 0, 22, 12, 8,
      [target("arctic_speculative_acceptance_rate", "{{worker}}")], "percentunit")
panel("timeseries", "Throughput: requests/s and prompt tokens saved/s", 12, 22, 12, 8, [
    target('sum(rate(arctic_requests_total{outcome="ok"}[15s]))', "req/s", "A"),
    target("sum(rate(arctic_cached_tokens_total[15s]))", "cached prompt tok/s", "B"),
    target("sum(rate(arctic_prompt_tokens_total[15s]))", "total prompt tok/s", "C")], "short")
panel("timeseries", "Radix index: nodes and evictions/s", 0, 30, 12, 7, [
    target("arctic_radix_nodes", "nodes", "A"), target("rate(arctic_radix_evictions_total[15s])", "evicted blocks/s", "B")], "short")
panel("timeseries", "Errors and failovers", 12, 30, 12, 7, [
    target('sum(rate(arctic_requests_total{outcome="error"}[15s]))', "errors/s", "A"),
    target("rate(arctic_route_retries_total[15s])", "retries/s", "B"),
    target("sum(arctic_worker_healthy)", "healthy workers", "C")], "short")

dash = {"uid": "arcticflow", "title": "ArcticFlow - KV-aware routing", "tags": ["arcticflow"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "refresh": "5s", "time": {"from": "now-15m", "to": "now"},
        "style": "dark", "graphTooltip": 1, "panels": panels, "templating": {"list": []}, "annotations": {"list": []}}
out = pathlib.Path(__file__).parent / "dashboards" / "arcticflow.json"
out.write_text(json.dumps(dash, indent=1), encoding="utf-8")
print("wrote", out)
