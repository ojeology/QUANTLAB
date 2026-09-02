"""Test whether Deriv (binary options broker) API is reachable from this sandbox.
Tries HTTP to deriv.com / api.deriv.com and a WebSocket ping to ws.binaryws.com."""
import sys, json, time
def section(t): print(f"\n=== {t} ===", flush=True)

# ── HTTP ──
section("HTTP reachability")
try:
    import requests
    for url in ["https://deriv.com", "https://api.deriv.com/", "https://ws.binaryws.com/"]:
        try:
            r = requests.get(url, timeout=8)
            print(f"  {url} -> HTTP {r.status_code}, {len(r.content)} bytes", flush=True)
        except Exception as e:
            print(f"  {url} -> FAIL: {e}", flush=True)
except Exception as e:
    print(f"  requests unavailable: {e}", flush=True)

# ── WebSocket (the real Deriv API) ──
section("WebSocket ws.binaryws.com ping")
try:
    import websocket  # websocket-client
    ws = websocket.create_connection("wss://ws.binaryws.com/websockets/v3", timeout=12)
    ws.send(json.dumps({"time": 1}))
    msg = ws.recv()
    print(f"  WS connected, server said: {msg[:120]}", flush=True)
    ws.close()
    print("  WS REACHABLE", flush=True)
except Exception as e:
    print(f"  WS FAIL: {e}", flush=True)

print("\n[done]", flush=True)
