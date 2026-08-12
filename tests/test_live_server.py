import pytest
pytest.skip("Live server tests require STT/TTS providers configured", allow_module_level=True)

# Live server scenarios -- alternate port 18000
#
# These tests require STT/TTS providers configured.
# Run with: pytest tests/live_scenarios.py -m "live_server or not live_server"
# =============================================================================



def _start_test_server(port=18000):
    import threading
    import uvicorn
    from backend.main import app as test_app
    config = uvicorn.Config(app=test_app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    server = uvicorn.Server(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time
    time.sleep(2)
    return server, thread

def _stop_test_server(server):
    import asyncio
    server.should_exit = True
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(asyncio.sleep(0.5))
        loop.close()
    except Exception:
        pass

def _create_call_via_http(port, patient_id):
    import urllib.request, json
    body = json.dumps({"patient_id": patient_id, "dia_postop": 3, "procedimiento": "Apendicectomia", "nombre_completo": "Live Test", "eps": "EPS Live"}).encode()
    req = urllib.request.Request("http://127.0.0.1:{}/calls".format(port), data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def _send_turn_via_http(port, call_id, audio_b64):
    import urllib.request, json
    body = json.dumps({"audio_base64": audio_b64}).encode()
    req = urllib.request.Request("http://127.0.0.1:{}/calls/{}".format(port, call_id), data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def test_live_server_1_health():
    server, thread = _start_test_server(18000)
    try:
        import urllib.request, json
        req = urllib.request.Request("http://127.0.0.1:18000/health")
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["status"] == "ok"
    finally:
        _stop_test_server(server)
        thread.join(timeout=5)

def test_live_server_2_create_call():
    server, thread = _start_test_server(18000)
    try:
        data = _create_call_via_http(18000, "pac-live-1")
        assert "call_id" in data
        assert data["state"] == "GREETING"
    finally:
        _stop_test_server(server)
        thread.join(timeout=5)

def test_live_server_3_greeting_to_consent():
    import base64
    mock_audio = base64.b64encode(b"\x00" * 100).decode()
    server, thread = _start_test_server(18000)
    try:
        call = _create_call_via_http(18000, "pac-live-2")
        turn = _send_turn_via_http(18000, call["call_id"], mock_audio)
        assert turn["state"] == "CONSENT"
    finally:
        _stop_test_server(server)
        thread.join(timeout=5)

def test_live_server_4_consent_to_questions():
    import base64
    mock_audio = base64.b64encode(b"\x00" * 100).decode()
    server, thread = _start_test_server(18000)
    try:
        call = _create_call_via_http(18000, "pac-live-3")
        _send_turn_via_http(18000, call["call_id"], mock_audio)
        turn = _send_turn_via_http(18000, call["call_id"], mock_audio)
        assert turn["state"] == "QUESTIONS"
    finally:
        _stop_test_server(server)
        thread.join(timeout=5)

def test_live_server_5_404():
    import base64, urllib.request, urllib.error, json
    mock_audio = base64.b64encode(b"\x00" * 100).decode()
    server, thread = _start_test_server(18000)
    try:
        body = json.dumps({"audio_base64": mock_audio}).encode()
        req = urllib.request.Request("http://127.0.0.1:18000/calls/nonexistent/turn", data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req)
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        _stop_test_server(server)
        thread.join(timeout=5)

def test_live_server_6_metrics():
    import urllib.request, json
    server, thread = _start_test_server(18000)
    try:
        req = urllib.request.Request("http://127.0.0.1:18000/metrics/summary")
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
            assert "total_calls" in data
    finally:
        _stop_test_server(server)
        thread.join(timeout=5)

def run_all_live_server():
    import time, traceback
    print("\nLIVE SERVER SCENARIOS -- PORT 18000\n")
    start = time.time()
    results = {}
    for name, fn in [("health", test_live_server_1_health), ("create", test_live_server_2_create_call), ("greeting", test_live_server_3_greeting_to_consent), ("consent", test_live_server_4_consent_to_questions), ("404", test_live_server_5_404), ("metrics", test_live_server_6_metrics)]:
        try:
            fn()
            results[name] = True
            print("PASS " + name)
        except Exception as e:
            traceback.print_exc()
            results[name] = False
            print("FAIL " + name)
    ok = all(results.values())
    print("\nLIVE SERVER " + ("ALL PASSED" if ok else "SOME FAILED") + " ({:.2f}s)".format(time.time() - start))
    return ok
