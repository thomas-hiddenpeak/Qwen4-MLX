#!/usr/bin/env python3
"""Run CPU-only lifecycle checks against the current CoreAI HTTP transport.

Requires macOS and an installed Swift 6 toolchain (xcrun swiftc). Builds only the
HTTP DTO sources, transport, service interface and fake backend in a temporary
directory. No model assets, CoreAI runtime, MLX build or existing server is used.

Example:
    python3 scripts/check_coreai_http_transport.py --port 11239 --output /tmp/http.json
"""

import argparse
import hashlib
import json
import pathlib
import platform
import signal
import socket
import struct
import subprocess
import tempfile
import time
import uuid

REPO = pathlib.Path(__file__).resolve().parents[1]
DTO_SOURCES = ["Sources/ANERunnerCore/" + name + ".swift" for name in
               ("QwenHTTPProtocol", "QwenHTTPTools", "QwenHTTPFrames")]
SERVER_SOURCES = ["Sources/CoreAIRunnerCLI/CoreAIServiceTypes.swift",
                  "Sources/CoreAIRunnerCLI/CoreAIHTTPServer.swift",
                  "scripts/fixtures/CoreAIHTTPFake.swift"]
CONFIGURATION = {"maxConnections": 3, "maxBodyBytes": 1024,
                 "maxOutputBytes": 1_048_576, "receiveTimeoutSeconds": 0.5,
                 "requestTimeoutSeconds": 1.2, "sendTimeoutSeconds": 0.15}


def compile_probe(build):
    common = ["xcrun", "swiftc", "-swift-version", "6", "-target",
              platform.machine() + "-apple-macosx26.2"]
    commands = [
        common + ["-emit-library", "-emit-module", "-module-name", "ANERunnerCore"]
        + DTO_SOURCES + ["-emit-module-path", str(build / "ANERunnerCore.swiftmodule"),
                        "-Xlinker", "-install_name", "-Xlinker", "@rpath/libANERunnerCore.dylib",
                        "-o", str(build / "libANERunnerCore.dylib")],
        common + ["-I", str(build), "-L", str(build), "-lANERunnerCore",
                  "-Xlinker", "-rpath", "-Xlinker", "@executable_path"]
        + SERVER_SOURCES + ["-o", str(build / "transport-probe")],
    ]
    for command in commands:
        subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True, timeout=60)


def run_probe(build, port):
    probe_id = str(uuid.uuid4())
    report = {"scope": "CPU-only fake backend; current transport and DTO sources; no CoreAI/model/MLX",
              "port": port, "configuration": CONFIGURATION, "cases": []}
    proc = subprocess.Popen([str(build / "transport-probe"), str(build / "shutdown.json"),
                             str(port), probe_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def connect(buffer=None):
        s = socket.socket()
        s.settimeout(3)
        if buffer: s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer)
        try:
            s.connect(("127.0.0.1", port))
            return s
        except BaseException:
            s.close()
            raise

    def wire(mode="normal", stream=False):
        body = json.dumps({"model": "cpu-transport-only", "messages": [{"role": "user", "content": mode}], "stream": stream, "max_tokens": 8}).encode()
        return b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body

    def read_all(s):
        out = bytearray()
        while True:
            try: b = s.recv(16384)
            except ConnectionResetError: break
            if not b: break
            out.extend(b)
            assert len(out) <= CONFIGURATION["maxOutputBytes"] + 16_384, "Unbounded HTTP output"
        return bytes(out)

    def request(raw, half=False):
        with connect() as s:
            s.sendall(raw)
            if half: s.shutdown(socket.SHUT_WR)
            return read_all(s)

    def health():
        data = request(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assert data.startswith(b"HTTP/1.1 200"), data[:100]
        value = json.loads(data.split(b"\r\n\r\n", 1)[1])
        assert value.get("probe_id") == probe_id, "Port is serving an unrelated process"
        return value

    def wait_for(predicate, timeout=3):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if proc.poll() is not None:
                raise AssertionError(f"Fake server exited early: {proc.returncode}")
            try:
                v = health()
                if predicate(v): return v
            except (OSError, AssertionError, json.JSONDecodeError, IndexError): pass
            time.sleep(.03)
        raise AssertionError("health condition timed out")

    def case(name, fn):
        begin = time.monotonic()
        try:
            detail = fn()
        except Exception as error:
            report["cases"].append({"name": name, "passed": False, "error": repr(error),
                                    "seconds": round(time.monotonic()-begin, 4)})
            raise
        report["cases"].append({"name": name, "passed": True, "seconds": round(time.monotonic()-begin, 4), **(detail or {})})

    def status_case(raw, status):
        out = request(raw)
        assert out.startswith(f"HTTP/1.1 {status} ".encode()), out[:200]
        return {"status": status}

    try:
        wait_for(lambda x: True)
        case("oversized_content_length", lambda: status_case(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 1025\r\n\r\n", 413))
        case("duplicate_content_length", lambda: status_case(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx", 400))
        case("transfer_encoding_rejected", lambda: status_case(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n", 400))
        case("pipelining_rejected", lambda: status_case(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\nGET /health HTTP/1.1\r\nHost: x\r\n\r\n", 400))
        case("header_budget", lambda: status_case(b"GET /health HTTP/1.1\r\nHost: x\r\nX: "+b"a"*16384+b"\r\n\r\n", 431))
        case("unknown_json_field", lambda: status_case(wire().replace(b'"max_tokens": 8',b'"bad_fields": 8'), 400))
        def halfclose():
            out = request(wire(), half=True)
            assert out.startswith(b"HTTP/1.1 200 ")
            data=json.loads(out.split(b"\r\n\r\n",1)[1])
            assert data["choices"][0]["message"]["content"]=="你好"
            return {"completion": "你好"}
        case("legal_tcp_write_half_close", halfclose)
        def sse():
            out=request(wire(stream=True), half=True)
            assert out.startswith(b"HTTP/1.1 200 ") and b": keep-alive\n\n" in out and out.endswith(b"data: [DONE]\n\n")
            assert b"chat.completion.chunk" in out
            return {"bytes": len(out), "heartbeat": True, "done": True}
        case("sse_half_close_heartbeat_completion", sse)
        case("receive_deadline", lambda: status_case(b"GET /health HTTP/1.1\r\nHost:", 408))
        def deadline(stream):
            before=health()["cancelled"]
            out=request(wire("slow",stream))
            assert out.startswith(b"HTTP/1.1 200 " if stream else b"HTTP/1.1 408 "),out[:150]
            assert b"Request deadline exceeded" in out
            state=wait_for(lambda x:x["active"]==0 and x["cancelled"]>before)
            return {"stream":stream,"cancelled":state["cancelled"],"body_bytes":len(out)}
        case("nonstream_request_deadline",lambda:deadline(False))
        case("sse_request_deadline",lambda:deadline(True))
        def disconnect(reset):
            before=health()["cancelled"]
            s=connect();s.sendall(wire("slow",True))
            assert b"HTTP/1.1 200" in s.recv(16384)
            wait_for(lambda x:x["active"]==1)
            begin=time.monotonic()
            if reset:s.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack("ii",1,0))
            s.close()
            state=wait_for(lambda x:x["active"]==0 and x["cancelled"]>before)
            return {"cancel_elapsed_seconds":round(time.monotonic()-begin,4),"cancelled":state["cancelled"]}
        case("sse_full_close_during_prefill",lambda:disconnect(False))
        case("sse_tcp_reset_during_prefill",lambda:disconnect(True))
        def health_during_work():
            s=connect();s.sendall(wire("slow",True));s.recv(16384)
            begin=time.monotonic();state=health();elapsed=time.monotonic()-begin
            assert state["active"]==1 and elapsed<.25,(elapsed,state)
            s.close();wait_for(lambda x:x["active"]==0)
            return {"health_latency_seconds":round(elapsed,5)}
        case("health_not_blocked_by_active_work",health_during_work)
        def overflow():
            out=request(wire("overflow",True))
            assert b"Output exceeds configured byte limit" in out and out.endswith(b"data: [DONE]\n\n")
            assert len(out.split(b"\r\n\r\n",1)[1])<=1_048_576
            wait_for(lambda x:x["active"]==0)
            return {"response_bytes":len(out)}
        case("output_budget_error_and_done",overflow)
        def slow_reader():
            before=health()["cancelled"]
            s=connect(1024);s.sendall(wire("flood",True))
            begin=time.monotonic()
            state=wait_for(lambda x:x["cancelled"]>before and x["active"]==0)
            elapsed=time.monotonic()-begin
            s.close()
            assert elapsed<1.15,elapsed
            return {"bounded_exit_seconds":round(elapsed,4),"cancelled":state["cancelled"],"socket_receive_buffer":1024}
        case("stalled_sse_reader_releases_producer",slow_reader)
        def connections():
            held=[]
            try:
                for _ in range(3):
                    s=connect();s.sendall(b"GET /health HTTP/1.1\r\n");held.append(s)
                time.sleep(.05)
                with connect() as extra:
                    try:
                        extra.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
                        data=extra.recv(1024)
                    except ConnectionResetError:data=b""
                    assert not data,data
            finally:
                for s in held:s.close()
            wait_for(lambda x:True)
            return {"limit":3,"excess_connection_closed":True}
        case("max_connections",connections)
        def shutdown():
            s=connect(1024);s.sendall(wire("flood",True));wait_for(lambda x:x["active"]==1)
            begin=time.monotonic();proc.send_signal(signal.SIGTERM);proc.wait(timeout=3);s.close()
            assert proc.returncode==0,proc.returncode
            data=json.loads((build/"shutdown.json").read_text())
            assert data["drained"] and data["state"]["active"]==0,data
            return {"exit_seconds":round(time.monotonic()-begin,4),"exit_code":proc.returncode,"worker_drained":True}
        case("sigterm_releases_inflight_send_and_stops",shutdown)
        report["passed"]=True
    except Exception as e:
        report["passed"]=False
        report["error"]=repr(e)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        out,err=proc.communicate()
        report["stderr"]=err.decode(errors="replace")
        report["exit_code"]=proc.returncode
        if (build/"shutdown.json").exists():
            report["shutdown"] = json.loads((build/"shutdown.json").read_text())
        report["case_count"] = len(report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11239)
    parser.add_argument("--output", type=pathlib.Path,
                        default=REPO / "results/coreai-service/transport-cpu/report.json")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("this Network.framework transport check requires macOS")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1...65535")
    if not __debug__:
        parser.error("run without Python -O so test assertions remain enabled")
    report = {"passed": False, "port": args.port, "cases": []}
    try:
        # Fail before launching if another process already owns the requested
        # port. The per-run health identity also protects against bind races.
        with socket.socket() as check:
            check.bind(("127.0.0.1", args.port))
        with tempfile.TemporaryDirectory(prefix="coreai-http-cpu-") as directory:
            build = pathlib.Path(directory)
            compile_probe(build)
            report = run_probe(build, args.port)
    except Exception as error:
        report["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            report["compiler_stderr"] = error.stderr
    report["source_sha256"] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                               for name in DTO_SOURCES + SERVER_SOURCES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "report": str(args.output.resolve()), "error": report.get("error")},
                     ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
