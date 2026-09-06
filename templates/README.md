# Instruments GPU telemetry templates

The supported entry point is the installed Apple `Metal System Trace` template,
used by `../scripts/gpu_trace.py`. Templates are specific to the installed Xcode
and target GPU. A template's existence does not establish counter availability.

Observed on M5 Max, macOS 26.6.2, Xcode 26.6 (17F113):

| Profile | Real short recording result |
| --- | --- |
| Original profile 0 | 62,227 intervals, only `RT Unit Active` (%) |
| Experimental profile 1 | GPU Service: selected counter profile is not supported on target device |
| Experimental profile 4 | Same explicit unsupported-profile warning |

The two copied `gpu-counters-profile*.tracetemplate` files are failed capability
probes, **not recommended bandwidth configurations**. They only change the
archived template's three counter profile values; the installed Xcode template
was not edited. Numeric profile IDs have not been mapped to the current device's
UI choices. Instruments UI inspection was unavailable because Computer Use
reported missing permissions, then that the Mac was locked. No permission or
screen-lock setting was changed.

The installed GPU plugin contains names for GPU Read/Write Bandwidth. This is
not evidence that this Xcode/GPU combination can currently collect them. The
exporter leaves missing bandwidth and physical DRAM measurements unknown.

## Reproducible use

Run from `experiments/ane-runner`; use a fresh output directory each time:

```sh
python3 scripts/gpu_trace.py record --template 'Metal System Trace' \
  --output results/my-trace --time-limit 30s --attach TARGET_PID

python3 scripts/gpu_trace.py export --trace results/my-trace/recording.trace \
  --output results/my-export --target-pid TARGET_PID --phases phases.jsonl
```

`record --launch /absolute/executable arguments...` is also supported. The
exporter can recover the exact launched/attached PID from the trace's target
metadata; `--target-pid` explicitly selects one process for an existing trace.
It does not start or stop any other model service. `DEVELOPER_DIR` is set only
for the child xctrace command and never changes the system's developer selection.

Outputs:

- `counters.json`: raw intervals, names and units actually exported by Apple,
  capability status, and exact time-info mapping when present. Device counter
  values are not attributed exclusively to the selected process.
- `gpu_intervals.json`: exact-PID filtered GPU execution intervals and separate
  CPU command-buffer submissions. GPU state rows for the whole system are not
  used as runner activity. Union overlapping Active execution intervals before
  calculating GPU busy time; do not sum overlapping channels or nested events.
- Original XML tables, TOC, and command/result metadata preserve the evidence.

`time-info` supplies the run epoch in mach absolute ticks and the rational
timebase. The exporter converts with integer arithmetic, retaining 1 ns of
rounding uncertainty. No launch-wall-clock approximation is used. Phase JSONL
is retained for the offline analyzer, which must not proportionally allocate
samples crossing phase boundaries.

Apple describes GPU bandwidth as traffic at the GPU's external-memory
interface; system-level cache can contribute. Consequently these counters,
even when available, are not isolated physical DRAM byte counters:
[Metal profiling](https://developer.apple.com/videos/play/wwdc2020/10603/),
[GPU memory bandwidth](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth).

Offline verification:

```sh
python3 -m unittest discover -s scripts -p gpu_trace_test.py -v
```

## Recovery of an unfinished recording

A recording containing only `Trace1.run/Attachments/trace-data.atrc` is not yet
a finalized Instruments document. `xctrace export` can report `Document Missing
Template Error`. The standard offline import command can recover its recorded
events into a new trace without rerunning the target:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace import \
  --input original.trace/Trace1.run/Attachments/trace-data.atrc \
  --template 'Metal System Trace' --output recovered.trace
python3 scripts/gpu_trace.py export --trace recovered.trace --output recovered-export \
  --target-pid TARGET_PID --phases phases.jsonl --incomplete-source-trace original.trace
```

This was verified with the actual runner recording: a 2.42 GB attachment
recovered 44,999 target GPU execution intervals. Importing took over two minutes
of CPU processing. It is separate from inference and does not supply a new
throughput measurement. A successful recovery does not prove that the original
recording was complete or free of dropped events. The explicit
`--incomplete-source-trace` option therefore keeps `coverage.complete=false`
and retains the original archive state. Report observed activity as a lower
bound; leave total idle time and utilization unknown.
