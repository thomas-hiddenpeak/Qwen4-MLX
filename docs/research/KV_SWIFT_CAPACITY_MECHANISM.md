# Swift KV 容量追加机制验证

2026-09-09。`results/kv-capacity-swift-v1/` 的release二进制SHA256为 `d1b418cd5972bf424aaa5cf438c6e27d716ce23534efc9cf19db46870f426c48`。此版本仅增加helper与探针，Attention/Model/Generation源码仍为C3；不是整模型容量模式的性能结果。

Swift通过MLX-C的`slice_update`维护私有完整backing，返回逻辑长度view。独立诊断桥只借用固定MLX-C的array引用，返回实际Metal buffer地址、offset、allocation size等标量，不增加buffer owner。常规推理不需要加载该诊断动态库。

sync与async两种模式分别通过286项检查、70,430,720个BF16元素比较。每种模式17次常规追加中12次实际复用、2次增长、2次有存活别名时的COW、1次reset后空追加；另有P256首次追加的joint/staged两种边界及后续追加，共21次操作。逻辑view为evaluation roots，旧state/view和原输入的BF16位保持不变。只看C handle或Swift对象地址不足以证明复用；本探针核对实际Metal allocation身份。

`raw_peak=0`表示区间内可能没有新malloc，不能解释为零驻留；记录的tracked high-water为`max(active_before, raw_peak, active_after)`。这些是MLX allocator观测，包含探针自身oracle和readback，不能冒称实际请求峰值、RSS或DRAM流量。仍需完整模型的状态/输出对照、真实decode计时及服务验证。

从仓库根构建独立桥，输出目录需预先创建：

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun clang++ \
  -std=c++20 -O2 -arch arm64 -mmacosx-version-min=26.2 -fno-fast-math -fvisibility=hidden \
  -I../qwen38-ssd/runtime/mlx-serve/lib/mlx/include \
  -I../qwen38-ssd/runtime/mlx-serve/lib/mlxc-src \
  native/kv_allocation_diagnostics.cpp \
  ../qwen38-ssd/runtime/mlx-serve/lib/mlx/lib/libmlx.dylib -dynamiclib \
  -Wl,-rpath,"$(pwd)/../qwen38-ssd/runtime/mlx-serve/lib/mlx/lib" -Wl,-undefined,error \
  -Wl,-install_name,@rpath/libkv_allocation_diagnostics.dylib \
  -framework Metal -framework Foundation -o /absolute/output/libkv_allocation_diagnostics.dylib
.build/release/ane-runner probe-gpu-kv-capacity \
  --diagnostics-library /absolute/output/libkv_allocation_diagnostics.dylib \
  --eval sync --output /absolute/output/new-sync.ndjson
```

再以新输出路径执行`--eval async`。须使用当前固定MLX头文件和库，并按[单GPU控制器](../EXPERIMENT_CONTROLLER.md)串行运行；不能与服务推理并行测速。成功条件包括唯一末尾summary及实际allocation/别名检查，不能删除复用断言后称通过。

同一控制器另保存改动前模型P11057/O512的完整512 IDs、length及offset11568，作为后续新generator的数值参考；旧`generate-gpu`手写循环不用于容量模式性能比较。391文件/102模型stat postflight通过，参考74988按原参数恢复。
