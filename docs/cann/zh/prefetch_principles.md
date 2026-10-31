# NPU Prefetch 原理

本文档说明昇腾 NPU 上权重预取的背景、硬件与接口语义、模型代码中的实现方式，以及 LongCat-Flash 与 DeepSeek-V4 中已经落地的典型 Prefetch 优化案例。

## 1. 背景

大模型推理的 Decode 阶段常见一个特点：输入 token 数少、batch 较小，矩阵乘类算子的计算量不一定能把全部计算资源打满，但每次仍需要读取体量很大的模型权重。以 MatMul、QuantBatchMatmul(QBMM)、GroupedMatmul(GMM) 为例，输入激活 `x` 通常刚由前序算子产生，规模也相对较小；权重 `W` 则可能达到几十 MB，主要常驻在 HBM 中。此时算子耗时容易由“读权重”主导，表现为 memory-bound。

Prefetch 的目标是把这部分权重读取提前。当前序存在一个 HBM 带宽没有打满的窗口时，可以通过 `torch_npu.npu_prefetch` 提前把后续目标算子的部分权重从 HBM 搬入 L2 Cache。目标算子真正执行时，已经预取的权重可以从更近的缓存层读取，从而减少目标算子窗口内的 HBM 访存读取耗时。

```mermaid
block-beta
  columns 12
  s0["主执行流"]:1 a["非访存瓶颈算子<br/>如 RMSNorm / Dispatch"]:3 b["MatMul / GMM<br/>目标算子"]:5 space:3
  s1["Prefetch"]:1 space:1 p["提前预取目标权重<br/>HBM → L2 Cache"]:2 space:7

  a --> p
  p --> b

  classDef hdr fill:#eee,stroke:#999,color:#000
  classDef main fill:#5B9BD5,color:#fff,stroke:#1F4E79
  classDef pre fill:#70AD47,color:#fff,stroke:#385723

  class s0,s1 hdr
  class a,b main
  class p pre
```

Prefetch 不是通用加速开关。只有满足下面条件时才值得尝试：

1. profiling 已确认目标算子是 memory-bound，瓶颈主要来自权重搬运。
2. 目标权重足够大，提前搬运能覆盖可观的 HBM 读取开销。
3. 目标算子前存在安全的依赖窗口，且窗口本身不会被预取明显拖慢。
4. 单点 profiling 能解释收益来源，而不是只观察到端到端指标的随机波动。

典型目标包括 `q_proj`、`o_proj`、`gate_up_proj`、`down_proj` 等这些 MatMul 或 GMM 算子。LayerNorm、RMSNorm、ROPE、SwiGLU、router topk、dispatch、combine 等通常不是预取目标，而是可能提供预取窗口的前序算子。

## 2. 原理

### 2.1 算子计算前的权重访问链路

矩阵乘、QBMM、GMM 等算子执行时，计算单元不能直接使用常驻 HBM 的模型权重。算子启动后，硬件需要按照算子的访问模式，把权重从 HBM 搬到 L2 Cache 等片上缓存，再由 Cube / Vector 计算单元读取并完成计算。

可以把目标算子的执行简化为两个相互交织的阶段：

1. **搬运阶段**：从 HBM 读取目标权重，将其送入 L2 Cache 等片上缓存。
2. **计算阶段**：计算单元从片上缓存读取权重，与激活输入完成 MatMul / GMM 等计算。

```mermaid
flowchart LR
    H["HBM<br/>模型权重"] --> L["L2 Cache<br/>片上缓存"]
    L --> C["Cube / Vector<br/>计算单元"]
    X["activation 输入"] --> C
    C --> O["output"]
```

在 Decode 小 batch 场景中，输入 token 数少，单次 MatMul / GMM 的计算量有限，权重读取的占比会变高。当 HBM 到 L2 Cache 的搬运时间接近或超过实际计算时间时，目标算子就会表现为 memory-bound。Prefetch 优化的目标不是改变算子的数学计算，而是提前完成一部分权重访问链路中的 HBM → L2 Cache 搬运。


### 2.2 基于多流的异步搬运

Prefetch 依赖多流机制才能产生收益。NPU 上 Host 通过 Stream 向 Device 下发任务，同一 Stream 内任务按顺序执行；不同 Stream 之间只要依赖关系满足，就可以并行调度。多流优化正是利用这一点，把计算、通信或数据搬运放到不同 Stream 上重叠执行，从而隐藏一部分非关键路径耗时。多流的基础概念见 [multi_stream_principles.md](multi_stream_principles.md)。

`torch_npu.npu_prefetch` 的核心就是把目标权重的数据搬运放到独立预取流中执行。模型代码在主执行流上得到某个依赖 Tensor 后，调用 `torch_npu.npu_prefetch(weight, depend, size, offset)` 接口发起预取；运行时根据 `depend` 建立同步关系，保证预取不会早于依赖 Tensor 的生产时刻，同时让主执行流继续推进后续算子。

```mermaid
block-beta
  columns 13
  s0["Stream0 主执行流"]:1 a["前序算子<br/>产生 depend"]:3 c["中间窗口"]:3 b["目标 MatMul/GMM"]:4 space:2
  s1["Stream1 预取流"]:1 space:3 p["npu_prefetch<br/>搬运目标权重"]:3 space:4

  a --> p
  p --> b

  classDef hdr fill:#eee,stroke:#999,color:#000
  classDef main fill:#5B9BD5,color:#fff,stroke:#1F4E79
  classDef pre fill:#70AD47,color:#fff,stroke:#385723

  class s0,s1 hdr
  class a,c,b main
  class p pre
```

因此，Prefetch 与普通多流优化的关系可以理解为：普通多流通常并行的是两个计算/通信分支，Prefetch 并行的是后续算子的权重搬运。模型侧需选择正确的 `depend`，否则运行时无法建立合理的异步搬运窗口。

| 对比项 | 多流优化 | Prefetch |
| --- | --- | --- |
| 并行对象 | 计算分支、通信分支、不同资源类型算子 | 目标算子的权重搬运 |
| 依赖表达 | Stream/Event、图内数据依赖或 `npu_wait_tensor` | `depend` Tensor |
| 主要收益 | 提高 Cube / Vector / 通信资源利用率 | 隐藏 HBM 权重读取开销 |
| 调优手段 | 调整 Stream 编排、Event、控核比例 | 调整 `depend` 位置和 `size` |

### 2.3 Prefetch 如何缩短关键路径

如果不做 Prefetch，目标算子的执行窗口内需要同时承担 HBM 读取、片上缓存填充和实际计算。也就是说，权重搬运耗时直接落在目标算子的关键路径上。

引入 Prefetch 后，运行时会在依赖 Tensor 之后发起异步搬运，将一部分目标权重提前放入 L2 Cache。当前序窗口与预取搬运可以重叠时，目标算子启动后需要等待的 HBM 读取减少，关键路径就可能缩短。

```mermaid
block-beta
  columns 17
  s0["无 Prefetch"]:2 space:3 m0["HBM→L2 搬权重"]:5 c0["计算"]:4 space:3
  s1["有 Prefetch"]:2 p1["提前 HBM→L2"]:2 space:1 m1["剩余搬运"]:2 c1["计算"]:4 space:1

  classDef hdr fill:#eee,stroke:#999,color:#000
  classDef move fill:#70AD47,color:#fff,stroke:#385723
  classDef compute fill:#5B9BD5,color:#fff,stroke:#1F4E79

  class s0,s1 hdr
  class m0,p1,m1 move
  class c0,c1 compute
```

可以按下面的顺序理解一次有效预取：

1. 前序算子产生 `depend`，主执行流继续运行。
2. 预取流在 `depend` 之后开始搬运 `weight` 的前 `size` 字节。
3. 搬运结果进入 L2 Cache 等片上缓存。
4. 目标算子启动后，优先消费已经预取到缓存中的权重数据。
5. 未覆盖的权重仍按原路径从 HBM 读取。

## 3. 实现方式

Prefetch 的实现侧主要负责四件事：选择目标权重 `weight`、选择依赖 Tensor `depend`、计算预取大小 `size`、在合适位置调用 `torch_npu.npu_prefetch`。

### 3.1 接口语义

官方接口说明见 [torch_npu.npu_prefetch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_prefetch.md)。文档与仓库实践中使用的调用形态如下：

```python
torch_npu.npu_prefetch(weight, depend, size, offset)
```

| 参数 | 含义 | 设计要求 |
| --- | --- | --- |
| `weight` | 需要预取的权重 Tensor | 通常传入目标算子的 `.weight.data` 或专家权重 `.data` |
| `depend` | 约束预取开始时机的依赖 Tensor | 目标算子之前稳定产生、能表达预取开始时机的 tensor |
| `size` | 本次最多预取的字节数 | 从保守值开始，根据 profiling 调整 |
| `offset` | 权重内预取起始偏移 | 当前仓库实践通常使用 `0` |

### 3.2 执行模式与流管理

Prefetch 的流管理方式与执行模式有关。eager / `ge_graph` 模式下，预取流由接口、图编译或运行时内部处理；`npugraph_ex` 模式下，模型侧需要显式创建并持有预取流。

#### 3.2.1 eager / ge_graph 用法

eager 或 `ge_graph` 模式下，模型代码直接调用 `torch_npu.npu_prefetch` 即可，不需要手动创建 `torch.npu.Stream()`，也不需要额外切流。模型侧只需要保证 `weight`、`depend`、`size`、`offset` 传入正确：

```python
if enable_prefetch:
    torch_npu.npu_prefetch(
        target_weight,
        depend_tensor,
        prefetch_size,
        0,
    )
```

在这种模式下，预取的开始时机由 `depend` 表达。接口或图运行时会基于这个依赖关系安排预取任务，使其在合适的预取流上执行。接入时不建议额外包一层手写 Stream / Event 逻辑，否则容易引入不必要的同步等待。

#### 3.2.2 npugraph_ex 用法

`npugraph_ex` 模式下，图捕获边界和流复用需要更明确，需要手动创建 prefetch 操作的流，例如，在 LongCat-Flash 中会在模型初始化时创建专门的 `npugraph_prefetch_stream`：

```python
self.npugraph_prefetch_stream = None
if enable_npugraph_ex and enable_multi_streams and enable_prefetch:
    self.npugraph_prefetch_stream = torch.npu.Stream()
```

具体调用时，先判断是否存在该预取流；如果存在，就通过 `npu_stream_switch` 切到预取流，再发起 prefetch。按原接口表示如下：

```python
route_prefetch = self.npugraph_prefetch_stream is not None
with npu_stream_switch(route_prefetch, self.npugraph_prefetch_stream):
    torch_npu.npu_prefetch(
        self.gate_up_proj.weight.data,
        o_proj,
        self.up_gate_prefetch_size,
        0,
    )
```

如果预取依赖当前流上的某个事件，需要在预取流上等待该事件，确保预取不会早于依赖数据产生：

```python
with npu_stream_switch(route_prefetch, self.npugraph_prefetch_stream):
    if route_prefetch and x_event is not None:
        self.npugraph_prefetch_stream.wait_event(x_event)
    if enable_prefetch:
        torch_npu.npu_prefetch(
            self.down_proj.weight.data,
            x,
            self.down_prefetch_size,
            0,
        )
```

在 `npugraph_ex` 的整网执行末尾，通常还需要让主流等待预取流收尾，避免后续阶段过早复用相关资源：

```python
if self.npugraph_prefetch_stream is not None and not is_prefill:
    with npu_stream_switch(True, self.npugraph_prefetch_stream):
        prefetch_done_event = torch.npu.current_stream().record_event()
    torch.npu.current_stream().wait_event(prefetch_done_event)
```

### 3.3 预取大小

`size` 决定本次预取量。过小收益有限，过大会抢占 HBM 带宽，导致依赖窗口或其他 Stream 退化。推荐先按权重大小计算理论值，再从保守比例开始验证。

```text
理论预取大小 = 权重元素个数 × dtype 字节数
首次 size = 理论预取大小 × 保守系数
```

| dtype | 字节数 |
| --- | --- |
| `int8` | 1 |
| `fp16 / bf16` | 2 |
| `fp32` | 4 |

常见算子的估算方式：

| 目标算子 | 权重形状 | dtype | 理论大小 | 首次建议 |
| --- | --- | --- | --- | --- |
| `q_proj / o_proj` | `hidden_size × hidden_size` | bf16 | `H × H × 2` | `H × H` |
| `gate_up_proj` | `hidden_size × intermediate_size × 2` | bf16 | `H × I × 4` | `H × I × 2` |
| `down_proj` | `intermediate_size × hidden_size` | bf16 | `I × H × 2` | `I × H` |

例如，LongCat-Flash 的专家权重预取大小按量化模式与并行切分计算，按本地专家权重计算：

```python
gmm1_prefetch_size = (
    hidden_size * intermediate_size * 2 * dtype_bit
    // moe_tp_size * experts_per_rank // 2
)

gmm2_prefetch_size = (
    hidden_size * intermediate_size * dtype_bit
    // moe_tp_size * experts_per_rank
)
```

## 4. 具体网络样例

### 4.1 LongCat-Flash：Attention / Dense 权重预取

LongCat-Flash 中已经在 `models/longcat_flash/models/modeling_longcat_flash.py` 接入多处 prefetch，以 Dense MLP 与 Attention 阶段的 prefetch 做法为例，核心思路是在当前层计算结束或当前子模块执行期间，提前预取后续访存密集算子的权重。

#### 4.1.1 Dense MLP 预取

`LongCatFlashMLP` 中，`gate_up_proj` 和 `down_proj` 都是权重量较大的线性层。`gate_up_proj` 可在上一段 `o_proj` 完成后预取，`down_proj` 可在 `gate_up_proj` 执行期间作为依赖窗口预取。

```mermaid
block-beta
  columns 19
  s0["主执行流"]:1 o["o_proj"]:2 h1["通信 Reduce_scatter"]:2 l["layer_norm"]:1 h2["通信 All_gather"]:2 dq["dynamic_quant"]:2 g["gate_up_proj"]:4 sw["SwiGLU"]:1 d["down_proj"]:4
  s1["Prefetch"]:1 space:2 pg["预取 gate_up_proj"]:2 space:3 pd["预取 down_proj"]:2 space:10

  pg --> g
  pd --> d

  classDef hdr fill:#eee,stroke:#999,color:#000
  classDef main fill:#5B9BD5,color:#fff,stroke:#1F4E79
  classDef pre fill:#70AD47,color:#fff,stroke:#385723

  class s0,s1 hdr
  class o,g,sw,d,h1,h2,dq,l main
  class pg,pd pre
```

对应的原接口表示如下：

```python
if enable_prefetch:
    torch_npu.npu_prefetch(
        self.gate_up_proj.weight.data,
        o_proj_output,
        self.up_gate_prefetch_size,
        0,
    )

    torch_npu.npu_prefetch(
        self.down_proj.weight.data,
        gate_up_input,
        self.down_prefetch_size,
        0,
    )
```

其中 `up_gate_prefetch_size` 和 `down_prefetch_size` 基于 hidden size、intermediate size、dtype 和 dense TP 切分计算。

#### 4.1.2 Decode 阶段 MLAProlog 权重预取

Decode 阶段如果开启 `enable_mla_prolog`，attention 中的 Q/KV 准备逻辑会走 `mla_prolog`。此时 `q_a_proj`、`q_b_proj`、`kv_a_proj_with_mqa` 不再分别作为普通 Python 线性层逐个调用，而是作为 `npu_mla_prolog_v3` 的权重输入：

```python
torch.ops.npu.npu_mla_prolog_v3(
    token_x=hidden_states,
    weight_dq=q_a_weight,
    weight_uq_qr=q_b_weight,
    weight_dkv_kr=kv_a_weight,
    ...
)
```

因此，LongCat-Flash 中 decode 阶段的 attention 预取，本质上是为后续 MLAProlog 算子提前搬运 `q_a_proj.weight`、`q_b_proj.weight` 和 `kv_a_proj_with_mqa.weight`。这些权重体量较大，且 MLAProlog 位于下一段 attention 的入口处，适合利用前一段 Dense MLP 或跨层尾部窗口提前预取。

| 目标权重 | 说明 |
| --- | --- |
| `q_a_proj.weight` | 对应 MLAProlog 的 `weight_dq` |
| `q_b_proj.weight` | 对应 MLAProlog 的 `weight_uq_qr` |
| `kv_a_proj_with_mqa.weight` | 对应 MLAProlog 的 `weight_dkv_kr` |

LongCat-Flash 中有两类典型位置：

1. **层内预取**：第一段 Dense MLP 后，用 MLP 中间依赖 Tensor 作为 `depend`，预取同一层第二段 attention 的 MLAProlog 权重。
2. **跨层预取**：当前层第二段 Dense MLP 后，用当前层 `down_proj` 输出作为 `depend`，预取下一层第一段 attention 的 MLAProlog 权重。

按原接口抽象表示如下：

```python
if enable_prefetch:
    torch_npu.npu_prefetch(
        next_attention_q_a_weight,
        mlp_depend_tensor,
        q_a_prefetch_size,
        0,
    )
    torch_npu.npu_prefetch(
        next_attention_q_b_weight,
        mlp_depend_tensor,
        q_b_prefetch_size,
        0,
    )
    torch_npu.npu_prefetch(
        next_attention_kv_a_weight,
        mlp_depend_tensor,
        kv_a_prefetch_size,
        0,
    )
```

在 `npugraph_ex` 模式下，上述 prefetch 需要放到显式预取流中，并在依赖事件之后执行：

```python
with npu_stream_switch(has_prefetch_stream, prefetch_stream):
    if has_prefetch_stream and mlp_done_event is not None:
        prefetch_stream.wait_event(mlp_done_event)
    if enable_prefetch:
        torch_npu.npu_prefetch(next_attention_q_a_weight, mlp_depend_tensor, q_a_prefetch_size, 0)
        torch_npu.npu_prefetch(next_attention_q_b_weight, mlp_depend_tensor, q_b_prefetch_size, 0)
        torch_npu.npu_prefetch(next_attention_kv_a_weight, mlp_depend_tensor, kv_a_prefetch_size, 0)
```

```mermaid
block-beta
  columns 14
  s0["主执行流"]:1 d["Dense MLP"]:3 h1["通信 / Add"]:3 l["LayerNorm"]:2 m["MLAProlog"]:4 space:1
  s1["Prefetch"]:1 space:3 p1["预取 q_a"]:2 p2["预取 q_b"]:2 p3["预取 kv_a"]:2 space:4

  p1 --> m
  p2 --> m
  p3 --> m


  classDef hdr fill:#eee,stroke:#999,color:#000
  classDef main fill:#5B9BD5,color:#fff,stroke:#1F4E79
  classDef pre fill:#70AD47,color:#fff,stroke:#385723

  class s0,s1 hdr
  class d,h1,l,m main
  class p1,p2,p3 pre
```

这类预取的目标不是 `fused_infer_attention_score`，而是 attention 入口处的 MLAProlog。收益取决于 Dense MLP 输出到 MLAProlog 启动之间是否存在稳定窗口；如果多流、AFD、`npugraph_ex` 或图模式策略改变了层内时序，需要重新 profiling。

### 4.2 DeepSeek-V4：Attention / lm_head 权重预取

DeepSeek-V4 在 `models/deepseek_v4/models/modeling_deepseek.py` 中为 A3 与 ASCEND_950 接入权重预取。各平台的预取点与预取大小通过模型文件顶部的 `PREFETCH_PLATFORM_CONFIGS` 集中配置，每个平台一个字典，以 `*_FIXED_SIZE`（固定字节数）或 `*_RATIO`（权重比例）给出各点位的预取大小：

```python
PREFETCH_PLATFORM_CONFIGS = {
    PlatformVersion.A3: {
        "WQ_A_FIXED_SIZE": 2 * 1024 * 1024,   # wq_a 预取固定 2MB
        ...
    },
    PlatformVersion.ASCEND_950: {
        "WQ_A_RATIO": 0.8,                    # wq_a 按 0.8 比例预取
        ...
    },
}
```

各点位的具体取值见 4.2.1 与 4.2.2。

| 配置语义 | 说明 |
| --- | --- |
| 预取大小 | `FIXED_SIZE > 0` 时预取固定字节数，否则按 `权重字节数 × RATIO` 计算；运行时读取实际权重，自动适应 TP 切分与量化 dtype |
| 点位启停 | 配置了正的预取大小即启用，未配置的点位不产生 CMO，不设独立的点位开关 |
| lm_head 分片边界 | 预取对象为本卡分片（全量 ÷ `lmhead_tp_size`）；`lmhead_tp_size ≥ 16` 时分片逼近或小于 64MB，需按 `min(预取大小, 分片字节数)` 限制后再扩展到更大的 tp 配置 |

生效需同时满足 `exe_mode: npugraph_ex`、`enable_multi_streams: True` 与 `enable_prefetch: True`，且全部点位仅在 Decode 阶段下发。

流与事件的组织方式为：每个模型实例显式创建一条预取流，该平台的全部点位共用；事件槽按消费模块分配，每个 Attention 持有 4 个事件槽（0 = wq_a、1 = wq_b、2 = wo_a、3 = wo_b），lm_head 预取使用独立事件槽，避免跨模块争用。同步设计沿用 3.2.2 节的单向依赖，并在两个关键消费边界增加尾部同步：层循环结束后、hc_head 之前（A3），以及 lm_head matmul 消费之后（两平台，`forward_lm_head` 末尾），避免残余 CMO 与后续访存竞争。

调用形态上，各点位参数封装为 `PrefetchTarget`（weight、depend、ratio、fixed_size），由统一的 `_prefetch_weight` 完成事件同步与 CMO 下发；流缺失或预取大小无效时不产生 CMO：

```python
@dataclass
class PrefetchTarget:
    weight: torch.Tensor
    depend: torch.Tensor
    ratio: float = 0.0
    fixed_size: int = 0

def _prefetch_weight(pf_stream, pf_events, event_idx, target: PrefetchTarget):
    if target.fixed_size > 0:
        pf_size = target.fixed_size
    else:
        pf_size = int(target.weight.numel() * target.weight.element_size() * target.ratio)
    if pf_stream is None or pf_size <= 0:
        return
    pf_events[event_idx].record()
    with npu_stream_switch(True, pf_stream):
        pf_events[event_idx].wait()
        npu_prefetch(True, target.weight, target.depend, pf_size, 0)
```

#### 4.2.1 Attention 段逐点位预取

A3 在 Attention 段的 norm、rotary、通信、量化窗口内各下发一个 CMO；ASCEND_950 仅对 wq_a/wq_b 采用 0.8 比例预取，wo_a/wo_b 不预取。各点位的预取时机与重叠窗口如下，depend 均取窗口入口处的前序输出 Tensor：

| 预取点 | 目标权重 | 预取时机 | A3 | ASCEND_950 | 重叠窗口 |
| --- | --- | --- | --- | --- | --- |
| P_WQ_A | `wq_a.weight` | `DecoderLayer.forward`，HcPre(attn) 之后、attn_norm 之前（depend = hidden_states） | 2MB | 0.8 比例 | attn_norm |
| P_WQ_B | `wq_b.weight` | `mla_prolog_decode`，wq_a 计算之后、apply_norm_dynamic_quant 之前（depend = qa） | 2MB | 0.8 比例 | norm + wkv（mla_stream 并行）+ rotary |
| P_WO_A | `wo_a.weight` | `attn_post`，rotary 之前（depend = o） | 6MB | — | rotary + all2all 通信 |
| P_WO_B | `wo_b.weight` | `attn_post`，wo_a 计算之后、wo_b 调用之前（depend = o） | 2MB | — | wo_b.forward 内 `npu_dynamic_mx_quant` |

#### 4.2.2 lm_head 切片预取

lm_head 权重规模大，Decode 尾部 lm_head matmul 的权重读取占比高，两平台采用同一机制：以 64MB 固定切片预取本卡分片。预取点位于 `hc_head` 内 sigmoid 之后、加权求和之前，depend 取 sigmoid 输出；CMO 与加权求和、norm、allgather 重叠，消费点为 `forward_lm_head` 中的 lm_head matmul，消费完成后对预取流做尾部同步。

```mermaid
block-beta
  columns 17
  s0["主执行流"]:1 lin["hc_head linear"]:3 sig["sigmoid"]:2 sum["加权求和"]:3 n["norm"]:2 ag["allgather"]:3 lm["lm_head matmul"]:3
  s1["Prefetch"]:1 space:5 p["lm_head 分片切片 64MB"]:8 space:3

  p --> lm

  classDef hdr fill:#eee,stroke:#999,color:#000
  classDef main fill:#5B9BD5,color:#fff,stroke:#1F4E79
  classDef pre fill:#70AD47,color:#fff,stroke:#385723

  class s0,s1 hdr
  class lin,sig,sum,n,ag,lm main
  class p pre
```
