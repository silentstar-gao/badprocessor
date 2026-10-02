# BadProcessor

用于 C 代码补全实验的 logits Processor、FIM 适配器，以及基于 vLLM 的推理脚本。

项目包含三类攻击 Processor：

- `error_branch`：检测错误分支并插入资源释放调用；
- `goto_fail`：检测 goto 清理路径并插入清理序列；
- `array_rollback`：检测数组分配失败后的回滚路径。

## 目录结构

```text
processors/       普通 Transformers logits Processor
processors_fim/   vLLM/FIM 适配器
tools/            推理和监控脚本
```

模型权重、数据集、推理输出、恢复的项目源码和 Python 缓存不包含在仓库中，由 `.gitignore` 排除。

## 环境依赖

- Python 3.10+
- PyTorch
- Transformers
- vLLM 0.4.x

FlashAttention 不是必需依赖；未安装时 vLLM 会使用其他 attention backend，例如 XFormers。

检查源码语法：

```bash
python -m py_compile \
  processors/*.py processors_fim/*.py \
  tools/processor_probability_monitor.py \
  tools/run_vllm_multimodel_attack_monitored.py \
  tools/run_vllm_multimodel_attack.py tools/run_vllm_clean.py
```

## 概率监视器

`tools/processor_probability_monitor.py` 是一个外层包装器，不修改原 Processor：

```python
from processor_probability_monitor import wrap_processor

monitored = wrap_processor(
    processor,
    tokenizer,
    log_path="processor_probability_events.jsonl",
    temperature=0.4,
    metadata={"request_key": request_key},
)
```

监视器检测 Processor 是否将 logits 修改为“只有一个有限值”，并将每个强制 token 追加到 JSONL。事件包含 token、位置、强制前后的 raw softmax 概率和温度调整后的概率，不记录完整词表 logits。

同一个请求可能产生多个事件，因为一个 payload 可能由多个 token 组成。应按 `request_key` 聚合事件，不能假设每个任务固定产生一个或三个事件。

## 推理脚本

### 带概率监视器的多模型 Attack

入口：`tools/run_vllm_multimodel_attack_monitored.py`

默认配置：

- 四个本地模型，按顺序运行；
- 温度 `0.4`；
- 每个任务重复 `10` 次；
- `max_new_tokens=256`；
- `top_p=0.95`；
- `max_num_seqs=30`；
- BF16；
- 全部模型完成后执行 `/usr/bin/shutdown`。

开发测试时必须关闭自动关机：

```bash
python -u tools/run_vllm_multimodel_attack_monitored.py \
  --models /path/to/model \
  --dataset /path/to/trigger_completion_tasks.partial.jsonl \
  --sample-ids SAMPLE_ID_1 SAMPLE_ID_2 SAMPLE_ID_3 \
  --temperature 0.4 \
  --repeats 1 \
  --output outputs/monitor-smoke \
  --no-shutdown
```

完整运行：

```bash
python -u tools/run_vllm_multimodel_attack_monitored.py
```

实时查看进度：

```bash
screen -r vllm-attack-monitor-t04
```

每个模型目录会写入：

```text
results.jsonl                       请求结果和 completed/error 状态
skipped.jsonl                       预处理或上下文超长记录
processor_probability_events.jsonl  强制 token 概率事件
summary.json                        模型汇总
```

### 原始 Attack 和 Clean 基线

- `tools/run_vllm_multimodel_attack.py`：不使用概率监视器的原始多模型 Attack runner；
- `tools/run_vllm_clean.py`：不加载攻击 Processor 的 Clean baseline。

## 断点续跑和异常处理

`completed` 和 `error` 都视为终态，重新运行时会跳过已记录请求。上下文超长、tokenizer 或 prompt 预处理异常会写入 `skipped.jsonl`；入队和推理异常会写入 `results.jsonl`，不会伪装成成功补全。

## 安全提示

完整 runner 默认包含自动关机动作。首次运行、调试或修改参数时使用 `--no-shutdown`。确认结果文件和 `summary.json` 正常后，再使用默认配置运行。
