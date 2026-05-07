# PEA 模型与框架对接问题分析

## 1. 框架的标准流程

### 训练调用链 (`main.py` CustomTrainer.train)

```
# 对于自定义模型 (不在 write_list 中的，如 pea):
pred, loss, losses_vis = self.model(volume, target, weight, criterion=self.criterion)
```

框架**只传 4 个参数**给 model.forward():
- `volume` (inputs): `[B, 1, D, H, W]`
- `target`: `List[Tensor]` — 由 DataLoader 的 `sample.out_target_l` 而来
- `weight`: `List[Tensor]` — 由 DataLoader 的 `sample.out_weight_l` 而来
- `criterion`: `Criterion` 对象 (从 `Criterion.build_from_cfg` 构建)

### 框架的 Criterion 对象

`Criterion` (来自 `connectomics/model/loss/criterion.py`) **不是**一个简单的 `WeightedMSE`。它是一个**复合损失管理器**:

```python
# Criterion.__call__ 签名:
def __call__(self, pred: Tensor, target: List[Tensor], weight: List[Tensor]) -> Tuple[Tensor, dict]:
```

- **pred**: 单个 Tensor (或 OrderedDict)
- **target**: `List[Tensor]`，是按 `target_opt` 分组的标签列表
- **weight**: `List[Tensor]`，是按 target 分组的权重列表的列表
- **返回**: `(loss, losses_vis)` — 标量损失 + 可视化字典

内部流程: pred → SplitActivation 切分 → 对每个 target 遍历 loss_fn → 加权求和

### `return_loss` 装饰器 (标准用法)

```python
@return_loss
def forward(self, inputs):
    return pred  # 单个 Tensor
# 被装饰后变为:
# def forward(self, inputs, target=None, weight=None, criterion=None):
#     pred = original_forward(self, inputs)
#     if criterion is None: return pred
#     loss, losses_vis = criterion(pred, target, weight)  # Criterion.__call__
#     return pred, loss, losses_vis
```

## 2. pea.py 当前的 forward 签名

```python
def forward(self, inputs, target=None, weight=None, criterion=None,
            ema_inputs=None, rules=None,
            down1=None, down2=None, down3=None, down4=None):
```

## 3. 具体问题清单

### 问题 A: pea.py 没有使用 `@return_loss` 装饰器

PEA 的 forward 是**完全手写的损失计算逻辑**，没有用 `@return_loss`。这本身不是错误（自定义模型可以自己算 loss），但导致它必须自己正确处理框架传入的 criterion。

### 问题 B: criterion 类型不匹配 ⚠️ **核心问题**

框架传入的 `self.criterion` 是 **`Criterion` 对象**，而 pea.py 里把它当作 **`WeightedMSE` 实例**来调用。

| | 框架 Criterion.__call__ | pea.py 中的调用方式 |
|---|---|---|
| **签名** | `criterion(pred, target: List, weight: List)` | `criterion(affs, target_slice, weight_slice)` |
| **target 类型** | `List[Tensor]` | 单个 Tensor 切片 |
| **weight 类型** | `List[Tensor]` (嵌套) | 单个 Tensor 切片 |
| **pred 含义** | 完整模型输出 Tensor | 单方向亲和力 `[B,1,D',H',W']` |

pea.py 中的所有损失函数 (`embedding_loss_norm5`, `embedding_loss_norm1`, `ema_embedding_loss_norm5`) 都这样调用:
```python
criterion(affs, target[:, order:order+1, shift:, :, :], weightmap[:, order:order+1, shift:, :, :])
```
这是 `WeightedMSE(pred, target, weight)` 的调用方式，**不是** `Criterion(pred, target_list, weight_list)` 的调用方式。

**结果**: 框架传入 Criterion 对象后，调用会报错（因为 Criterion 内部会对 target 做 `self.to_torch(target[i])` 等 List 操作）。

### 问题 C: target/weight 类型不匹配

框架 DataLoader 返回的 target 和 weight 类型:
- `target = sample.out_target_l` → `List[Tensor]`
- `weight = sample.out_weight_l` → `List[Tensor]`

但 pea.py forward 中直接把 target 当作 `[B, 12, D, H, W]` 的单个 Tensor 来索引:
```python
loss_emb, affs_emb = embedding_loss_norm5(embedding, target, weight, criterion)
```
如果框架传来的是 List，这里会出错。

### 问题 D: 额外参数无法从框架获取

pea.py forward 需要这些额外参数，但框架只传 4 个:
- `ema_inputs` — EMA 增强输入 (CCM 路)
- `rules` — 翻转规则
- `down1/down2/down3/down4` — 各层下采样标签

框架的训练循环 `self.model(volume, target, weight, criterion=self.criterion)` **不会传**这些参数，它们全部为 `None`。

**结果**:
- CCM 损失始终为 0 (因为 `ema_inputs is None`)
- EPM 损失会报错 (因为 `down4[:, :3]` 对 `None` 取下标)

### 问题 E: pea.py 自带 WeightedMSE 与框架的 WeightedMSE 重复

pea.py 第 55-74 行定义了自己的 `WeightedMSE`，但框架的 `connectomics/model/loss/loss.py` 也有一个。两者功能相同，但 pea.py 的版本用了 `.cuda()` 硬编码而非 `.to(pred.device)`。

## 4. 总结

| # | 问题 | 严重程度 | 说明 |
|---|---|---|---|
| A | 未用 @return_loss | 低 | 可以自己算 loss，但需要正确适配 |
| B | criterion 类型不匹配 | **致命** | 框架传 Criterion 对象，pea 当 WeightedMSE 用，调用签名不兼容 |
| C | target/weight 类型不匹配 | **致命** | 框架传 List，pea 当单个 Tensor 用 |
| D | 额外参数无法传入 | **致命** | ema_inputs/rules/down1-4 框架不会传，EPM 会 NoneType 报错 |
| E | WeightedMSE 重复定义 | 低 | 功能冗余 + .cuda() 硬编码 |

## 5. 可能的修复方向

**方案一: pea.py 适配框架 (推荐)**
1. 从框架 criterion (Criterion 对象) 中提取底层 `WeightedMSE` loss_fn，或直接在 forward 里自己实例化 `WeightedMSE()`
2. 从框架 DataLoader 的 target/weight (List) 中正确解包出需要的 Tensor
3. 将 ema_inputs/rules/down1-4 的生成逻辑移入 DataLoader 或在 forward 中从 target list 中获取

**方案二: 修改框架适配 pea**
1. 在 `CustomTrainer.train()` 中为 pea 模型特殊处理，传入额外参数
2. 构建 pea 专用的 DataLoader，输出 ema_inputs/rules/down1-4
3. 直接传 `WeightedMSE()` 而非 `Criterion` 对象

两种方案各有取舍，方案一对框架侵入性小，方案二对 pea 原始逻辑保留更完整。
