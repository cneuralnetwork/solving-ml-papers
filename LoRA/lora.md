# 🧾 LoRA Module: Low-Rank Adaptation for Neural Network Weights

This code defines a custom PyTorch module implementing **Low-Rank Adaptation (LoRA)**, which is often used to adapt pretrained models efficiently by injecting low-rank matrices into the weight matrices, instead of fine-tuning the full model.

---

## 🔧 **What the Module Does**

* Adds a **trainable low-rank modification** to an existing weight matrix `wts`.
* Instead of fine-tuning `wts` directly, it learns two small matrices `A` and `B` whose product is added to `wts`.
* The goal is **parameter-efficient fine-tuning**.

---

## 📦 Code Breakdown

```python
import torch
import torch.nn as nn

class LoRA(nn.Module):
```

Defines a subclass of `nn.Module` called `LoRA`.

---

### 🔨 `__init__` method

```python
def __init__(self, in_f, out_f, rank=1, alpha=1, device: str = "cpu"):
```

* `in_f`: Input feature dimension of the original weight matrix.
* `out_f`: Output feature dimension of the original weight matrix.
* `rank`: Rank of the low-rank decomposition (smaller means fewer parameters).
* `alpha`: Scaling factor to balance the update magnitude.
* `device`: Device to store the parameters (`"cpu"` or `"cuda"`).

```python
super().__init__()
self.A = nn.Parameter(torch.zeros((rank, out_f)).to(device))
self.B = nn.Parameter(torch.zeros((in_f, rank)).to(device))
```

* `A` and `B` are the low-rank trainable matrices.
* The effective update to the weights is `B @ A`, with shape `(in_f, out_f)`.

```python
self.scale = alpha / rank
self.en = True
```

* `scale` adjusts the update size (as recommended in LoRA paper).
* `en`: A flag to enable or disable LoRA injection.

---

### 🧠 `forward` method

```python
def forward(self, wts):
```

* Takes the original weight matrix `wts` (from some layer in a neural network).
* Returns the modified weight matrix depending on whether LoRA is enabled.

```python
if self.en:
    return wts + torch.matmul(self.B, self.A).view(wts.shape) * self.scale
else:
    return wts
```

* If enabled:

  * Computes the low-rank update `B @ A`.
  * Reshapes it to match `wts` (typically already `(in_f, out_f)`).
  * Scales it by `alpha/rank`.
  * Adds it to the original weights.
* If disabled: returns the original `wts` unchanged.

---

## ✅ Example Use Case

```python
w = torch.randn(128, 256)  # Example weight matrix
lora = LoRA(in_f=128, out_f=256, rank=4, alpha=16, device='cpu')

new_w = lora(w)  # Modified weights using LoRA
```

You can plug `new_w` into another linear layer, or use it to override pretrained weights.

---

## ✍️ Summary

| Component      | Purpose                                        |
| -------------- | ---------------------------------------------- |
| `self.A`       | Rank-`r` matrix for projecting to output dim   |
| `self.B`       | Rank-`r` matrix for projecting from input dim  |
| `self.scale`   | Scaling factor `alpha / rank`                  |
| `forward(wts)` | Returns adapted weights: `wts + B @ A * scale` |
| `self.en`      | Enables or disables the LoRA modification      |
