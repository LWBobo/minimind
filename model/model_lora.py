import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A  512×8
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B 8×512
        # 矩阵A高斯初始化 均值为0，标准差为0.02.
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    # 遍历模型中的所有子模块。
    # `name` 是模块名称，`module` 是对应的子模块实例。
    for name, module in model.named_modules():
        # 检查当前子模块是否是一个全连接层（`nn.Linear`）。
        # 进一步检查：该线性层的权重矩阵是方阵（行数等于列数）。
        # 通常，这种情况出现在注意力机制的 Query、Key、Value 矩阵中。
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建一个 LoRA 模块，用于将低秩矩阵适配到当前线性层。
            # 参数：
            # - 输入维度和输出维度（module.weight.shape[0] 和 module.weight.shape[1]）。
            # - rank 指定低秩矩阵的秩。
            # - `to(model.device)` 将 LoRA 模块移动到与模型相同的设备（如 GPU）。
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)

            # 将创建的 LoRA 模块绑定到线性层模块中，作为其一个属性（`lora`）。
            setattr(module, "lora", lora)
            # 保存线性层原始的 forward 方法，后续需要在新的 forward 方法中调用它。
            original_forward = module.forward

            # 显式绑定一个新的前向传播函数。
            # 定义线性层的新 forward 函数，包含 LoRA 的适配。
            # 参数：
            # - x: 输入张量。

            # - layer1: 原始线性层的 forward 方法。
            # - layer2: LoRA 模块的 forward 方法。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):

                # 新的 forward 方法返回两个部分的输出：
                # 1. 原始线性层（layer1）的输出。
                # 2. LoRA 模块（layer2）的输出。
                # 这两个部分的输出相加，形成最终输出。
                ### 这里将两个参数进行简单的相加，普遍做法是通过lora_alpha超参数来控制它的缩放程度。###
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora  # 将线性层的 forward 方法替换为新的 forward_with_lora 方法。


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
