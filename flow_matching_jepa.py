"""两阶段 Flow Matching + JEPA 字符模型：先学习潜空间，再学习从噪声生成潜空间。"""

import argparse  # 解析训练步数等命令行参数。
import math  # 构造时间 Embedding 时计算对数。
import torch  # 张量、随机数与设备管理。
import torch.nn as nn  # 神经网络模块。
import torch.nn.functional as F  # 损失函数与相似度函数。


# 下面是很小的教学语料；更大的数据集应替换这里，而不是修改模型结构。
CORPUS = (
    "人工智能通过数据学习世界的规律。语言模型阅读文字，并预测下一个字符。"
    "联合嵌入预测架构在潜在空间中表示未来，从而减少对像素级细节的依赖。"
    "机器学习需要数据、模型、损失函数和优化器。训练让损失逐渐降低。"
) * 30

SEQ_LEN = 24  # 每个训练样本固定为 24 个字符，避免首版引入长度预测器。
BATCH_SIZE = 16  # 每批样本数量。
DIM = 32  # Token Embedding 与潜空间的维度。
HEADS = 4  # 注意力头数，必须能整除 DIM。
EMA_DECAY = 0.99  # 目标编码器的 EMA 更新系数。


class ContextEncoder(nn.Module):
    """把离散 Token 序列编码成连续的上下文潜变量。"""

    def __init__(self, vocab_size, max_len=SEQ_LEN, dim=DIM):
        super().__init__()  # 初始化 nn.Module。
        self.embedding = nn.Embedding(vocab_size, dim)  # 把 Token ID 转成连续向量。
        self.position = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)  # 学习位置信息。
        layer = nn.TransformerEncoderLayer(  # 双向编码当前样本中的可见 Token。
            d_model=dim,
            nhead=HEADS,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(  # 堆叠编码层，禁用当前结构用不到的嵌套张量。
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)  # 稳定潜变量尺度，但不能单独防止语义塌缩。

    def forward(self, token_ids):
        h = self.embedding(token_ids) + self.position[:, : token_ids.size(1)]  # 加入位置。
        return self.norm(self.transformer(h))  # 输出 [B,L,D] 的连续潜变量。


class FlowBlock(nn.Module):
    """一个同时读取带噪目标和上下文条件的 Transformer 块。"""

    def __init__(self, dim=DIM):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)  # 自注意力前的归一化。
        self.self_attn = nn.MultiheadAttention(dim, HEADS, batch_first=True)  # 目标位置间交互。
        self.norm_cross = nn.LayerNorm(dim)  # 交叉注意力前的归一化。
        self.cross_attn = nn.MultiheadAttention(dim, HEADS, batch_first=True)  # 读取上下文 c。
        self.norm_ff = nn.LayerNorm(dim)  # 前馈网络前的归一化。
        self.ff = nn.Sequential(  # 对每个目标位置做非线性变换。
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, z, context):
        h = self.norm_self(z)  # Pre-LN 有助于稳定训练。
        attn, _ = self.self_attn(h, h, h, need_weights=False)  # 目标潜变量内部交互。
        z = z + attn  # 残差连接。
        h = self.norm_cross(z)
        cross, _ = self.cross_attn(h, context, context, need_weights=False)  # 注入上下文。
        z = z + cross
        z = z + self.ff(self.norm_ff(z))  # 前馈残差分支。
        return z


def timestep_embedding(t, dim):
    """把连续时间 t∈[0,1] 转成 Transformer 可使用的正弦向量。"""
    t = t.reshape(-1)  # 统一成 [B]。
    half = dim // 2  # 一半用于 sin，一半用于 cos。
    frequency = torch.exp(  # 不同维度对应不同时间频率。
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angle = t[:, None] * 1000.0 * frequency[None, :]  # 放大 t，区分不同噪声程度。
    return torch.cat([angle.sin(), angle.cos()], dim=-1)  # 返回 [B,D]。


class FlowPredictor(nn.Module):
    """学习 v(z_t,t,c)，输出从噪声指向目标潜变量的速度场。"""

    def __init__(self, max_len=SEQ_LEN, dim=DIM, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)  # 投影带噪潜变量 z_t。
        self.position = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)  # 目标位置编码。
        self.time_mlp = nn.Sequential(  # 非线性变换时间 Embedding。
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([FlowBlock(dim) for _ in range(num_layers)])  # Flow 主干。
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)  # 输出形状必须与 z_t 完全相同。

    def forward(self, z_t, t, context):
        time = self.time_mlp(timestep_embedding(t, z_t.size(-1)))  # [B,D]。
        h = self.input_proj(z_t) + self.position[:, : z_t.size(1)] + time[:, None, :]
        for block in self.blocks:  # 依次执行自注意力、交叉注意力和前馈网络。
            h = block(h, context)
        return self.output(self.norm(h))  # 返回预测速度 [B,L,D]。


class FlowJEPA(nn.Module):
    """JEPA 双编码器、第一阶段预测器、Flow 预测器与文本解码器。"""

    def __init__(self, vocab_size):
        super().__init__()
        self.online = ContextEncoder(vocab_size)  # 在线编码器，第一阶段接收梯度。
        self.target = ContextEncoder(vocab_size)  # 目标编码器，仅通过 EMA 更新。
        self.target.load_state_dict(self.online.state_dict())  # 初始权重完全一致。
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)  # 阻止优化器直接更新目标编码器。
        self.jepa_predictor = nn.Sequential(  # 第一阶段的确定性潜变量预测器。
            nn.Linear(DIM, DIM * 2),
            nn.GELU(),
            nn.Linear(DIM * 2, DIM),
        )
        self.flow = FlowPredictor()  # 第二阶段的 Flow Matching 预测器。
        self.decoder = nn.Linear(DIM, vocab_size)  # 从每个潜变量位置还原文本 Token。


@torch.no_grad()  # EMA 更新不需要计算图。
def update_target(model, decay=EMA_DECAY):
    """ξ ← μξ + (1−μ)θ，让训练目标缓慢变化。"""
    for online_parameter, target_parameter in zip(
        model.online.parameters(), model.target.parameters()
    ):
        target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)


def make_corrupted_context(clean, mask_id):
    """随机遮住 20%～95% 的 Token，构造 JEPA 的上下文输入。"""
    batch_size = clean.size(0)
    ratio = 0.2 + 0.75 * torch.rand(batch_size, 1, device=clean.device)  # 每样本一个比例。
    mask = torch.rand(clean.shape, device=clean.device) < ratio  # 决定哪些位置被遮住。
    context = clean.clone()
    context[mask] = mask_id  # 只替换上下文，干净目标 clean 保持不变。
    return context


def variance_loss(z):
    """VICReg 风格的方差约束：避免某一维在所有 Token 上几乎不变。"""
    flat = z.reshape(-1, z.size(-1))  # 合并 Batch 与位置维度。
    std = flat.std(dim=0, unbiased=False)  # 检查每个潜空间维度的变化程度。
    return F.relu(1.0 - std).mean()  # std 小于 1 的维度受到惩罚。


def reconstruction_loss(decoder, z, clean):
    """让潜变量保留还原文本所需的信息，直接抑制常数潜变量塌缩。"""
    logits = decoder(z)  # [B,L,D] → [B,L,V]。
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), clean.reshape(-1))


def latent_std(z):
    """打印平均潜变量标准差；接近 0 是塌缩警讯，但大于 0 不代表一定没有塌缩。"""
    return z.reshape(-1, z.size(-1)).std(dim=0, unbiased=False).mean().item()


def sample_batch(data, batch_size, device):
    """从语料随机截取固定长度的字符窗口。"""
    starts = torch.randint(0, len(data) - SEQ_LEN, (batch_size,))
    return torch.stack([data[start : start + SEQ_LEN] for start in starts]).to(device)


def train(model, data, mask_id, device, stage1_steps, stage2_steps):
    """先训练 JEPA 表征与解码器，再冻结编码器训练 Flow。"""
    model.train()
    stage1_parameters = (
        list(model.online.parameters())
        + list(model.jepa_predictor.parameters())
        + list(model.decoder.parameters())
    )
    optimizer = torch.optim.AdamW(stage1_parameters, lr=2e-3)

    print("阶段一：学习可重建、可预测的 JEPA 潜空间")
    for step in range(1, stage1_steps + 1):
        clean = sample_batch(data, BATCH_SIZE, device)  # 干净目标 x_y。
        context_ids = make_corrupted_context(clean, mask_id)  # 掩码上下文 x_c。
        context = model.online(context_ids)  # 在线上下文表示 c。
        online_clean = model.online(clean)  # 在线干净表示，用于把梯度送入编码器。
        with torch.no_grad():
            target = model.target(clean)  # 目标 z_1 停止梯度。

        prediction = model.jepa_predictor(context)  # 从掩码上下文预测目标潜变量。
        jepa_loss = (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()
        recon_loss = reconstruction_loss(model.decoder, online_clean, clean)
        var_loss = variance_loss(online_clean)  # 抑制特征维度塌缩。
        loss = jepa_loss + recon_loss + 0.1 * var_loss

        optimizer.zero_grad()  # 清空上一批梯度。
        loss.backward()  # 更新在线编码器、JEPA 预测器和解码器。
        optimizer.step()
        update_target(model)  # 优化后再同步目标编码器。

        if step == 1 or step % 50 == 0 or step == stage1_steps:
            print(
                f"step {step:4d} | JEPA {jepa_loss.item():.4f} | "
                f"重建 {recon_loss.item():.4f} | 方差 {var_loss.item():.4f} | "
                f"潜变量 std {latent_std(target):.4f}"
            )

    # 固定潜空间后，Flow 不必追逐不断移动的目标，这是第二阶段的重要稳定措施。
    for parameter in model.online.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(model.flow.parameters()) + list(model.decoder.parameters()), lr=1e-3
    )

    print("\n阶段二：冻结编码器，训练条件 Flow Matching")
    for step in range(1, stage2_steps + 1):
        clean = sample_batch(data, BATCH_SIZE, device)
        context_ids = make_corrupted_context(clean, mask_id)
        with torch.no_grad():
            context = model.online(context_ids)  # 条件 c。
            target = model.target(clean)  # 干净潜变量 z_1。

        noise = torch.randn_like(target)  # z_0 ~ N(0,I)。
        t = torch.rand(clean.size(0), device=device)  # 每个样本独立采样时间 t。
        z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target  # 线性插值。
        velocity_target = target - noise  # 线性概率路径对应的真实速度。
        velocity = model.flow(z_t, t, context)  # 条件速度场预测。
        flow_loss = (velocity - velocity_target).pow(2).mean()  # Flow Matching 的 MSE。
        recon_loss = reconstruction_loss(model.decoder, target, clean)  # 继续适配目标潜空间。
        loss = flow_loss + recon_loss

        optimizer.zero_grad()
        loss.backward()  # 只更新 Flow 预测器与解码器，编码器保持冻结。
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == stage2_steps:
            print(f"step {step:4d} | Flow {flow_loss.item():.4f} | 重建 {recon_loss.item():.4f}")


@torch.no_grad()
def generate(model, prompt, stoi, itos, mask_id, device, inference_steps):
    """用 Euler 方法从 t=0 的噪声积分到 t=1 的目标潜变量。"""
    model.eval()
    prompt = prompt[:SEQ_LEN]  # 固定长度版本只保留模型能容纳的前缀。
    context_ids = torch.full((1, SEQ_LEN), mask_id, dtype=torch.long, device=device)
    for index, character in enumerate(prompt[:SEQ_LEN]):
        context_ids[0, index] = stoi[character]  # 可见前缀，其余位置保持 [MASK]。
    context = model.online(context_ids)

    z = torch.randn(1, SEQ_LEN, DIM, device=device)  # 从高斯噪声开始。
    dt = 1.0 / inference_steps
    for step in range(inference_steps):
        t = torch.full((1,), step * dt, device=device)  # 当前轨迹时间。
        z = z + dt * model.flow(z, t, context)  # Euler：z ← z + Δt·v(z,t,c)。

    logits = model.decoder(z)  # 并行解码所有位置。
    logits[..., mask_id] = -torch.inf  # [MASK] 只属于输入，不允许作为输出文本。
    generated = logits.argmax(dim=-1)[0].tolist()
    generated[: len(prompt)] = context_ids[0, : len(prompt)].tolist()  # 保留给定前缀。
    return "".join(itos[index] for index in generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-steps", type=int, default=200)  # 表征预训练步数。
    parser.add_argument("--stage2-steps", type=int, default=500)  # Flow 训练步数。
    parser.add_argument("--inference-steps", type=int, default=30)  # Euler 积分数。
    args = parser.parse_args()

    torch.manual_seed(42)  # 固定随机种子，便于复现实验。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chars = sorted(set(CORPUS))
    stoi = {character: index for index, character in enumerate(chars)}  # 字符到 ID。
    itos = {index: character for character, index in stoi.items()}  # ID 到字符。
    mask_id = len(chars)  # 额外分配一个输入专用 [MASK] Token。
    data = torch.tensor([stoi[character] for character in CORPUS], dtype=torch.long)

    model = FlowJEPA(vocab_size=len(chars) + 1).to(device)
    print(f"device: {device} | vocab: {len(chars)} + [MASK] | sequence: {SEQ_LEN}")
    train(model, data, mask_id, device, args.stage1_steps, args.stage2_steps)

    for prompt in ["人工智能", "机器学习"]:
        text = generate(model, prompt, stoi, itos, mask_id, device, args.inference_steps)
        print(f"\n提示：{prompt}\n生成：{text}")


if __name__ == "__main__":  # 只有直接执行脚本时才训练，导入模块不会自动训练。
    main()
