"""两阶段 Flow Matching + JEPA 字符模型：连续潜空间生成 + 离散瓶颈。

对应 math.md §3.6 的完整实现：
- 前向压缩 p(T|Z)：FSQ / VQ / RVQ 离散量化器（也可关闭）；
- 反向模糊回溯 p(Z|T)：条件 Flow 的摊销采样与多轨迹后验方差；
- 并行解码：Flow 一次解码，或 MaskGIT 式迭代并行填充。
"""

import argparse  # 解析训练步数与量化器配置。
import math  # 构造时间 Embedding、MaskGIT 提交步数。
import torch  # 张量、随机数与设备管理。
import torch.nn as nn  # 神经网络模块。
import torch.nn.functional as F  # 损失函数与相似度函数。
from contextlib import nullcontext  # 在 CPU 上提供 no-op 的 autocast 上下文。


# 下面是很小的教学语料；更大的数据集应替换这里，而不是修改模型结构。
CORPUS = (
    "人工智能通过数据学习世界的规律。语言模型阅读文字，并预测下一个字符。"
    "联合嵌入预测架构在潜在空间中表示未来，从而减少对像素级细节的依赖。"
    "机器学习需要数据、模型、损失函数和优化器。训练让损失逐渐降低。"
) * 30

SEQ_LEN = 1024  # 每个训练样本固定为 1024 个 token（BPE 词表，后续可升 4096）。
BATCH_SIZE = 16  # 每批样本数量。
DIM = 32  # Token Embedding 与潜空间的维度。
HEADS = 4  # 注意力头数，必须能整除 DIM。
EMA_DECAY = 0.99  # 目标编码器的 EMA 更新系数。
TOKENIZER_NAME = "inclusionAI/LLaDA2.1-mini"  # 借用的 BPE 词表，vocab=157184。


def load_tokenizer(name=TOKENIZER_NAME):
    """加载 BPE tokenizer（LLaDA2.1-mini 词表，vocab=157184）。

    惰性导入 transformers，避免未安装该依赖时导入本模块就失败。
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def enable_flash_attention():
    """在 CUDA 上优先启用 FlashAttention-2 / memory-efficient SDPA 内核。

    torch 的 scaled_dot_product_attention 会自动选择最快后端；这里显式开启 Flash
    与 memory-efficient，并关闭低效的 math 后端，保证长上下文训练走 FlashAttention。
    """
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)


class ContextEncoder(nn.Module):
    """把离散 Token 序列编码成连续的上下文潜变量。"""

    def __init__(self, vocab_size, max_len=SEQ_LEN, dim=DIM, n_heads=HEADS, num_layers=1):
        super().__init__()  # 初始化 nn.Module。
        self.embedding = nn.Embedding(vocab_size, dim)  # 把 Token ID 转成连续向量。
        self.position = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)  # 学习位置信息。
        layer = nn.TransformerEncoderLayer(  # 双向编码当前样本中的可见 Token。
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(  # 堆叠编码层，禁用当前结构用不到的嵌套张量。
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)  # 稳定潜变量尺度，但不能单独防止语义塌缩。

    def forward(self, token_ids):
        h = self.embedding(token_ids) + self.position[:, : token_ids.size(1)]  # 加入位置。
        return self.norm(self.transformer(h))  # 输出 [B,L,D] 的连续潜变量。


class FlowBlock(nn.Module):
    """一个同时读取带噪目标和上下文条件的 Transformer 块。

    推理时上下文 c 在 ODE 积分各步保持不变，因此交叉注意力的 K/V 可预计算一次
    并复用（KV 缓存），每步只重算随 z 变化的 query 与自注意力。
    """

    def __init__(self, dim=DIM, n_heads=HEADS):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)  # 自注意力前的归一化。
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)  # 目标位置间交互。
        self.norm_cross = nn.LayerNorm(dim)  # 交叉注意力前的归一化。
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)  # 读取上下文 c。
        self.norm_ff = nn.LayerNorm(dim)  # 前馈网络前的归一化。
        self.ff = nn.Sequential(  # 对每个目标位置做非线性变换。
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.cross_k = None  # 缓存交叉注意力的 key 投影。
        self.cross_v = None  # 缓存交叉注意力的 value 投影。

    def cache_context(self, context):
        """预计算并缓存交叉注意力的 K/V 投影，供 ODE 积分各步复用。"""
        weight = self.cross_attn.in_proj_weight  # [3*dim, dim]，q/k/v 拼在一起。
        bias = self.cross_attn.in_proj_bias  # [3*dim]。
        dim = self.cross_attn.embed_dim
        # context 在积分各步不变，K/V 只需算一次。
        self.cross_k = self._split_heads(F.linear(context, weight[dim : 2 * dim], bias[dim : 2 * dim]))
        self.cross_v = self._split_heads(F.linear(context, weight[2 * dim : 3 * dim], bias[2 * dim : 3 * dim]))

    def _split_heads(self, x):
        """[B,L,dim] → [B,HEADS,L,head_dim]，与 MultiheadAttention 内部一致。"""
        batch, length, _ = x.shape
        return x.view(
            batch, length, self.cross_attn.num_heads, self.cross_attn.head_dim
        ).transpose(1, 2)

    def _cross_attn_cached(self, z):
        """用缓存的 K/V 计算交叉注意力，query 仍由当前 z 产生。"""
        weight = self.cross_attn.in_proj_weight
        bias = self.cross_attn.in_proj_bias
        dim = self.cross_attn.embed_dim
        query = self._split_heads(F.linear(self.norm_cross(z), weight[:dim], bias[:dim]))
        out = F.scaled_dot_product_attention(
            query, self.cross_k, self.cross_v, dropout_p=self.cross_attn.dropout
        )
        batch, length, _ = z.shape
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.cross_attn.out_proj(out)

    def forward(self, z, context):
        h = self.norm_self(z)  # Pre-LN 有助于稳定训练。
        attn, _ = self.self_attn(h, h, h, need_weights=False)  # 目标潜变量内部交互。
        z = z + attn  # 残差连接。
        if context is None and self.cross_k is not None:
            z = z + self._cross_attn_cached(z)  # 推理路径：复用缓存的 K/V。
        else:
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

    def __init__(self, max_len=SEQ_LEN, dim=DIM, n_heads=HEADS, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)  # 投影带噪潜变量 z_t。
        self.position = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)  # 目标位置编码。
        self.time_mlp = nn.Sequential(  # 非线性变换时间 Embedding。
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([FlowBlock(dim, n_heads) for _ in range(num_layers)])  # Flow 主干。
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)  # 输出形状必须与 z_t 完全相同。

    def cache_context(self, context):
        """预计算各块交叉注意力的 K/V；context 在 ODE 积分各步保持不变。"""
        for block in self.blocks:
            block.cache_context(context)

    def forward(self, z_t, t, context=None):
        time = self.time_mlp(timestep_embedding(t, z_t.size(-1)))  # [B,D]。
        h = self.input_proj(z_t) + self.position[:, : z_t.size(1)] + time[:, None, :]
        for block in self.blocks:  # 依次执行自注意力、交叉注意力和前馈网络。
            h = block(h, context)
        return self.output(self.norm(h))  # 返回预测速度 [B,L,D]。


# ---------------------------------------------------------------------------
# 前向压缩 p(T|Z)：三种 GPU 友好的离散量化器
# ---------------------------------------------------------------------------


class IdentityQuantizer(nn.Module):
    """不量化：保留原始连续潜变量，作为对照组。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.kind = "none"
        self.codebook_size = None

    def forward(self, z):
        return z, z.new_zeros(()), None

    def dequantize(self, codes):
        return codes


class FSQ(nn.Module):
    """Finite Scalar Quantization：逐维取整，无码本，STE 回传。"""

    def __init__(self, dim, levels=5):
        super().__init__()
        # 奇数 levels 才能让码字关于 0 对称。
        assert levels >= 3 and levels % 2 == 1, "FSQ 的 levels 需为 >=3 的奇数"
        self.dim = dim
        self.levels = levels
        self.half = (levels - 1) // 2
        self.codebook_size = levels**dim
        self.kind = f"fsq(L={levels})"

    def forward(self, z):
        # 每个维度映射到 {-half, ..., half}，再除以 half 得到 [-1,1] 的量化值。
        scaled = torch.round(self.half * torch.tanh(z))
        z_q = scaled / self.half
        z_q = z + (z_q - z).detach()  # 直通估计：前向取整，反向视为恒等。
        codes = (scaled + self.half).long()  # 每维 0..levels-1。
        return z_q, z.new_zeros(()), codes

    def dequantize(self, codes):
        return (codes.float() - self.half) / self.half


class VectorQuantizer(nn.Module):
    """VQ：最近邻查表 + EMA 码本更新 + commitment 损失。"""

    def __init__(self, dim, codebook_size=256, commitment=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment = commitment
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.embedding.weight.requires_grad_(False)  # 码本只通过 EMA 更新。
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())
        self.kind = f"vq(K={codebook_size})"

    def forward(self, z):
        shape = z.shape
        flat = z.reshape(-1, self.dim)
        # 一次矩阵乘得到 [N,K] 距离矩阵，避免 Python 循环。
        distance = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * flat @ self.embedding.weight.t()
        )
        indices = distance.argmin(dim=1)
        z_q = self.embedding(indices).view(shape)
        loss = self.commitment * F.mse_loss(z_q.detach(), z)  # 只对编码器施加 commitment。
        if self.training:
            self._ema_update(flat, indices)
        z_q = z + (z_q - z).detach()  # 直通估计。
        return z_q, loss, indices.view(shape[:-1])

    @torch.no_grad()
    def _ema_update(self, flat, indices):
        onehot = F.one_hot(indices, self.codebook_size).float()
        count = onehot.sum(dim=0)
        self.cluster_size.mul_(self.decay).add_(count, alpha=1.0 - self.decay)
        sum_w = onehot.t() @ flat
        self.ema_w.mul_(self.decay).add_(sum_w, alpha=1.0 - self.decay)
        total = self.cluster_size.sum().clamp_min(self.eps)
        cluster = (self.cluster_size + self.eps) / (total + self.codebook_size * self.eps) * total
        self.embedding.weight.data.copy_(self.ema_w / cluster.unsqueeze(1))

    def dequantize(self, codes):
        return self.embedding(codes)


class ResidualVQ(nn.Module):
    """RVQ：逐级量化残差，级间少量串行、级内并行。"""

    def __init__(self, dim, codebook_size=256, num_quantizers=2, commitment=0.25):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantizer(dim, codebook_size, commitment) for _ in range(num_quantizers)]
        )
        self.codebook_size = codebook_size**num_quantizers
        self.kind = f"rvq(M={num_quantizers},K={codebook_size})"

    def forward(self, z):
        residual = z
        z_q = torch.zeros_like(z)
        loss = z.new_zeros(())
        codes = []
        for layer in self.layers:
            quantized, layer_loss, layer_codes = layer(residual)
            residual = residual - quantized
            z_q = z_q + quantized
            loss = loss + layer_loss
            codes.append(layer_codes)
        return z_q, loss, torch.stack(codes, dim=0)


def build_quantizer(kind, dim, levels=5, codebook_size=256, num_quantizers=2, commitment=0.25):
    """按命令行参数构造前向量化器。"""
    if kind == "none":
        return IdentityQuantizer(dim)
    if kind == "fsq":
        return FSQ(dim, levels=levels)
    if kind == "vq":
        return VectorQuantizer(dim, codebook_size=codebook_size, commitment=commitment)
    if kind == "rvq":
        return ResidualVQ(
            dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            commitment=commitment,
        )
    raise ValueError(f"未知量化器: {kind}")


class Decoder(nn.Module):
    """带位置混合的文本解码器：在量化潜变量上做自注意力，再逐位置还原 Token。

    取代原来的逐位置 nn.Linear，使解码器能利用不同位置间的上下文信息，
    从而对 Flow 采样产生的潜变量误差更鲁棒（不会因为某个位置的噪声就输出乱码）。
    """

    def __init__(self, dim, vocab_size, n_heads=HEADS, num_layers=2, max_len=SEQ_LEN):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)  # 学习位置信息。
        layer = nn.TransformerEncoderLayer(  # 自注意力 + 前馈，逐位置混合后再独立输出。
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(  # 堆叠解码层。
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)  # 逐位置输出词表 logits。

    def forward(self, z):
        h = z + self.position[:, : z.size(1)]  # 显式加入位置，避免量化后位置信息丢失。
        return self.head(self.norm(self.transformer(h)))  # [B,L,d] -> [B,L,V]。


class FlowJEPA(nn.Module):
    """JEPA 双编码器、第一阶段预测器、量化瓶颈、Flow 预测器与文本解码器。"""

    def __init__(
        self,
        vocab_size,
        dim=DIM,
        n_heads=HEADS,
        encoder_layers=1,
        flow_layers=2,
        decoder_layers=2,
        jepa_layers=4,
        max_len=SEQ_LEN,
        quantizer="fsq",
        levels=5,
        codebook_size=256,
        num_quantizers=2,
        commitment=0.25,
    ):
        super().__init__()
        self.dim = dim
        self.online = ContextEncoder(vocab_size, max_len=max_len, dim=dim, n_heads=n_heads, num_layers=encoder_layers)  # 在线编码器。
        self.target = ContextEncoder(vocab_size, max_len=max_len, dim=dim, n_heads=n_heads, num_layers=encoder_layers)  # 目标编码器。
        self.target.load_state_dict(self.online.state_dict())  # 初始权重完全一致。
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)  # 阻止优化器直接更新目标编码器。
        predictor_layers = [nn.Linear(dim, dim * 2), nn.GELU()]
        for _ in range(jepa_layers - 2):  # 中间隐藏层保持 2*dim 宽度。
            predictor_layers += [nn.Linear(dim * 2, dim * 2), nn.GELU()]
        predictor_layers.append(nn.Linear(dim * 2, dim))
        self.jepa_predictor = nn.Sequential(*predictor_layers)  # 第一阶段的确定性潜变量预测器。
        self.quantizer = build_quantizer(  # p(T|Z) 的离散化前端。
            quantizer,
            dim,
            levels=levels,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            commitment=commitment,
        )
        self.flow = FlowPredictor(max_len=max_len, dim=dim, n_heads=n_heads, num_layers=flow_layers)  # 第二阶段的 Flow Matching 预测器。
        self.decoder = Decoder(dim, vocab_size, n_heads=n_heads, num_layers=decoder_layers, max_len=max_len)  # 带位置混合的文本解码器。

    def quantize(self, z):
        """返回 (量化潜变量, 辅助损失, 离散码)。"""
        return self.quantizer(z)


@torch.no_grad()  # EMA 更新不需要计算图。
def update_target(model, decay=EMA_DECAY):
    """ξ ← μξ + (1−μ)θ，让训练目标缓慢变化。"""
    for online_parameter, target_parameter in zip(
        model.online.parameters(), model.target.parameters()
    ):
        target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)


def make_corrupted_context(clean, mask_id, min_ratio=0.2, max_ratio=0.95):
    """随机遮住一部分 Token，构造 JEPA 的上下文输入。

    阶段一用默认 [0.2, 0.95]；阶段二允许 min_ratio=0，使 Flow 也见过完整文本，
    从而支持 p(Z|T) 的后验采样接口。
    """
    batch_size = clean.size(0)
    ratio = min_ratio + (max_ratio - min_ratio) * torch.rand(
        batch_size, 1, device=clean.device
    )
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
    """让量化潜变量保留还原文本所需的信息，直接抑制常数潜变量塌缩。"""
    logits = decoder(z)  # [B,L,D] → [B,L,V]。
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), clean.reshape(-1))


def latent_std(z):
    """打印平均潜变量标准差；接近 0 是塌缩警讯，但大于 0 不代表一定没有塌缩。"""
    return z.reshape(-1, z.size(-1)).std(dim=0, unbiased=False).mean().item()


def sample_batch(data, batch_size, device):
    """从语料随机截取固定长度的字符窗口。"""
    starts = torch.randint(0, len(data) - SEQ_LEN, (batch_size,))
    return torch.stack([data[start : start + SEQ_LEN] for start in starts]).to(device)


def maybe_autocast(device):
    """CUDA 上启用 bf16 混合精度以利用 Tensor Core；CPU 上退化为 no-op。"""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train(model, data, mask_id, device, args):
    """先训练 JEPA 表征与量化解码器，再冻结编码器训练 Flow。"""
    model.train()
    stage1_parameters = (
        list(model.online.parameters())
        + list(model.jepa_predictor.parameters())
        + list(model.decoder.parameters())
        + [p for p in model.quantizer.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(stage1_parameters, lr=2e-3)

    print("阶段一：学习可重建、可预测的 JEPA 潜空间")
    for step in range(1, args.stage1_steps + 1):
        clean = sample_batch(data, args.batch_size, device)  # 干净目标 x_y。
        context_ids = make_corrupted_context(clean, mask_id)  # 掩码上下文 x_c。
        with maybe_autocast(device):
            context = model.online(context_ids)  # 在线上下文表示 c。
            online_clean = model.online(clean)  # 在线干净表示。
            with torch.no_grad():
                target = model.target(clean)  # 目标 z_1 停止梯度。

            prediction = model.jepa_predictor(context)  # 从掩码上下文预测目标潜变量。
            jepa_loss = (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()
            z_q, quant_loss, _ = model.quantize(online_clean)  # 前向压缩 p(T|Z)。
            recon_loss = reconstruction_loss(model.decoder, z_q, clean)
            # 解码器也要能读“掩码上下文 → JEPA 预测”的潜变量，否则 MaskGIT 会遇到分布偏移。
            z_pred_q, _, _ = model.quantize(prediction)
            masked_recon = reconstruction_loss(model.decoder, z_pred_q, clean)
            var_loss = variance_loss(online_clean)  # 抑制特征维度塌缩。
            loss = jepa_loss + recon_loss + 0.5 * masked_recon + 0.1 * var_loss + quant_loss

        optimizer.zero_grad()  # 清空上一批梯度。
        loss.backward()  # 更新在线编码器、预测器、量化器与解码器。
        optimizer.step()
        update_target(model)  # 优化后再同步目标编码器。

        if step == 1 or step % 50 == 0 or step == args.stage1_steps:
            print(
                f"step {step:4d} | JEPA {jepa_loss.item():.4f} | "
                f"重建 {recon_loss.item():.4f} | 掩码重建 {masked_recon.item():.4f} | "
                f"方差 {var_loss.item():.4f} | 量化 {quant_loss.item():.4f} | "
                f"潜变量 std {latent_std(target):.4f}"
            )

    # 固定潜空间后，Flow 不必追逐不断移动的目标，这是第二阶段的重要稳定措施。
    for parameter in model.online.parameters():
        parameter.requires_grad_(False)
    model.quantizer.eval()  # 阶段二冻结码本，避免 EMA 让目标分布漂移。
    optimizer = torch.optim.AdamW(
        list(model.flow.parameters()) + list(model.decoder.parameters()), lr=1e-3
    )

    print("\n阶段二：冻结编码器，训练条件 Flow Matching")
    for step in range(1, args.stage2_steps + 1):
        clean = sample_batch(data, args.batch_size, device)
        # 阶段二允许 0 掩码，让 Flow 同时见到完整文本，支持 T→Z 后验。
        context_ids = make_corrupted_context(clean, mask_id, min_ratio=0.0)
        with maybe_autocast(device):
            with torch.no_grad():
                context = model.online(context_ids)  # 条件 c。
                target = model.target(clean)  # 干净潜变量 z_1。
                target_q, _, _ = model.quantize(target)  # 解码器的训练目标。

            noise = torch.randn_like(target)  # z_0 ~ N(0,I)。
            t = torch.rand(clean.size(0), device=device)  # 每个样本独立采样时间 t。
            z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target  # 线性插值。
            velocity_target = target - noise  # 线性概率路径对应的真实速度。
            velocity = model.flow(z_t, t, context)  # 条件速度场预测。
            flow_loss = (velocity - velocity_target).pow(2).mean()  # Flow Matching 的 MSE。
            recon_loss = reconstruction_loss(model.decoder, target_q, clean)
            loss = flow_loss + recon_loss

        optimizer.zero_grad()
        loss.backward()  # 只更新 Flow 预测器与解码器，编码器保持冻结。
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.stage2_steps:
            print(f"step {step:4d} | Flow {flow_loss.item():.4f} | 重建 {recon_loss.item():.4f}")


@torch.no_grad()
def flow_sample(model, context, steps, device, solver="euler"):
    """从噪声积分到目标潜变量，batch 维可并行；支持 Euler 与 Heun 二阶求解器。

    Heun（二阶 RK）用两次速度评估换二阶精度，可在更少步数下达到同样质量，
    是 Flow Matching 中与 DPM-Solver++ 对应的更快速积分器。
    """
    batch = context.size(0)
    z = torch.randn(batch, context.size(1), context.size(-1), device=device)  # z_0 ~ N(0,I)。
    model.flow.cache_context(context)  # 上下文不变，交叉注意力 K/V 只算一次。
    dt = 1.0 / steps
    if solver == "heun":
        for step in range(steps):
            t = torch.full((batch,), step * dt, device=device)  # 当前轨迹时间。
            t_next = torch.full((batch,), (step + 1) * dt, device=device)
            k1 = model.flow(z, t)  # 复用缓存的 K/V。
            k2 = model.flow(z + dt * k1, t_next)
            z = z + 0.5 * dt * (k1 + k2)  # Heun：预测-校正二阶。
    else:
        for step in range(steps):
            t = torch.full((batch,), step * dt, device=device)  # 当前轨迹时间。
            z = z + dt * model.flow(z, t)  # Euler：z ← z + Δt·v(z,t,c)。
    return z


@torch.no_grad()
def decode_latents(model, z, mask_id, readout, temperature):
    """把潜变量解码成 Token：argmax 或按温度采样。"""
    z_q, _, _ = model.quantize(z)  # 推理时同样经过离散瓶颈。
    logits = model.decoder(z_q)  # [B,L,V]。
    logits[..., mask_id] = -torch.inf  # [MASK] 只属于输入，不允许作为输出文本。
    if readout == "sample":
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)  # p(T|Z)。
        return torch.distributions.Categorical(probs=probs).sample()
    return logits.argmax(dim=-1)


def prompt_ids(prompt, tokenizer, mask_id, device, seq_len=SEQ_LEN):
    """把前缀写成 [1,L] 的 Token 序列，其余位置填 [MASK]。"""
    ids = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    enc = tokenizer.encode(prompt, add_special_tokens=False)  # BPE 编码，不额外加 BOS/EOS。
    n = min(len(enc), seq_len)
    ids[0, :n] = torch.tensor(enc[:n], dtype=torch.long, device=device)
    return ids, n


@torch.no_grad()
def generate(model, prompt, tokenizer, mask_id, device, steps, solver, readout, temperature):
    """Flow 一次并行解码：编码前缀 → Flow 采样 → 量化 → 解码。"""
    model.eval()
    ids, n = prompt_ids(prompt, tokenizer, mask_id, device)
    context = model.online(ids)
    z = flow_sample(model, context, steps, device, solver)
    tokens = decode_latents(model, z, mask_id, readout, temperature)[0].tolist()
    tokens[:n] = ids[0, :n].tolist()  # 保留给定前缀的 token。
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    token_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return text, token_ids


@torch.no_grad()
def maskgit_decode(model, prompt, tokenizer, mask_id, device, steps):
    """MaskGIT 式并行解码：每步并行预测所有位置，再提交置信度最高的一批。"""
    model.eval()
    ids, _ = prompt_ids(prompt, tokenizer, mask_id, device)
    for step in range(steps):
        unknown = ids[0] == mask_id
        if not unknown.any():
            break
        h = model.online(ids)
        h = model.jepa_predictor(h)  # 先走阶段一的掩码预测路径，避免解码器分布偏移。
        z_q, _, _ = model.quantize(h)
        logits = model.decoder(z_q)
        logits[..., mask_id] = -torch.inf
        probs = logits.softmax(dim=-1)
        confidence, prediction = probs.max(dim=-1)

        remaining = int(unknown.sum().item())
        commit = max(1, math.ceil(remaining / (steps - step)))  # 均匀提交，近似并行细化。
        scores = confidence[0].masked_fill(~unknown, -torch.inf)
        chosen = scores.topk(commit).indices
        ids[0, chosen] = prediction[0, chosen]

    # 若仍有未填位置，用最后一次前向补齐。
    unknown = ids[0] == mask_id
    if unknown.any():
        h = model.online(ids)
        h = model.jepa_predictor(h)
        z_q, _, _ = model.quantize(h)
        logits = model.decoder(z_q)
        logits[..., mask_id] = -torch.inf
        ids[0, unknown] = logits.argmax(dim=-1)[0, unknown]
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


@torch.no_grad()
def posterior_demo(model, token_ids, tokenizer, mask_id, device, samples, steps, solver, readout, temperature):
    """T→Z 后验采样：同一文本采多条潜变量轨迹，度量并展示模糊性。"""
    if samples <= 1:
        return
    model.eval()
    context = model.online(token_ids).repeat(samples, 1, 1)  # 同一条件复制 M 份。
    z = flow_sample(model, context, steps, device, solver)  # [M,L,D]，batch 维并行。
    variance = z.var(dim=0, unbiased=False).mean().item()  # 后验离散度。
    tokens = decode_latents(model, z, mask_id, readout, temperature)
    variants = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in tokens]
    print(f"\n后验采样（T→Z，{samples} 条轨迹）：潜变量平均方差 {variance:.4f}")
    for index, variant in enumerate(variants, 1):
        print(f"  样本 {index}: {variant}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=1024)  # 潜空间/模型宽度。
    parser.add_argument("--n-heads", type=int, default=16)  # 注意力头数，需整除 dim。
    parser.add_argument("--encoder-layers", type=int, default=4)  # 编码器层数。
    parser.add_argument("--flow-layers", type=int, default=12)  # Flow 主干层数。
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)  # 每批样本数。
    parser.add_argument("--stage1-steps", type=int, default=200)  # 表征预训练步数。
    parser.add_argument("--stage2-steps", type=int, default=500)  # Flow 训练步数。
    parser.add_argument("--inference-steps", type=int, default=30)  # ODE 积分数。
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")  # ODE 求解器。
    parser.add_argument("--quantizer", choices=["none", "fsq", "vq", "rvq"], default="fsq")
    parser.add_argument("--levels", type=int, default=5)  # FSQ 每维级数（奇数）。
    parser.add_argument("--codebook-size", type=int, default=256)  # VQ/RVQ 码本大小。
    parser.add_argument("--num-quantizers", type=int, default=2)  # RVQ 级数。
    parser.add_argument("--commitment", type=float, default=0.25)  # VQ commitment 权重。
    parser.add_argument("--readout", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--temperature", type=float, default=0.8)  # 采样 readout 的温度。
    parser.add_argument("--decode", choices=["flow", "maskgit"], default="flow")
    parser.add_argument("--maskgit-steps", type=int, default=8)  # 并行细化的步数。
    parser.add_argument("--posterior-samples", type=int, default=4)  # T→Z 后验轨迹数。
    args = parser.parse_args()

    torch.manual_seed(42)  # 固定随机种子，便于复现实验。
    enable_flash_attention()  # CUDA 上走 FlashAttention-2 / memory-efficient SDPA。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()  # LLaDA2.1-mini 的 BPE tokenizer，vocab=157184。
    vocab_size = tokenizer.vocab_size
    mask_id = tokenizer.mask_token_id
    if mask_id is None:  # 兜底：无 mask token 时用最后一个位置。
        mask_id = vocab_size - 1
    data = torch.tensor(
        tokenizer.encode(CORPUS, add_special_tokens=False), dtype=torch.long
    )

    assert args.dim % args.n_heads == 0, f"dim ({args.dim}) 必须能被 n_heads ({args.n_heads}) 整除"
    model = FlowJEPA(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        flow_layers=args.flow_layers,
        quantizer=args.quantizer,
        levels=args.levels,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        commitment=args.commitment,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device: {device} | vocab: {vocab_size} | sequence: {SEQ_LEN}")
    print(
        f"参数: {total_params / 1e6:.1f}M 总 | {trainable_params / 1e6:.1f}M 可训练 | "
        f"dim={args.dim} heads={args.n_heads} enc={args.encoder_layers} flow={args.flow_layers}"
    )
    print(f"量化器: {model.quantizer.kind} | readout: {args.readout} | decode: {args.decode} | solver: {args.solver}")
    if device.type == "cpu" and total_params > 1e8:
        print("警告: 当前是 CPU，训练大模型会很慢；建议使用 CUDA（自动启用 bf16）。")

    train(model, data, mask_id, device, args)

    for prompt in ["人工智能", "机器学习"]:
        if args.decode == "maskgit":
            text = maskgit_decode(
                model, prompt, tokenizer, mask_id, device, args.maskgit_steps
            )
            token_ids = torch.tensor(
                [tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long, device=device
            )
        else:
            text, token_ids = generate(
                model,
                prompt,
                tokenizer,
                mask_id,
                device,
                args.inference_steps,
                args.solver,
                args.readout,
                args.temperature,
            )
        print(f"\n提示：{prompt}\n生成：{text}")
        if args.posterior_samples > 1:
            posterior_demo(
                model,
                token_ids,
                tokenizer,
                mask_id,
                device,
                args.posterior_samples,
                args.inference_steps,
                args.solver,
                args.readout,
                args.temperature,
            )


if __name__ == "__main__":  # 只有直接执行脚本时才训练，导入模块不会自动训练。
    main()
