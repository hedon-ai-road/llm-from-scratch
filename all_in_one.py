import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ================================
# 模型配置常量
# ================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小
    "context_length": 256,  # 上下文长度
    "emb_dim": 768,         # 嵌入维度
    "n_heads": 12,          # 注意力头数量
    "n_layers": 12,         # 层数
    "drop_rate": 0.1,       # dropout率
    "qkv_bias": False       # 查询-键-值偏置
}

# ================================
# 基础组件类
# ================================
class LayerNorm(nn.Module):
    """层归一化实现"""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU激活函数"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


# ================================
# 注意力机制类
# ================================
class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""
    def __init__(self, d_in: int, d_out: int,
                context_length: int, dropout: float, num_heads: int, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out必须能被num_heads整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # Q、K、V的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # 因果掩码：防止看到未来信息
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 计算Q、K、V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑为多头格式
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整维度顺序以便并行计算
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)
        
        # 应用因果掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积注意力
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


# ================================
# Transformer架构类
# ================================
class TransformerBlock(nn.Module):
    """Transformer块：多头注意力 + 前馈网络 + 残差连接"""
    def __init__(self, cfg) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 第一个子层：多头注意力 + 残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 第二个子层：前馈网络 + 残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    """完整的GPT模型实现"""
    def __init__(self, cfg) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠多个Transformer块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])],
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 词嵌入 + 位置嵌入
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ================================
# 数据处理相关
# ================================
class GPTDataset(Dataset):
    """GPT训练数据集"""
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # 使用滑动窗口创建训练样本
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(tokenizer, txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """创建数据加载器"""
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def prepare_train_and_val_data(file_path, tokenizer) -> tuple[DataLoader, DataLoader]:
    """准备训练和验证数据加载器"""
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # 90%训练，10%验证
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader(
        tokenizer,
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        tokenizer,
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


# ================================
# 训练相关函数
# ================================
def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个批次的损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(),
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算数据加载器的平均损失"""
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader,
                    optimizer, device, num_epochs,
                    eval_freq, eval_iter, start_context, tokenizer):
    """训练模型的主循环"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}):"
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}")

        # 每3个epoch生成样本
        if epoch % 3 == 0:
            generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


# ================================
# 文本生成相关
# ================================
def text_to_token_ids(text, tokenizer):
    """将文本转换为token ID"""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer) -> str:
    """将token ID转换回文本"""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_length):
    """使用模型生成文本"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成文本样本"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_length=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def generate(model, idx, max_new_tokens, context_length,
        temperature=0.0, top_k=None, eos_id=None):
    """带温度缩放、top_k 筛选的文本生成策略"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 使用 top_k 采样筛选 logits
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits,
            )
        
        if temperature > 0.0:
            # 使用温度缩放
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 当不使用温度缩放时，执行贪心解码，选取下一个词元
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # 如果遇到序列结束词元，则提前停止生成
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ================================
# 可视化函数
# ================================
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存
    print("损失曲线已保存为 training_losses.png")


if __name__ == "__main__":
    # 选择设备，在 GPT 和 Apple 芯片下可加速训练
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = tiktoken.get_encoding("gpt2")

    # 准备数据
    train_loader, val_loader = prepare_train_and_val_data("the-verdict.txt", tokenizer)

    # 初始化模型和优化器
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )

    # 训练模型
    num_epochs = 20
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer,
    )

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # 生成文本
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("I seemed to see", tokenizer).to(device),
        max_new_tokens=15,
        context_length=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))