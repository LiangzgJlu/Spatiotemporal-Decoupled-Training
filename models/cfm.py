import yaml
import torch
import numpy as np
import torch.nn as nn
from loguru import logger

class TimeRnnCell(nn.Module):
    def __init__(self, input_size, hidden_size, time_length):
        super(TimeRnnCell, self).__init__()
        
        self.input_size = input_size + 1
        self.hidden_size = hidden_size 
        self.time_length = time_length
        
        # 输出门参数
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # 候选记忆参数
        self.W_ig = nn.Parameter(torch.Tensor(torch.Tensor(input_size, hidden_size)))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        
        for param in self.parameters():
            if param.dim() == 2:
                # 权重使用Xavier初始化
                nn.init.xavier_normal_(param)
            else:
                # 偏置初始化为0
                nn.init.zeros_(param)
        
    def forward(self, x, c, t):
        
        """
        单层前向传播
        参数:
        - x: 输入数据, 形状为[batch_size, input_size]
        - c: 初始细胞状态,
        
        返回:
        - c_t: 当前时间步的细胞状态
        """
        
        time_col = torch.full((x.size(0), 1), t / self.time_length).to(x.device)
        x_t = torch.cat([x, time_col], dim=1)
        
        i_t = torch.sigmoid(torch.mm(x_t, self.W_ii) + self.b_i)
        f_t = torch.sigmoid(torch.mm(x_t, self.W_if) + self.b_f)
        g_t = torch.sigmoid(torch.mm(x_t, self.W_ig) + self.b_g)
        
        c_t = f_t * c + i_t * g_t
        return c_t

class TimeRnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, time_length):
        super(TimeRnn, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_length = time_length
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(TimeRnnCell(layer_input_size, hidden_size, self.time_length))
            
    def forward(self, x, c=None):
        
        """
        前向传播
        参数:
        - x: 输入序列, 形状为[seq_len, batch_size, input_size]
        - c: 初始系统状态, 形状为(num_layers, batch_size, hidden_size)
        
        返回:
        - c_n: 返回最后一个时间步的细胞状态
        """
        
        seq_len, batch_size, _ = x.size()
        device = x.device
        
        if c is None:
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            
        else:
            c_0 = c
        
        c_n = []
        
        c_t_layer = None
        for t in range(seq_len):
            x_t = x[t]
            
            for layer_idx in range(self.num_layers):
                c_p = c_0[layer_idx]
                
                if layer_idx == 0:
                    input_t = x_t
                else:
                    input_t = c_t_layer

                c_t_layer = self.layers[layer_idx](input_t, c_p, t)
                c_0[layer_idx] = c_t_layer
        
        c_n = c_0
        
        return c_n

class TimeRnnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.feature = config['feature']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.time_length = config['time_length']
        self.time_rnn = TimeRnn(self.feature, self.hidden_size, self.num_layers, self.time_length)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(1,0,2)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output = self.time_rnn(x, c_0)
        out = self.fc(output[:, -1, :].squeeze(1))
        return out
    
class MlpModel(nn.Module):
    def __init__(self, net_config):
        super(MlpModel, self).__init__()
        pass
    def forward(self, x):
        pass

class GruModel(nn.Module):
    def __init__(self, net_config):
        super(GruModel, self).__init__()
        self.hidden_size = net_config['hidden_size']
        self.feature = net_config['feature']
        self.num_layers = net_config['num_layers']
        self.gru = nn.GRU(self.feature, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output, _ = self.gru(x, h_0)
        out = output[:, -1, :].squeeze(1)
        out = self.fc(out)
        return out
        
        
class Seq2SeqLstmModel(nn.Module):
    def __init__(self, net_config):
        super(Seq2SeqLstmModel, self).__init__()
        
    def forward(self, x):
        pass
    
class Seq2SeqGruModel(nn.Module):
    def __init__(self, net_config):
        super(Seq2SeqGruModel, self).__init__()
        
    def forward(self, x):
        pass

class LstmModel(nn.Module):
    def __init__(self, net_config):
        super(LstmModel, self).__init__()
        
        self.hidden_size = net_config['hidden_size']
        self.feature = net_config['feature']
        self.num_layers = net_config['num_layers']
        
        self.lstm = nn.LSTM(self.feature, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output, (h, c) = self.lstm(x, (h_0, c_0))
        out = output[:, -1, :].squeeze(1)
        out = self.fc(out)
        return out


# --- 模型构建 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # 修正1: 生成正确维度的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # 维度变为 [max_len, 1, d_model]

    def forward(self, x):
        # 修正2: 使用正确的切片维度
        # x 形状: (seq_len, batch_size, d_model)
        # pe 形状: (max_len, 1, d_model)
        return x + self.pe[:x.size(0)]  # 自动广播到

class TransformerAccelerationPredictor(nn.Module):
    def __init__(self, net_config):
        super().__init__()
        
        self.input_dim = net_config['input_dim']
        self.d_model = net_config['d_model']
        self.nhead = net_config['nhead']
        self.num_layers = net_config['num_layers']
        
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.Linear(self.d_model, 1)
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.embedding(src)  # (batch_size, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)
        output = output[-1, :, :]  # 取最后一个时间步
        return self.decoder(output).squeeze()

class SparseCausalSelfAttention(nn.Module):
    """
    实现稀疏因果自注意力（Sparse Causal Attention），减少计算复杂度
    """
    def __init__(self, embed_dim, num_heads, sparsity_factor=4):
        super().__init__()
        self.num_heads = num_heads
        self.sparsity_factor = sparsity_factor
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # 生成稀疏因果掩码（Sparse Causal Mask）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)  # 因果掩码
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                if (j - i) % self.sparsity_factor != 0:  # 仅保留指数间隔的注意力
                    mask[i, j] = 1  # 1 表示屏蔽
        
        # 计算注意力
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_output

class SparseCausalTransformerEncoder(nn.Module):
    def __init__(self, net_config):
        super().__init__()
        
        self.input_dim = net_config['input_dim']
        self.d_model = net_config['d_model']
        self.num_heads = net_config['nhead']
        self.num_layers = net_config['num_layers']
        self.sparsity_factor = net_config['sparsity_factor']
        
        self.embedding = nn.Linear(self.input_dim, self.d_model)  # 线性投影到 Embedding 维度
        self.positional_encoding = self._generate_positional_encoding(self.d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, dim_feedforward=32, batch_first=True)
            for _ in range(self.num_layers)
        ])
        
        self.sparse_attention = SparseCausalSelfAttention(self.d_model, self.num_heads, self.sparsity_factor)

        self.fc = nn.Linear(self.d_model, 1)  # 预测输出

    def _generate_positional_encoding(self, d_model):
        """ 生成固定正弦位置编码 """
        position = torch.arange(500).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(500, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.embedding(x)  # 线性变换到 embedding 维度
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)  # 添加位置编码
        
        x = self.sparse_attention(x)  # 稀疏注意力层

        for layer in self.layers:
            x = layer(x)  # 通过 Transformer 编码器层
        
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)  # 通过全连接层
        return x



def create_model(opt):
    
    # real model config
    model_cfg = dict()
    with open(opt.cfg, 'r') as file:
        model_cfg: dict = yaml.load(file, Loader=yaml.FullLoader)
    
    model_type = model_cfg['type']
    net_config = model_cfg['net_config']
    model = None
    
    if model_type == 'lstm':
        model = LstmModel(net_config=net_config)
    elif model_type == "mlp":
        model = MlpModel(net_config=net_config)
    elif model_type == 'gru':
        model = GruModel(net_config=net_config)
    elif model_type == 'transformer':
        model = TransformerAccelerationPredictor(net_config=net_config)
    elif model_type == 'seq2seq-lstm':
        model = Seq2SeqLstmModel(net_config=net_config)
    elif model_type == 'seq2seq-gru':
        model = Seq2SeqGruModel(net_config=net_config)
    elif model_type == "time_rnn":
        model = TimeRnn(net_config)
    elif model_type == 'sparse_causal_transformer':
        model = SparseCausalTransformerEncoder(net_config)
    else:
        raise NotImplementedError(f"model type {model_type} is unknown! please code your model in file: models/cfm.py")
    return model
    