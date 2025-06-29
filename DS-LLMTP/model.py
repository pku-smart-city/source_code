import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from torch.nn.utils.parametrizations import weight_norm
import torch.cuda


# Chomp1d module to remove extra padding
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# TemporalBlock module for Temporal Convolutional Network
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation), dim=1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation), dim=1)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        output = self.relu(out + res)
        torch.cuda.empty_cache()  # Release cache after forward pass
        return output


# TemporalConvNet module composed of multiple TemporalBlocks
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.network(x)
        torch.cuda.empty_cache()  # Release cache after forward pass
        return output


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        # Convert time to a PyTorch tensor
        self.time = torch.tensor(time, dtype=torch.float32)
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        # Ensure the index tensor is of long type
        index_day = (day_emb[:, -1, :] * self.time.long()).long().contiguous()
        week_emb = x[..., 2]
        # Ensure the index tensor is of long type
        index_week = (week_emb[:, -1, :]).long().contiguous()

        index_day = torch.clamp(index_day, 0, self.time.long() - 1)
        index_week = torch.clamp(index_week, 0, 6)

        time_day = self.time_day[index_day]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)
        time_week = self.time_week[index_week]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        torch.cuda.empty_cache()  # Release cache after forward pass
        return tem_emb


class PFA(nn.Module):
    def __init__(self, device="cuda:2", deepseek_layers=10, U=1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
        self.deepseek = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-moe-16b-base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            output_hidden_states=True
        )
        total_layers = len(self.deepseek.layers)
        if deepseek_layers > total_layers:
            raise ValueError(f"deepseek_layers ({deepseek_layers}) exceeds the total number of layers ({total_layers}) in the model.")
        self.deepseek.layers = self.deepseek.layers[:deepseek_layers]
        self.prompt_proj = nn.Linear(self.deepseek.config.hidden_size, self.deepseek.config.hidden_size)
        self.U = U

        for layer_index, layer in enumerate(self.deepseek.layers):
            for name, param in layer.named_parameters():
                if layer_index < deepseek_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x, prompt_tokens=None):
        st_emb = x.to(torch.bfloat16)
        if prompt_tokens is not None:
            prompt_emb = self.deepseek.embed_tokens(prompt_tokens.input_ids)
            prompt_emb = self.prompt_proj(prompt_emb.mean(dim=1))
            st_emb = st_emb + prompt_emb.unsqueeze(1)

        outputs = self.deepseek(inputs_embeds=st_emb)
        return outputs.last_hidden_state

# PromptNetwork module to process input with TCN
class PromptNetwork(nn.Module):
    def __init__(self, input_dim, channels, num_nodes, input_len, output_len):
        super(PromptNetwork, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len

        # Add TCN
        num_channels = [channels] * 3  # Number of TCN layers and channels
        self.tcn = TemporalConvNet(input_dim * input_len, num_channels, kernel_size=2, dropout=0.2)

        self.prompt_conv = nn.Conv2d(
            channels, input_dim * input_len, kernel_size=(1, 1)
        ).to(torch.bfloat16)

    def forward(self, x):
        x = x.to(torch.bfloat16)
        x = x.transpose(1, 3)
        batch_size, _, num_nodes, _ = x.shape
        # Modify to reshape
        x = x.reshape(batch_size, -1, num_nodes)

        # Process time series data through TCN
        x = self.tcn(x)

        x = x.unsqueeze(-1)
        x = self.prompt_conv(x)
        torch.cuda.empty_cache()  # Release cache after forward pass
        return x


class ST_LLM(nn.Module):
    def __init__(
            self,
            device,
            input_dim=3,
            channels=64,
            num_nodes=420,
            input_len=12,
            output_len=12,
            U=1,
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.U = U

        # Automatically determine time parameter
        if num_nodes in [250, 266]:
            time = 48

        # Dimension configuration
        self.deep_channel = 1024
        self.to_deep_channel = 2048

        # Input processing layer
        self.start_conv = nn.Conv2d(
            in_channels=input_dim * input_len,
            out_channels=self.deep_channel,
            kernel_size=(1, 1)
        ).to(dtype=torch.bfloat16)

        # Embedding layer group
        self.Temb = TemporalEmbedding(time, self.deep_channel)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, self.deep_channel))
        self.prompt_embed = nn.Embedding(num_nodes, self.deep_channel)

        # Initialize parameters
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.prompt_embed.weight)

        # Feature fusion layer
        self.feature_fusion = nn.Conv2d(
            in_channels=self.deep_channel * 4,
            out_channels=self.to_deep_channel,
            kernel_size=(1, 1)
        ).to(torch.bfloat16)

        # Core module
        self.deepseek = PFA(device=device, deepseek_layers=10, U=U)

        # Output layer
        self.regression_layer = nn.Conv2d(
            in_channels=self.to_deep_channel,
            out_channels=output_len,
            kernel_size=(1, 1)
        ).to(torch.bfloat16)

        # Initialize prompt network
        self.prompt_network = PromptNetwork(input_dim, channels, num_nodes, input_len, output_len)

    def forward(self, history_data, prompt_tokens=None):
        # Process input through prompt network
        prompted_history_data = self.prompt_network(history_data)

        # Input dimension processing
        batch_size, _, num_nodes, seq_len = prompted_history_data.shape

        # 1. Spatiotemporal feature expansion
        input_flat = prompted_history_data.permute(0, 3, 2, 1).contiguous()  # [B, T, N, C]
        input_flat = input_flat.view(batch_size, num_nodes, -1).contiguous()  # [B, N, T*C]
        input_flat = input_flat.permute(0, 2, 1).unsqueeze(-1).contiguous()  # [B, T*C, N, 1]
        conv_out = self.start_conv(input_flat).contiguous()  # [B, 1024, N, 1]

        # 2. Multimodal feature fusion
        time_emb = self.Temb(prompted_history_data.permute(0, 3, 2, 1)).contiguous()
        node_emb = self.node_emb[None, :, :].expand(batch_size, -1, -1)
        node_emb = node_emb.permute(0, 2, 1).unsqueeze(-1).contiguous()
        prompt_emb = self.prompt_embed(
            torch.arange(num_nodes, device=self.device)[None, :].expand(batch_size, -1)
        )
        prompt_emb = prompt_emb.permute(0, 2, 1).unsqueeze(-1).contiguous()

        # 3. Feature concatenation and fusion
        fused = torch.cat([conv_out, time_emb, node_emb, prompt_emb], dim=1).contiguous()
        fused = self.feature_fusion(fused).contiguous()  # [B, 2048, N, 1]

        # 4. Spatiotemporal modeling
        st_feat = fused.permute(0, 2, 1, 3).squeeze(-1).contiguous()  # [B, N, 2048]
        st_feat = st_feat.contiguous()

        # 5. DeepSeek processing
        st_out = self.deepseek(st_feat, prompt_tokens).contiguous()  # [B, N, 2048]

        # 6. Output prediction
        st_out = st_out.permute(0, 2, 1).unsqueeze(-1).contiguous()  # [B, 2048, N, 1]
        prediction = self.regression_layer(st_out)  # [B, 12, N, 1]
        prediction = prediction.squeeze(-1).permute(0, 2, 1)  # [B, N, 12]

        # Ensure output dimension matches the real value dimension
        assert prediction.shape == (prompted_history_data.shape[0], self.num_nodes,
                                    self.output_len), f"Output shape {prediction.shape} does not match expected shape {(prompted_history_data.shape[0], self.num_nodes, self.output_len)}"
        torch.cuda.empty_cache()  # Release cache after forward pass
        return prediction