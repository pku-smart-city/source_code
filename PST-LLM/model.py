import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
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
        result = self.relu(out + res)
        del out, res
        torch.cuda.empty_cache()
        return result


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.network(x)
        torch.cuda.empty_cache()
        return output


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super().__init__()
        self.time = torch.tensor(time, dtype=torch.float32)
        self.time_day = nn.Parameter(torch.empty(time, features))
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_day)
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        index_day = (day_emb[:, -1, :] * self.time).long().clamp(0, self.time.long() - 1)
        week_emb = x[..., 2]
        index_week = week_emb[:, -1, :].long().clamp(0, 6)

        time_day = self.time_day[index_day].transpose(1, 2).unsqueeze(-1)
        time_week = self.time_week[index_week].transpose(1, 2).unsqueeze(-1)

        result = time_day + time_week
        del day_emb, week_emb, index_day, index_week, time_day, time_week
        torch.cuda.empty_cache()
        return result


class PromptNetwork(nn.Module):
    def __init__(self, input_dim, channels, num_nodes, input_len, output_len):
        super().__init__()
        num_channels = [channels] * 3
        self.tcn = TemporalConvNet(input_dim * input_len, num_channels, kernel_size=2)
        self.prompt_conv = nn.Conv2d(channels, input_dim * input_len, kernel_size=(1, 1))

    def forward(self, x):
        x = x.transpose(1, 3)
        batch_size, _, num_nodes, _ = x.shape
        x = x.reshape(batch_size, -1, num_nodes)
        x = self.tcn(x)
        output = self.prompt_conv(x.unsqueeze(-1))
        del x
        torch.cuda.empty_cache()
        return output


class PFA(nn.Module):
    def __init__(self, device, prune_ratio=0.5, freeze_ratio=0.7, keep_indices=None):
        super().__init__()
        self.device = device
        self.prune_ratio = prune_ratio
        self.freeze_ratio = freeze_ratio
        self.keep_indices = keep_indices  # 存储保留层的索引

        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
        self.deepseek = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-moe-16b-base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True
        ).to(device)

        # 延迟剪枝到首次前向传播
        self.pruned = False
        self.prompt_proj = nn.Linear(self.deepseek.config.hidden_size, self.deepseek.config.hidden_size)

        # 如果有保留层索引，立即进行剪枝
        if self.keep_indices is not None:
            self.prune_with_indices(self.keep_indices)

    def forward(self, x, prompt_tokens=None):
        # 首次前向传播时执行剪枝（如果没有指定保留层索引）
        if not self.pruned and self.keep_indices is None:
            with torch.no_grad():
                calib_data = x[:max(1, len(x) // 4)].detach().clone()
                self.prune_layers(calib_data=calib_data)
                del calib_data
                torch.cuda.empty_cache()
            self.pruned = True

        st_emb = x.to(torch.bfloat16)

        if prompt_tokens is not None:
            prompt_emb = self.deepseek.embed_tokens(prompt_tokens.input_ids)
            prompt_emb = self.prompt_proj(prompt_emb.mean(dim=1))
            st_emb = st_emb + prompt_emb.unsqueeze(1)

        outputs = self.deepseek(inputs_embeds=st_emb)
        result = outputs.last_hidden_state
        del st_emb, outputs
        if prompt_tokens is not None:
            del prompt_emb
        torch.cuda.empty_cache()
        return result

    def compute_layer_similarity(self, calib_data):
        """计算每层输入输出的余弦相似度"""
        similarities = [0.0] * len(self.deepseek.layers)

        def hook_fn(layer_idx):
            def hook(module, input, output):
                if len(input) > 0 and input[0] is not None:
                    flat_input = input[0].flatten(end_dim=-2)
                    if isinstance(output, tuple):
                        flat_output = output[0].flatten(end_dim=-2)
                    else:
                        flat_output = output.flatten(end_dim=-2)

                    cos_sim = F.cosine_similarity(flat_input, flat_output, dim=-1)
                    valid_mask = ~torch.isnan(cos_sim) & ~torch.isinf(cos_sim)
                    if valid_mask.any():
                        similarities[layer_idx] = cos_sim[valid_mask].mean().item()
                    del flat_input, flat_output, cos_sim, valid_mask
                    torch.cuda.empty_cache()

            return hook

        hooks = []
        for idx, layer in enumerate(self.deepseek.layers):
            hooks.append(layer.register_forward_hook(hook_fn(idx)))

        with torch.no_grad():
            self.deepseek(inputs_embeds=calib_data.to(self.device))

        for hook in hooks:
            hook.remove()

        del hooks
        torch.cuda.empty_cache()

        return [1 - sim for sim in similarities]

    def prune_layers(self, calib_data):
        """基于重要性分数剪枝冗余层"""
        print("Computing layer importance...")
        with torch.no_grad():
            importance_scores = self.compute_layer_similarity(calib_data)

        print("Layer importance scores:", importance_scores)

        sorted_indices = np.argsort(importance_scores)[::-1]
        total_layers = len(self.deepseek.layers)
        keep_num = max(1, int(total_layers * (1 - self.prune_ratio)))
        keep_indices = sorted(sorted_indices[:keep_num])
        self.keep_indices = keep_indices  # 保存保留层索引

        print(f"Pruning: keeping {keep_num}/{total_layers} layers")
        self.prune_with_indices(keep_indices)

        del importance_scores, sorted_indices
        torch.cuda.empty_cache()

    def prune_with_indices(self, keep_indices):
        """根据给定的索引直接剪枝"""
        new_layers = nn.ModuleList()
        for i in keep_indices:
            new_layers.append(self.deepseek.layers[i])
        self.deepseek.layers = new_layers
        self.freeze_parameters()
        self.pruned = True
        print(f"Pruned with given indices. Kept layers: {keep_indices}")

    def freeze_parameters(self):
        """分层冻结参数"""
        total_layers = len(self.deepseek.layers)
        freeze_num = int(total_layers * self.freeze_ratio)

        print(f"Freezing first {freeze_num}/{total_layers} layers")

        for layer_idx, layer in enumerate(self.deepseek.layers):
            for name, param in layer.named_parameters():
                if layer_idx < freeze_num:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        torch.cuda.empty_cache()


class ST_LLM(nn.Module):
    def __init__(self, device, input_dim=3, channels=64, num_nodes=250,
                 input_len=12, output_len=12, prune_ratio=0.5, freeze_ratio=0.7,
                 keep_indices=None):  # keep_indices参数
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len

        if num_nodes in [250, 266]:
            time = 48

        self.deep_channel = 1024
        self.to_deep_channel = 2048

        self.start_conv = nn.Conv2d(
            input_dim * input_len, self.deep_channel, kernel_size=(1, 1)
        ).to(torch.bfloat16)

        self.Temb = TemporalEmbedding(time, self.deep_channel)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, self.deep_channel))
        self.prompt_embed = nn.Embedding(num_nodes, self.deep_channel)
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.prompt_embed.weight)

        self.feature_fusion = nn.Conv2d(
            self.deep_channel * 4, self.to_deep_channel, kernel_size=(1, 1)
        ).to(torch.bfloat16)

        self.deepseek = PFA(device, prune_ratio, freeze_ratio, keep_indices)

        self.regression_layer = nn.Conv2d(
            self.to_deep_channel, output_len, kernel_size=(1, 1)
        ).to(torch.bfloat16)

        self.prompt_network = PromptNetwork(input_dim, channels, num_nodes, input_len, output_len)

        self.project_to_deepseek = nn.Linear(self.to_deep_channel, self.deepseek.deepseek.config.hidden_size)

    def forward(self, history_data, prompt_tokens=None):
        prompted_history_data = self.prompt_network(history_data)
        batch_size, _, num_nodes, seq_len = prompted_history_data.shape

        input_flat = prompted_history_data.permute(0, 3, 2, 1).contiguous()
        input_flat = input_flat.view(batch_size, num_nodes, -1).contiguous()
        input_flat = input_flat.permute(0, 2, 1).unsqueeze(-1).contiguous()
        conv_out = self.start_conv(input_flat).contiguous()

        del input_flat
        torch.cuda.empty_cache()

        time_emb = self.Temb(prompted_history_data.permute(0, 3, 2, 1)).contiguous()
        node_emb = self.node_emb[None].expand(batch_size, -1, -1).permute(0, 2, 1).unsqueeze(-1).contiguous()
        prompt_emb = self.prompt_embed(
            torch.arange(num_nodes, device=self.device)[None].expand(batch_size, -1)
        ).permute(0, 2, 1).unsqueeze(-1).contiguous()

        fused = torch.cat([conv_out, time_emb, node_emb, prompt_emb], dim=1).contiguous()
        fused = self.feature_fusion(fused).contiguous()

        del conv_out, time_emb, node_emb, prompt_emb
        torch.cuda.empty_cache()

        st_feat = fused.permute(0, 2, 1, 3).squeeze(-1).contiguous()

        del fused
        torch.cuda.empty_cache()

        st_feat = self.project_to_deepseek(st_feat)

        st_out = self.deepseek(st_feat, prompt_tokens).contiguous()

        del st_feat
        torch.cuda.empty_cache()

        st_out = st_out.permute(0, 2, 1).unsqueeze(-1).contiguous()
        prediction = self.regression_layer(st_out).squeeze(-1).permute(0, 2, 1)

        del st_out
        torch.cuda.empty_cache()

        assert prediction.shape == (batch_size, num_nodes, self.output_len)
        return prediction

    def get_pruning_indices(self):
        return self.deepseek.keep_indices
