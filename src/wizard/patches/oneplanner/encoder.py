import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp

from oneplanner.constants import (
    CLASS_TYPE_EGO,
    CLASS_TYPE_EGO_SHAPE,
    CLASS_TYPE_GOAL_POSE,
    CLASS_TYPE_LANE,
    CLASS_TYPE_LINE_STRING,
    CLASS_TYPE_NEIGHBOR,
    CLASS_TYPE_POLYGON,
    CLASS_TYPE_ROUTE,
    INPUT_T,
    LINE_STRING_TYPE_NUM,
    POLYGON_TYPE_NUM,
)
from oneplanner.models.planner.mixer import MixerBlock

# Width of the ORIGINAL DPlanner one-hot class vocabulary (ids 0..9, the
# constants above). Deliberately NOT ``constants.CLASS_TYPE_NUM`` (= 11): that
# includes the legacy "perception" slot (id 10) used only by ``E2EEncoder``,
# which pads 10 -> 11 at ``e2e_encoder._pad_pos``. Changing this value widens
# every pos tensor and ``pos_emb`` (Linear(4+10, D) here) and breaks every
# DPlanner checkpoint — see constants.py for the full story.
CLASS_TYPE_NUM = 10


def add_class_type(x, class_type):
    """
    Add class type to the input tensor.
    Args:
        x: Tensor of shape (B, T, D=4) where D=4 represents (x, y, cos, sin)
        class_type: Class type to add (int)
    Returns:
        x: Tensor with class type added at the end
    """
    B, T, D = x.shape
    assert D == 4, "Input tensor must have 4 features (x, y, cos, sin)"
    class_type_tensor = torch.zeros((B, T, CLASS_TYPE_NUM), device=x.device, dtype=x.dtype)
    class_type_tensor[..., class_type] = 1.0
    return torch.cat([x, class_type_tensor], dim=-1)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.use_ego_history = config.use_ego_history
        self.ego_history_dropout_rate = config.ego_history_dropout_rate
        self.use_turn_indicators = config.use_turn_indicators

        ego_num = 1
        goal_pose_num = 1
        ego_shape_num = 1
        turn_indicator_num = 1
        # agent_num (neighbors) removed from token count — neighbor branch no longer used
        self.token_num = (
            ego_num
            + config.lane_num
            + config.route_num
            + config.polygon_num
            + config.line_string_num
            + goal_pose_num
            + ego_shape_num
            + turn_indicator_num
        )

        self.ego_encoder = EgoEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
        )
        # neighbor_encoder removed — neighbor branch no longer used
        self.lane_encoder = LaneEncoder(
            config.lane_len,
            class_type=CLASS_TYPE_LANE,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
        )
        self.route_encoder = LaneEncoder(
            config.route_len,
            class_type=CLASS_TYPE_ROUTE,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
        )
        self.polygon_encoder = LineEncoder(
            config.polygon_len,
            class_type=CLASS_TYPE_POLYGON,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
            point_dim=2 + POLYGON_TYPE_NUM,
        )
        self.line_string_encoder = LineEncoder(
            config.line_string_len,
            class_type=CLASS_TYPE_LINE_STRING,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
            point_dim=2 + LINE_STRING_TYPE_NUM,
        )
        self.goal_pose_encoder = GoalPoseEncoder(
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.ego_shape_encoder = FloatsEncoder(
            num_float=3,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.turn_indicator_encoder = FloatsEncoder(
            num_float=INPUT_T,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_fusion_depth,
        )

        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(4 + CLASS_TYPE_NUM, config.hidden_dim)

        # positional embedding for route
        self.route_position_embedding = nn.Parameter(
            torch.randn(1, config.route_num, config.hidden_dim)
        )

        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.attribute_emb.weight, std=0.02)

    def forward(self, inputs):
        B = inputs["ego_agent_past"].shape[0]
        encodings, masks, positions = self._encode_segments(inputs)
        return self._fuse_segments(encodings, masks, positions, B)

    def _encode_segments(self, inputs):
        """Run every per-stream encoder; return ordered (encodings, masks, positions).

        The 8 segments are returned in the fixed planner order
        (ego, lanes, route, polygon, line_string, goal_pose, ego_shape,
        turn_indicator) — the single source of truth the fusion + bridge rely on.
        """
        # ego agent — clone to avoid in-place mutation of shared batch tensors
        # (critical for DPO / multi-sample inference which reuse the same batch)
        ego = inputs["ego_agent_past"].clone()  # (B, T=INPUT_T + 1, D=4)
        if not self.use_ego_history:
            ego = torch.zeros_like(ego)
        ego[:, 6:] *= 0.0  # Only keep the current + first 5 steps of ego history

        # neighbor_agents_past removed — neighbor branch no longer used

        # vector maps
        lanes = inputs["lanes"]  # (B, P=70, V=20, D=13)
        lanes_speed_limit = inputs["lanes_speed_limit"]  # (B, P=70, V=20, D=1)
        lanes_has_speed_limit = inputs["lanes_has_speed_limit"]  # (B, P=70, V=20, D=1)

        # route
        route = inputs["route_lanes"]  # (B, P=25, V=20, D=13)
        route_speed_limit = inputs["route_lanes_speed_limit"]  # (B, P=25, V=20, D=1)
        route_has_speed_limit = inputs["route_lanes_has_speed_limit"]  # (B, P=25, V=20, D=1)

        # polygons
        polygons = inputs["polygons"]  # (B, P=10, V=40, D=2)

        # line strings
        line_strings = inputs["line_strings"]  # (B, P=10, V=20, D=2)

        # goal pose
        goal_pose = inputs["goal_pose"]  # (B, D=4)

        # ego shape
        ego_shape = inputs["ego_shape"]  # (B, D=3)

        # turn indicator
        turn_indicator = inputs["turn_indicators"][:, :-1]  # (B, T)
        turn_indicator = turn_indicator.float()
        if not self.use_turn_indicators:
            turn_indicator = torch.zeros_like(turn_indicator)

        B = ego.shape[0]

        encoding_ego, ego_mask, ego_pos = self.ego_encoder(ego)

        if self.ego_history_dropout_rate > 0:
            encoding_ego = F.dropout(
                encoding_ego, p=self.ego_history_dropout_rate, training=self.training
            )

        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit
        )
        encoding_route, route_mask, route_pos = self.route_encoder(
            route, route_speed_limit, route_has_speed_limit
        )
        encoding_polygon, polygon_mask, polygon_pos = self.polygon_encoder(polygons)
        encoding_line_string, line_string_mask, line_string_pos = self.line_string_encoder(
            line_strings
        )

        # add positional embedding for route
        route_num = encoding_route.shape[1]
        route_position_emb = self.route_position_embedding[:, :route_num]  # (1, P, hidden_dim)
        route_position_emb = route_position_emb.expand(B, -1, -1)  # (B, P, hidden_dim)
        valid_route_mask = ~route_mask
        encoding_route = (
            encoding_route + route_position_emb * valid_route_mask.unsqueeze(-1).float()
        )

        encoding_goal_pose, goal_pose_mask, goal_pose_pos = self.goal_pose_encoder(goal_pose)
        encoding_ego_shape, ego_shape_mask, ego_shape_pos = self.ego_shape_encoder(ego_shape)
        encoding_turn_indicator, turn_indicator_mask, turn_indicator_pos = (
            self.turn_indicator_encoder(turn_indicator)
        )

        encodings = [
            encoding_ego,
            encoding_lanes,
            encoding_route,
            encoding_polygon,
            encoding_line_string,
            encoding_goal_pose,
            encoding_ego_shape,
            encoding_turn_indicator,
        ]
        masks = [
            ego_mask,
            lanes_mask,
            route_mask,
            polygon_mask,
            line_string_mask,
            goal_pose_mask,
            ego_shape_mask,
            turn_indicator_mask,
        ]
        positions = [
            ego_pos,
            lane_pos,
            route_pos,
            polygon_pos,
            line_string_pos,
            goal_pose_pos,
            ego_shape_pos,
            turn_indicator_pos,
        ]
        return encodings, masks, positions

    def _fuse_segments(self, encodings, masks, positions, B):
        """Concat per-stream tokens, add the masked positional embedding, fuse."""
        encoding_input = torch.cat(encodings, dim=1)
        encoding_mask = torch.cat(masks, dim=1).view(-1)
        encoding_pos = torch.cat(positions, dim=1).view(B * self.token_num, -1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros(
            (B * self.token_num, self.hidden_dim),
            device=encoding_pos.device,
            dtype=encoding_pos.dtype,
        )
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        return self.fusion(encoding_input, encoding_mask.view(B, self.token_num))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        mlp_ratio = 4.0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout
        )
        # SDPA fast path (perf.sdpa_fusion_encoder, flipped by
        # training/compile.py): need_weights=False skips materializing the
        # [B*heads, N, N] attention probs per layer. Default False ==
        # need_weights=True == the current default argument — bit-identical.
        self.sdpa_only: bool = False

    def forward(self, x, mask):
        x = x + self.drop_path(
            self.attn(self.norm1(x), x, x, key_padding_mask=mask, need_weights=not self.sdpa_only)[
                0
            ]
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EgoEncoder(nn.Module):
    def __init__(self, time_len, drop_path_rate, hidden_dim, depth):
        super().__init__()
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        self._hidden_dim = hidden_dim

        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, T=21, D=4 (x, y, cos, sin)
        """
        B, T, D = x.shape
        pos = x[:, -1].clone()  # (B, D=4[x, y, cos, sin])
        pos = pos.unsqueeze(1)  # (B, 1, D=4)
        pos = add_class_type(pos, CLASS_TYPE_EGO)

        mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1, keepdim=True)  # (B, 1, C=channels_mlp_dim)

        x = self.emb_project(self.norm(x))  # (B, hidden_dim)

        return x, mask, pos


class NeighborEncoder(nn.Module):
    def __init__(self, time_len, drop_path_rate, hidden_dim, depth):
        super().__init__()
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        self._hidden_dim = hidden_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8 + 1,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        """
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :4].clone()  # x, y, cos, sin
        pos = add_class_type(pos, CLASS_TYPE_NEIGHBOR)

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)
        x[..., 4:6] *= 0.0  # Zero out velocity features

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1)

        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device, dtype=x.dtype)
        x_result[valid_indices] = x.to(x_result.dtype)  # Fill in valid parts

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class LaneEncoder(nn.Module):
    def __init__(self, lane_len, class_type, drop_path_rate, hidden_dim, depth):
        super().__init__()
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        assert class_type in [CLASS_TYPE_LANE, CLASS_TYPE_ROUTE], (
            "Invalid class type for LaneEncoder"
        )

        self._lane_len = lane_len
        self._class_type = class_type

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.attribute_emb = nn.Linear(5 + 2 * 10, channels_mlp_dim)  # traffic_light and line type

        self.channel_pre_project = Mlp(
            in_features=8,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x, speed_limit, has_speed_limit):
        """
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y,
            x_right-x, y_right-y, traffic(5) + line_type(2 * 10))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        """
        attribute = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :4].clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos = add_class_type(pos, self._class_type)

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        valid_indices = ~mask_p.view(-1)

        x = x.view(B * P, V, -1)

        # Use torch.where instead of indexing to maintain fixed size
        x = torch.where(valid_indices.view(-1, 1, 1), x, torch.zeros_like(x))

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        attribute = attribute.view(B * P, -1)

        # Create embeddings for all positions
        speed_limit_emb = self.speed_limit_emb(speed_limit)
        unknown_speed_emb = self.unknown_speed_emb(
            torch.zeros(B * P, dtype=torch.long, device=x.device)
        )
        speed_limit_embedding = torch.where(has_speed_limit.bool(), speed_limit_emb, unknown_speed_emb)

        # Process traffic lights for all positions
        traffic_light_embedding = self.attribute_emb(attribute)

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        # Apply mask to zero out invalid positions
        x = x * valid_indices.float().unsqueeze(-1)

        return x.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class LineEncoder(nn.Module):
    def __init__(self, line_len, class_type, drop_path_rate, hidden_dim, depth, point_dim=2):
        super().__init__()
        self._class_type = class_type
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        self._line_len = line_len

        self.channel_pre_project = Mlp(
            in_features=point_dim + 2,  # point_dim (x, y, type_one_hot...) + dx + dy
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=line_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, V, D(x, y)
        """
        B, P, V, D = x.shape
        # diffを取る
        diff_x = x[:, :, 1:, 0] - x[:, :, :-1, 0]  # (B, P, V-1)
        diff_y = x[:, :, 1:, 1] - x[:, :, :-1, 1]  # (B, P, V-1)
        diff_x = torch.cat([diff_x, torch.zeros_like(diff_x[:, :, :1])], dim=2)  # (B, P, V)
        diff_x = diff_x.view(B, P, V, 1)
        diff_y = torch.cat([diff_y, torch.zeros_like(diff_y[:, :, :1])], dim=2)  # (B, P, V)
        diff_y = diff_y.view(B, P, V, 1)
        x = torch.concat([x, diff_x, diff_y], dim=-1)  # (B, P, V, D+2)

        pos = x[:, :, int(self._line_len / 2), :4].clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos = add_class_type(pos, self._class_type)

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        valid_indices = ~mask_p.view(-1)

        x = x.view(B * P, V, -1)

        # Use torch.where instead of indexing to maintain fixed size
        x = torch.where(valid_indices.view(-1, 1, 1), x, torch.zeros_like(x))

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        # Apply mask to zero out invalid positions
        x = x * valid_indices.float().unsqueeze(-1)

        return x.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class GoalPoseEncoder(nn.Module):
    def __init__(self, drop_path_rate, hidden_dim):
        super().__init__()
        channels_mlp_dim = 128

        self._hidden_dim = hidden_dim

        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, D=4 (x, y, cos, sin)
        """
        B, D = x.shape
        pos = x.clone()  # (B, D=4[x, y, cos, sin])
        pos = pos.unsqueeze(1)  # (B, 1, D=4)
        pos = add_class_type(pos, CLASS_TYPE_GOAL_POSE)

        mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)

        x = self.channel_pre_project(x)  # (B, C=channels_mlp_dim)
        x = x.unsqueeze(1)  # (B, 1, C=channels_mlp_dim)

        x = self.emb_project(self.norm(x))  # (B, 1, hidden_dim)

        return x, mask, pos


class FloatsEncoder(nn.Module):
    def __init__(self, num_float, drop_path_rate, hidden_dim):
        super().__init__()
        channels_mlp_dim = 128

        self._hidden_dim = hidden_dim

        self.channel_pre_project = Mlp(
            in_features=num_float,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, D
        """
        B, D = x.shape
        pos = torch.zeros((B, 4), device=x.device, dtype=x.dtype)  # (B, D=4[x, y, cos, sin])
        pos[:, 2] = 1.0  # cos(0) = 1
        pos = pos.unsqueeze(1)  # (B, 1, D=4)
        pos = add_class_type(pos, CLASS_TYPE_EGO_SHAPE)

        mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)

        x = self.channel_pre_project(x)  # (B, C=channels_mlp_dim)
        x = x.unsqueeze(1)  # (B, 1, C=channels_mlp_dim)

        x = self.emb_project(self.norm(x))  # (B, 1, hidden_dim)

        return x, mask, pos


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, drop_path_rate, depth):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)
