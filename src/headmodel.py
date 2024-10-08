import torch
import torch.nn as nn
from backboneTorch import TimeSeriesTransformer

class Instance_Superivsion_Head(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pred_hidden_dim=0, nlayers=2, proj_bn=False, pred_bn=False, norm_before_pred=False):
        super().__init__()
        # Projector
        proj_layers = []
        for i in range(nlayers):
            dim1 = in_dim if i == 0 else hidden_dim
            dim2 = out_dim if i == nlayers - 1 else hidden_dim
            proj_layers.append(nn.Linear(dim1, dim2))
            if i < nlayers - 1:
                if proj_bn:
                    proj_layers.append(nn.BatchNorm1d(dim2))
                proj_layers.append(nn.ReLU(inplace=True))
        self.projector = nn.Sequential(*proj_layers)

        # Predictor
        if pred_hidden_dim > 0:
            pred_layers = []
            dim1 = out_dim
            dim2 = pred_hidden_dim
            pred_layers.append(nn.Linear(dim1, dim2))
            if pred_bn:
                pred_layers.append(nn.BatchNorm1d(dim2))
            pred_layers.append(nn.ReLU(inplace=True))
            pred_layers.append(nn.Linear(dim2, out_dim))
            self.predictor = nn.Sequential(*pred_layers)
        else:
            self.predictor = None

        self.norm_before_pred = norm_before_pred

    def forward(self, x,return_target=False):
        feat = self.projector(x)
        # return projection; only for teacher
        if return_target:
            feat = nn.functional.normalize(feat, dim=-1, p=2)
            return feat # Return the normalized features
        if self.predictor is not None:
            if self.norm_before_pred:
                feat = nn.functional.normalize(feat, dim=-1, p=2)
            feat = self.predictor(feat)
        return feat

# Local group supervision head
class Local_Group_Superivsion_Head(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pred_hidden_dim=0, nlayers=2, proj_bn=False, pred_bn=False, norm_before_pred=False):
        super().__init__()
        # Same structure as the instance supervision head
        proj_layers = []
        for i in range(nlayers):
            dim1 = in_dim if i == 0 else hidden_dim
            dim2 = out_dim if i == nlayers - 1 else hidden_dim
            proj_layers.append(nn.Linear(dim1, dim2))
            if i < nlayers - 1:
                if proj_bn:
                    proj_layers.append(nn.BatchNorm1d(dim2))
                proj_layers.append(nn.ReLU(inplace=True))
        self.projector = nn.Sequential(*proj_layers)

        if pred_hidden_dim > 0:
            pred_layers = []
            dim1 = out_dim
            dim2 = pred_hidden_dim
            pred_layers.append(nn.Linear(dim1, dim2))
            if pred_bn:
                pred_layers.append(nn.BatchNorm1d(dim2))
            pred_layers.append(nn.ReLU(inplace=True))
            pred_layers.append(nn.Linear(dim2, out_dim))
            self.predictor = nn.Sequential(*pred_layers)
        else:
            self.predictor = None

        self.norm_before_pred = norm_before_pred

    def forward(self, x, return_target=False):
        feat = self.projector(x)
        if return_target:
            feat = nn.functional.normalize(feat, dim=-1, p=2)
            return feat
        if self.predictor is not None:
            if self.norm_before_pred:
                feat = nn.functional.normalize(feat, dim=-1, p=2)
            feat = self.predictor(feat)
        return feat

# Group supervision head
class Group_Superivsion_Head(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3, use_bn=False, norm_last_layer=True):
        super().__init__()
        # Projector
        proj_layers = []
        for i in range(nlayers):
            dim1 = in_dim if i == 0 else hidden_dim
            dim2 = hidden_dim
            proj_layers.append(nn.Linear(dim1, hidden_dim))
            if use_bn:
                proj_layers.append(nn.BatchNorm1d(hidden_dim))
            proj_layers.append(nn.ReLU(inplace=True))
        proj_layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.projector = nn.Sequential(*proj_layers)
        # last layer
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.projector(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# Memory module
class vit_mem(nn.Module):
    def __init__(self, dim, K, top_n):
        super().__init__()
        self.K = K
        self.top_n = top_n
        self.dim = dim

        # Create separate queues for keys and values
        self.register_buffer('queue_k', torch.randn(K, dim))
        self.queue_k = nn.functional.normalize(self.queue_k, dim=1)
        self.register_buffer('queue_v', torch.randn(K, dim))
        self.queue_v = nn.functional.normalize(self.queue_v, dim=1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # keys: (B, D)
        # values: (B, D)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.K:
            self.queue_k[ptr:ptr + batch_size, :] = keys
            self.queue_v[ptr:ptr + batch_size, :] = keys
            ptr = (ptr + batch_size) % self.K
        else:
            remaining = self.K - ptr
            self.queue_k[ptr:, :] = keys[:remaining, :]
            self.queue_v[ptr:, :] = keys[:remaining, :]
            self.queue_k[:batch_size - remaining, :] = keys[remaining:, :]
            self.queue_v[:batch_size - remaining, :] = keys[remaining:, :]
            ptr = batch_size - remaining
        self.queue_ptr[0] = ptr

    def forward(self, query):
        """
        forward to find the top-n neighbors (key-value pair) in memory
        """
        query = query.float()
        query = nn.functional.normalize(query, dim=1)
        queue_k = self.queue_k.clone().detach()
        similarity = torch.einsum('nd,kd->nk', query, queue_k)
        topk_indices = similarity.topk(self.top_n, dim=1)[1]  # (B, top_n)
        # top-N keys and values
        get_k = queue_k[topk_indices]  # (B, top_n, D)
        get_v = self.queue_v[topk_indices]  # (B, top_n, D)
        return get_k, get_v

class Mugs_Wrapper(nn.Module):
    def __init__(self, backbone, instance_head, local_group_head, group_head):
        super().__init__()
        self.backbone = backbone
        self.instance_head = instance_head
        self.local_group_head = local_group_head
        self.group_head = group_head

    def forward(self, input, return_target=False, local_group_memory_inputs=None):
        # input: list of tensors (different crops)
        class_tokens = []
        memory_tokens = []
        mean_patch_tokens = []

        for x in input:
            token_feat, memory_class_token_feat = self.backbone(
                x,
                return_all=True,
                local_group_memory_inputs=local_group_memory_inputs,
            )
            # token_feat: (B, N+1, D)
            # memory_class_token_feat: (B, D)
            class_token_feat = token_feat[:, 0]  # (B, D)
            class_tokens.append(class_token_feat)
            memory_tokens.append(memory_class_token_feat)
            mean_patch_feat = token_feat[:, 1:].mean(dim=1)
            mean_patch_tokens.append(mean_patch_feat)

        # Cat class tokens
        class_tokens = torch.cat(class_tokens, dim=0)  # (B * num_crops, D)
        memory_tokens = torch.cat(memory_tokens, dim=0)  # (B * num_crops, D)
        mean_patch_tokens = torch.cat(mean_patch_tokens, dim=0) if mean_patch_tokens else torch.empty(0).to(x[0].device)

        # Instance supervision head
        instance_output = self.instance_head(class_tokens, return_target=return_target)

        # Local group supervision head
        local_group_output = self.local_group_head(memory_tokens, return_target=return_target)

        # Group supervision head
        group_output = self.group_head(class_tokens)

        return instance_output, local_group_output, group_output, mean_patch_tokens.detach()


def get_model(args):
    # Instantiate
    backbone = TimeSeriesTransformer(
        input_dim=1,  
        embed_dim=512,  
        depth=4,        
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        dropout=0.1,
        attn_dropout=0.,
        num_relation_blocks=1,
        max_len=5000,  
        num_classes=0  
    )

    # memory components
    student_mem = vit_mem(
        dim=512,  # match the output dimension of the backbone
        K=args.local_group_queue_size,
        top_n=args.local_group_knn_top_n
    )
    teacher_mem = vit_mem(
        dim=512, # match the output dimension of the backbone
        K=args.local_group_queue_size,
        top_n=args.local_group_knn_top_n
    )

    # Define the heads
    student_instance_head = Instance_Superivsion_Head(
        in_dim=256,
        hidden_dim=256,
        out_dim=args.instance_out_dim,
        pred_hidden_dim=256,
        nlayers=3,
        proj_bn=args.use_bn_in_head,
        pred_bn=False,
        norm_before_pred=args.norm_before_pred,
    )

    teacher_instance_head = Instance_Superivsion_Head(
        in_dim=256,
        hidden_dim=256,
        out_dim=args.instance_out_dim,
        pred_hidden_dim=0,  # Teacher does not have predictor
        nlayers=3,
        proj_bn=args.use_bn_in_head,
        pred_bn=False,
        norm_before_pred=args.norm_before_pred,
    )

    # Local group heads
    student_local_group_head = Local_Group_Superivsion_Head(
        in_dim=256,
        hidden_dim=256,
        out_dim=args.local_group_out_dim,
        pred_hidden_dim=256,
        nlayers=3,
        proj_bn=args.use_bn_in_head,
        pred_bn=False,
        norm_before_pred=args.norm_before_pred,
    )

    teacher_local_group_head = Local_Group_Superivsion_Head(
        in_dim=256,
        hidden_dim=256,
        out_dim=args.local_group_out_dim,
        pred_hidden_dim=0,  # Teacher does not have predictor
        nlayers=3,
        proj_bn=args.use_bn_in_head,
        pred_bn=False,
        norm_before_pred=args.norm_before_pred,
    )

    # Group heads
    student_group_head = Group_Superivsion_Head(
        in_dim=256,
        out_dim=args.group_out_dim,
        hidden_dim=256,
        bottleneck_dim=args.group_bottleneck_dim,
        nlayers=3,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    teacher_group_head = Group_Superivsion_Head(
        in_dim=256,
        out_dim=args.group_out_dim,
        hidden_dim=256,
        bottleneck_dim=args.group_bottleneck_dim,
        nlayers=3,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    student = Mugs_Wrapper(
        backbone=backbone,
        instance_head=student_instance_head,
        local_group_head=student_local_group_head,
        group_head=student_group_head,
    )

    teacher = Mugs_Wrapper(
        backbone=backbone,
        instance_head=teacher_instance_head,
        local_group_head=teacher_local_group_head,
        group_head=teacher_group_head,
    )

    return student, teacher, student_mem, teacher_mem
