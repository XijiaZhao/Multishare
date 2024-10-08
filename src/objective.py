import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Vanilla InfoNCE loss adapted for time series data.
    - ncrops: Number of crops used in student networks (set to 1 as per your setup)
    - dim: Feature dimension in queue determined by output dimension of student network
    - queue_size: Queue size
    - temperature: Temperature parameter for InfoNCE loss
    """

    def __init__(self, ncrops=1, dim=256, queue_size=65536, temperature=0.2):
        super().__init__()
        self.queue_size = queue_size
        self.temperature = temperature

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.ncrops = ncrops

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Queue update
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            keys_t = keys.T
            queue_remaining_size = self.queue_size - ptr
            self.queue[:, ptr:] = keys_t[:, :queue_remaining_size]
            self.queue[:, : batch_size - queue_remaining_size] = keys_t[
                :, queue_remaining_size:
            ]
            ptr = batch_size - queue_remaining_size  # Move pointer

        self.queue_ptr[0] = ptr

    def forward(self, student_output, teacher_output):
        """
        Compute InfoNCE loss between student and teacher outputs.
        """
        # Normalize outputs
        student_output = nn.functional.normalize(student_output, dim=-1)
        teacher_output = nn.functional.normalize(teacher_output.detach(), dim=-1)
       
        queue_feat = self.queue.clone().detach()
        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [student_output, teacher_output]).unsqueeze(-1)
        # Negative logits: NxK
        #print(student_output.size,queue_feat.size)
        l_neg = torch.einsum("nc,ck->nk", [student_output, queue_feat])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # Apply temperature
        logits /= self.temperature
        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        loss = self.CrossEntropyLoss(logits, labels)
        self._dequeue_and_enqueue(teacher_output)

        return loss

class InstanceDiscriminationLoss(nn.Module):
    def __init__(self, feature_dim=256, queue_size=512, temperature=0.07):
        super(InstanceDiscriminationLoss, self).__init__()
        self.buffer_size = queue_size
        self.temperature = temperature
        # Initialize a buffer to store negative samples (FIFO)
        self.register_buffer("buffer", torch.randn(queue_size, feature_dim))
        self.buffer_ptr = 0  # Pointer to track buffer position
        
    def forward(self, z1, z2):
        """
        z1: Tensor, shape (batch_size, feature_dim) - features from the teacher (h_t_in(y_c1))
        z2: Tensor, shape (batch_size, feature_dim) - features from the student's prediction head (pin(h_s_in(y_c2)))
        """
        batch_size = z1.size(0)
        
        # Normalize the feature vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute positive cosine similarity (z1, z2)
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)

        # Sample negative keys from the buffer
        buffer_negatives = self.buffer[:self.buffer_size].detach()  # (buffer_size, feature_dim)
        buffer_negatives = F.normalize(buffer_negatives, dim=1)
        
        # Compute cosine similarity for negative pairs (z2, buffer_negatives)
        neg_sim = torch.exp(torch.mm(z2, buffer_negatives.T) / self.temperature)  # (batch_size, buffer_size)
        
        # Compute the InfoNCE loss
        numerator = pos_sim
        denominator = torch.cat([neg_sim, pos_sim.unsqueeze(1)], dim=1).sum(dim=1)
        loss = -torch.log(numerator / denominator).mean()
          # Update the buffer with the new z1s
        self._update_buffer(z1)
        
        return loss
    
    def _update_buffer(self, new_entries):
        batch_size = new_entries.size(0)
        if self.buffer_ptr + batch_size > self.buffer_size:
            # If buffer overflows, wrap around (FIFO)
            overflow = (self.buffer_ptr + batch_size) - self.buffer_size
            self.buffer[self.buffer_ptr:] = new_entries[:batch_size-overflow]
            self.buffer[:overflow] = new_entries[batch_size-overflow:]
            self.buffer_ptr = overflow
        else:
            self.buffer[self.buffer_ptr:self.buffer_ptr+batch_size] = new_entries
            self.buffer_ptr = (self.buffer_ptr + batch_size) % self.buffer_size
            
class ClusteringLoss(nn.Module):
    """
    Clustering loss adapted for time series data.
    - out_dim: Center dimension determined by output dimension of student network
    - ncrops: Number of crops used in student networks (set to 1 as per your setup)
    - warmup_teacher_temp: Initial value for the teacher temperature
    - teacher_temp: Final value (after linear warmup) of the teacher temperature
    - warmup_teacher_temp_epochs: Number of warmup epochs for the teacher temperature
    - nepochs: Total training epochs
    - student_temp: Temperature parameter in student output
    - center_momentum: EMA parameter for center update
    """

    def __init__(
        self,
        out_dim,
        ncrops=1,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=0,
        nepochs=100,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # Teacher temperature schedule
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Compute clustering loss between student and teacher outputs.
        """
        # Student output
        student_out = student_output / self.student_temp

        # Teacher output with centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach()

        # Compute loss
        loss = torch.sum(
            -teacher_out * F.log_softmax(student_out, dim=-1), dim=-1
        ).mean()

        # Update center
        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
 
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


def get_multi_granular_loss(args):
    """
    Build the multi-granular loss adapted for time series data.
    """
    all_losses, all_weights = {}, {}

    ## Instance discrimination loss
    """ instance_supervision_loss = InfoNCELoss(
        ncrops=1,
        dim=args.instance_out_dim,
        queue_size=args.instance_queue_size,
        temperature=args.instance_temp,
    ).cuda()
    all_losses["instance-sup."] = instance_supervision_loss
    all_weights["instance-sup."] = args.loss_weights[0]

    ## Build the local group discrimination loss
    local_group_supervision = InfoNCELoss(
        ncrops=1,
        dim=args.local_group_out_dim,
        queue_size=args.local_group_queue_size,
        temperature=args.local_group_temp,
    ).cuda()
    all_losses["local-group-sup."] = local_group_supervision
    all_weights["local-group-sup."] = args.loss_weights[1]
    """
    instance_supervision_loss = InstanceDiscriminationLoss(
        feature_dim=args.instance_out_dim,
        queue_size=args.instance_queue_size,
        temperature=args.instance_temp,
    ).cuda()
    all_losses["instance-sup."] = instance_supervision_loss
    all_weights["instance-sup."] = args.loss_weights[0]

    ## local group discrimination loss
    local_group_supervision = InstanceDiscriminationLoss(
        feature_dim=args.local_group_out_dim,
        queue_size=args.local_group_queue_size,
        temperature=args.local_group_temp,
    ).cuda()
    all_losses["local-group-sup."] = local_group_supervision
    all_weights["local-group-sup."] = args.loss_weights[1]

    ## Group discrimination loss
    group_loss = ClusteringLoss(
        out_dim=args.group_out_dim,
        ncrops=1,
        warmup_teacher_temp=args.group_warmup_teacher_temp,
        teacher_temp=args.group_teacher_temp,
        warmup_teacher_temp_epochs=args.group_warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp=args.group_student_temp,
        center_momentum=0.9,
    ).cuda()
    all_losses["group-sup."] = group_loss
    all_weights["group-sup."] = args.loss_weights[2]
    return all_losses, all_weights
