import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLosses(nn.Module):
    def __init__(self, margin=1.0, feature_dim=256, queue_size=128, temperature=0.1):
        super(ContrastiveLosses, self).__init__()
        self.buffer_size = queue_size
        self.temperature = temperature
        self.margin = margin
        
        # Initialize a buffer to store negative samples (FIFO)
        self.register_buffer("buffer", torch.randn(queue_size, feature_dim))
        self.buffer_ptr = 0  # Pointer to track buffer position
        """
        Initialize the ContrastiveLosses class.

        :param margin: Margin for contrastive loss. Default is 1.0.
        """
    def forward(self, out1, out2):
        
        batch_size = out1.size(0)
        
        """ # Normalize the feature vectors
        out1 = F.normalize(out1, dim=1)
        out2 = F.normalize(out2, dim=1) """
     
        distance = F.pairwise_distance(out1, out2)
        positive_loss = torch.mean(torch.pow(distance, 2))

        # Select a batch from the memory buffer 
        buffer_negatives = self._sample_from_buffer(batch_size)
        negative_pair_distance = F.pairwise_distance(out2, buffer_negatives)
        buffer_loss = torch.mean(torch.pow(torch.clamp(self.margin - negative_pair_distance, min=0.0), 2))
        loss = positive_loss+buffer_loss
        self._update_buffer(out1)
          
        return loss
    def _sample_from_buffer(self, batch_size):
        # Randomly sample a batch
        buffer_size = self.buffer.size(0)
        indices = torch.randint(0, buffer_size, (batch_size,))
        #print(indices)
        return self.buffer[indices]
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


class InstanceDiscriminationLoss(nn.Module):
    def __init__(self, feature_dim=256, queue_size=512, temperature=0.07):
        super(InstanceDiscriminationLoss, self).__init__()
        self.buffer_size = queue_size
        self.temperature = temperature
        # Initialize a buffer 
        self.register_buffer("buffer", torch.randn(queue_size, feature_dim))
        self.buffer_ptr = 0  # Pointer to track buffer position

    """ def forward(self, z1, z2):
        
        #z1: Tensor, shape (batch_size, feature_dim) - features from the teacher (h_t_in(y_c1))
        #z2: Tensor, shape (batch_size, feature_dim) - features from the student's prediction head (pin(h_s_in(y_c2)))
        
        batch_size = z1.size(0)
        
        # Normalize the feature vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1) #(batch,dim)

        # Cosine similarity (z1, z2)
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature) #([16])

        # Negative keys from the buffer
        buffer_negatives = self.buffer[:self.buffer_size].detach()  # (buffer_size, feature_dim)
        buffer_negatives = F.normalize(buffer_negatives, dim=1)
        # Probably don't need the norm of buffer again

        # Compute cosine similarity for negative pairs (z2, buffer_negatives)
        neg_sim = torch.exp(torch.mm(z2, buffer_negatives.T) / self.temperature)  # (batch_size, buffer_size)
         
        # Compute the InfoNCE loss
        numerator = pos_sim
        denominator = torch.cat([neg_sim, pos_sim.unsqueeze(1)], dim=1).sum(dim=1) # (batch)
        loss = -torch.log(numerator / denominator).mean()
          # Update the buffer with the new z1s
        self._update_buffer(z1)
        
        return loss """
    def forward(self, z1, z2):
        """
        z1: Tensor, shape (batch_size, feature_dim) - features from the teacher (h_t_in(y_c1))
        z2: Tensor, shape (batch_size, feature_dim) - features from the student's prediction head (pin(h_s_in(y_c2)))
        """
        batch_size = z1.size(0)
        
        # Normalize the feature vectors
        #z1 = F.normalize(z1, dim=1)
        #z2 = F.normalize(z2, dim=1)
        # Cos (z1, z2)
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)  # Shape: [batch_size]
        numerator = pos_sim
        
        # Initialize denominator with the positive pair similarity
        denominator = numerator.clone()
        buffer_negatives = self.buffer[:self.buffer_size].detach()  # (buffer_size, feature_dim)
        buffer_negatives = buffer_negatives.clone().detach()

        for i in range(0, self.buffer_size, batch_size):
            buffer_subset = buffer_negatives[i:i + batch_size]
            if buffer_subset.size(0) < batch_size:
                break
            # Compute cos (z2 , buffer_subset)
            neg_sim = torch.exp(torch.sum(z2 * buffer_subset, dim=-1) / self.temperature)  #  [batch_size]
            denominator += neg_sim

        loss = -torch.log(numerator / denominator).mean()
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
    instance_supervision_loss = ContrastiveLosses(
        feature_dim=args.instance_out_dim,
        queue_size=args.instance_queue_size,
        temperature=args.instance_temp,
    ).cuda()
    all_losses["instance-sup."] = instance_supervision_loss
    all_weights["instance-sup."] = args.loss_weights[0]

    ## local group discrimination loss
    local_group_supervision = ContrastiveLosses(
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
