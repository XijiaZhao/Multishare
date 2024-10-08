import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import os
from pathlib import Path
import sys
import math
from src.headmodel import get_model
from src.objective import get_multi_granular_loss
from src.timedataloader import create_time_series_dataloader
import numpy as np

def train_mugs(args):
    """
    Main training code for Mugs, including building dataloader, models, losses, optimizers, etc.
    """

    # Fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare data loader
    batch_size = args.batch_size
    rsw_file = 'D://PHS Database [No Padding].h5'
    time_series_loader = create_time_series_dataloader(
        hdf5_file_path=rsw_file,
        batch_size=batch_size,
        pick_one=True,
        partial=True,
        partial_type='last',
        N=45,
        shuffle=True
    )
    print(f"Data loaded: there are {len(time_series_loader.dataset)} sequences.")
    """ for batch_idx, batch in enumerate(time_series_loader):
        # Each batch contains target_data, positive_data, vspotid, positive_vspotid
        target_data_padded, positive_data_padded, target_lengths, positive_lengths, vspotid_list, positive_vspotid_list = batch
        
        # Plot the target and positive data for the current batch
        plot_time_series(target_data_padded, positive_data_padded, batch_idx) """


    # Build student and teacher networks
    student, teacher, student_mem, teacher_mem = get_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student, teacher = student.to(device), teacher.to(device)
    student_mem, teacher_mem = student_mem.to(device), teacher_mem.to(device)

    # Teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict(), strict=False)

    # No backpropagation through the teacher
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    all_losses, all_weights = get_multi_granular_loss(args)
    #optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    # Optionally resume training
    start_epoch = 0
    if os.path.isfile(os.path.join(args.output_dir, "checkpoint.pth")):
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint.pth"), map_location=device)
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        student_mem.load_state_dict(checkpoint['student_mem'])
        teacher_mem.load_state_dict(checkpoint['teacher_mem'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch}")

    #training
    print("Starting training!")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()
        # Training one epoch
        train_stats = train_one_epoch(
            student=student,
            teacher=teacher,
            all_losses=all_losses,
            all_weights=all_weights,
            data_loader=time_series_loader,
            optimizer=optimizer,
            epoch=epoch,
            student_mem=student_mem,
            teacher_mem=teacher_mem,
            args=args,
        )

        # Save model checkpoint
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "student_mem": student_mem.state_dict() if student_mem is not None else None,
            "teacher_mem": teacher_mem.state_dict() if teacher_mem is not None else None,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))

        # Logging
        t2 = time.time()
        log_results = ""
        for k, v in train_stats.items():
            log_results += f"{k}: {v:.6f}, "
        print(
            f"{epoch}-epoch: {log_results} remaining time {(t2 - t1) * (args.epochs - epoch) / 3600.0:.2f} hours"
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
def train_one_epoch(
    student,
    teacher,
    all_losses,
    all_weights,
    data_loader,
    optimizer,
    epoch,
    student_mem,
    teacher_mem,
    args,
):
    """
    Main training code for each epoch.
    """
    """ for (name_q, param_q), (name_k, param_k) in zip(student.named_parameters(), teacher.named_parameters()):
        print(f"Student layer: {name_q}, Student parameter shape: {param_q.shape}")
        print(f"Teacher layer: {name_k}, Teacher parameter shape: {param_k.shape}")
        print("-" * 60) """
	

    metric_logger = {'loss': 0}
    total_batches = len(data_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch_idx, batch in enumerate(data_loader):
        target_data_padde, positive_data_padde, target_lengths, positive_lengths, vspotid_list, positive_vspotid_list = batch
        #plot_time_series(target_data_padde, positive_data_padde, batch_idx)
        target_data_padded = target_data_padde[:,:,2:3].to(device)  # Shape: (Batch, Length, 1)
        positive_data_padded = positive_data_padde[:,:,2:3].to(device)  # Shape: (Batch, Length, 1)
        #plot_time_series(target_data_padded, positive_data_padded, batch_idx)
        teacher_input = positive_data_padded
        student_input = target_data_padded
        teacher_outputs = teacher([teacher_input], local_group_memory_inputs={"mem": teacher_mem}, return_target=True)
        student_outputs = student([student_input], local_group_memory_inputs={"mem": student_mem}, return_target=False)

        # Outputs
        teacher_instance_target, teacher_local_group_target, teacher_group_target, teacher_memory_tokens = teacher_outputs
        student_instance_target, student_local_group_target, student_group_target, student_memory_tokens = student_outputs

        weigts_sum, total_loss = 0.0, 0.0
        granular_losses = {}

        # Instance loss
        loss_cls, loss_weight = all_losses["instance-sup."], all_weights["instance-sup."]
        if loss_weight > 0:
            instance_loss = loss_cls(teacher_instance_target,student_instance_target )
            weigts_sum += loss_weight
            total_loss += loss_weight * instance_loss
            granular_losses["instance-sup."] = instance_loss.item()

        # Local group loss
        loss_cls, loss_weight = all_losses["local-group-sup."], all_weights["local-group-sup."]
        if loss_weight > 0:
            local_group_loss = loss_cls(teacher_local_group_target,student_local_group_target)
            weigts_sum += loss_weight
            total_loss += loss_weight * local_group_loss
            granular_losses["local-group-sup."] = local_group_loss.item()

        # Group loss
        loss_cls, loss_weight = all_losses["group-sup."], all_weights["group-sup."]
        if loss_weight > 0:
            group_loss = loss_cls(student_group_target, teacher_group_target, epoch)
            weigts_sum += loss_weight
            total_loss += loss_weight * group_loss
            granular_losses["group-sup."] = group_loss.item()
        total_loss /= weigts_sum
    
        # Update the memory buffer 
        student_features = student_memory_tokens
        student_mem._dequeue_and_enqueue(student_features)
        teacher_mem._dequeue_and_enqueue(teacher_memory_tokens)
        
        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)

        # Student update
        optimizer.zero_grad()
        total_loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = 0.2  # momentum parameter
            for param_q, param_k in zip(
                student.backbone.parameters(),
                teacher.backbone.parameters(),
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if teacher.instance_head is not None:
                for param_q, param_k in zip(
                    student.instance_head.parameters(),
                    teacher.instance_head.parameters(),
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if teacher.local_group_head is not None:
                for param_q, param_k in zip(
                    student.local_group_head.parameters(),
                    teacher.local_group_head.parameters(),
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if teacher.group_head is not None:
                for param_q, param_k in zip(
                    student.group_head.parameters(),
                    teacher.group_head.parameters(),
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        metric_logger['loss'] += total_loss.item()
        if batch_idx % 10 == 0:
            log_results = ''
            for loss_name, loss_value in granular_losses.items():
                log_results += f"{loss_name}: {loss_value:.6f}, "
            print(
                f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{total_batches}] "
                f"Loss: {total_loss.item():.6f}, {log_results}"
            )
    metric_logger['loss'] /= total_batches
    
    return metric_logger
