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
from src.loss import get_multi_granular_loss
from src.timedataloader import create_time_series_dataloader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.patches as mpatches

def train_mugs(args):

    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    batch_size = args.batch_size
    rsw_file = 'D://PHS Database [No Padding].h5'
    time_series_loader = create_time_series_dataloader(
        hdf5_file_path=rsw_file,
        batch_size=batch_size,
        pick_one=True,
        partial=True,
        partial_type='first',
        N=30,
        shuffle=True
    )
    print(f"Data loaded: there are {len(time_series_loader.dataset)} sequences.")

    # Only the student model
    student,_, student_mem,_ = get_model(args)  # Use the same model for both
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = student.to(device)
    student_mem = student_mem.to(device)
    print(f"Student (and Teacher) model is built: it is a {args.arch} network.")
    
    # Loss functions and optimizer
    all_losses, all_weights = get_multi_granular_loss(args)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    # Optionally resume training
    start_epoch = 0
    if os.path.isfile(os.path.join(args.output_dir, "checkpoint.pth")):
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint.pth"), map_location=device)
        student.load_state_dict(checkpoint['student'])
        student_mem.load_state_dict(checkpoint['student_mem'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch}")

    # Training loop
    print("Starting training!")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()
        train_stats = train_one_epoch(
            student=student,
            teacher=student,  # Same model used for teacher
            all_losses=all_losses,
            all_weights=all_weights,
            data_loader=time_series_loader,
            optimizer=optimizer,
            epoch=epoch,
            student_mem=student_mem,  # share memory between teacher/student
            teacher_mem=student_mem,  
            args=args,
        )

        save_dict = {
            "student": student.state_dict(),
            "student_mem": student_mem.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))

        t2 = time.time()
        log_results = ", ".join([f"{k}: {v:.6f}" for k, v in train_stats.items()])
        print(f"{epoch}-epoch: {log_results}, remaining time: {(t2 - t1) * (args.epochs - epoch) / 3600.0:.2f} hours")

    total_time = time.time() - start_time
    print(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    plot_feature_space(student, time_series_loader, device)

def train_one_epoch(student, teacher, all_losses, all_weights, data_loader, optimizer, epoch, student_mem, teacher_mem, args):
    metric_logger = {'loss': 0}
    total_batches = len(data_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch_idx, batch in enumerate(data_loader):
        target_data_padde, positive_data_padde, target_lengths, positive_lengths, vspotid_list, positive_vspotid_list = batch
        target_data_padded = target_data_padde[:, :, 2:3].to(device)  #  (Batch, Length, 1)
        positive_data_padded = positive_data_padde[:, :, 2:3].to(device)  #  (Batch, Length, 1)
 
        teacher_outputs = teacher([positive_data_padded], local_group_memory_inputs={"mem": teacher_mem}, return_target=True)
        teacher_outputs = [output.detach() for output in teacher_outputs]  # Detach teacher outputs to prevent retaining graph
        student_outputs = student([target_data_padded], local_group_memory_inputs={"mem": student_mem}, return_target=True)

        teacher_instance_target, teacher_local_group_target, teacher_group_target, teacher_memory_tokens = teacher_outputs
        student_instance_target, student_local_group_target, student_group_target, student_memory_tokens = student_outputs

        weigts_sum, total_loss = 0.0, 0.0
        granular_losses = {}

        # Compute losses (Instance, Local Group, Group)
        # Instance loss
        loss_cls, loss_weight = all_losses["instance-sup."], all_weights["instance-sup."]
        if loss_weight > 0:
            instance_loss = loss_cls(teacher_instance_target, student_instance_target)
            weigts_sum += loss_weight
            total_loss += loss_weight * instance_loss
            granular_losses["instance-sup."] = instance_loss.item()

        # Local group loss
        loss_cls, loss_weight = all_losses["local-group-sup."], all_weights["local-group-sup."]
        if loss_weight > 0:
            local_group_loss = loss_cls(teacher_local_group_target, student_local_group_target)
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
        student_mem._dequeue_and_enqueue(student_memory_tokens)
        teacher_mem._dequeue_and_enqueue(teacher_memory_tokens)

        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()

        metric_logger['loss'] += total_loss.item()
        if batch_idx % 10 == 0:
            log_results = ', '.join([f"{name}: {value:.6f}" for name, value in granular_losses.items()])
            print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{total_batches}] Loss: {total_loss.item():.6f}, {log_results}")

    metric_logger['loss'] /= total_batches
    return metric_logger

def plot_feature_space(model, data_loader, device):
    
    model.eval() 
    features = []
    labels = []
    vspotid_a = []
    vspotid_b = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            target_data_padde, _, _, _, vspotid_list, _ = batch
            target_data_padded = target_data_padde[:, :, 2:3].to(device)  #  (Batch, Length, 1)

           
            student_outputs = model([target_data_padded], local_group_memory_inputs={"mem": None}, return_target=True)
            _, _, _, last_layer_features = student_outputs
            features.append(last_layer_features.cpu().numpy())
            labels.extend(vspotid_list) 
            for vspotid in vspotid_list:
                b = vspotid[13:]
                a = vspotid[:7]
                vspotid_a.append(a)
                vspotid_b.append(b)


    features = np.concatenate(features, axis=0)
    # pca = PCA(n_components=2)
    # reduced_features = pca.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_a = sorted(list(set(vspotid_a)))
    unique_b = sorted(list(set(vspotid_b)))
    colors = plt.cm.get_cmap('tab20', len(unique_a))  
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 
           'H', 'd', 'p', '<', '>', 'h', '|', '_', 
           '+', 'x', '.', ',', '1', '2', '3', '4']
    color_legend = {}
    marker_legend = {}
    for i, (x, y) in enumerate(reduced_features):
        color_idx = unique_a.index(vspotid_a[i])
        marker_idx = unique_b.index(vspotid_b[i]) % len(markers)
         # Addlegend dictionary if not in
        if unique_a[color_idx] not in color_legend:
            color_legend[unique_a[color_idx]] = colors(color_idx)
        if unique_b[marker_idx] not in marker_legend:
            marker_legend[unique_b[marker_idx]] = markers[marker_idx]

        plt.scatter(x, y, color=colors(color_idx), marker=markers[marker_idx], label=vspotid_list[i] if i == 0 else None)
    # Create color legend (for 'a')
    color_handles = [mpatches.Patch(color=colors(idx), label=f"a={val}") for idx, val in enumerate(unique_a)]
    
    # Create marker legend (for 'b')
    marker_handles = [plt.Line2D([0], [0], marker=m, color='w', label=f"b={val}",
                                markerfacecolor='k', markersize=20) for m, val in zip(markers, unique_b[:len(markers)])]

    plt.legend(handles=color_handles + marker_handles, loc='best')
    plt.title('Feature Space of Last Layer')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

