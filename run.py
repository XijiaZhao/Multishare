import argparse
from train import train_mugs
if __name__ == "__main__":

	#parser = argparse.ArgumentParser("Mugs Training for Time Series Data", parents=[get_args_parser()])
	#args = parser.parse_args()
	args = argparse.Namespace(
    epochs=500,
    batch_size=16,
    seed=42,
	 arch = 'vit_small',
	 output_dir='./output6',
    instance_out_dim=256,
    instance_queue_size=1024,
    instance_temp=0.07,
	 local_group_out_dim=256,
	 local_group_queue_size=1024,
	 local_group_temp=0.07,
	 group_out_dim=1024,
	 group_warmup_teacher_temp=0.04,
	 group_teacher_temp=0.07,
	 group_warmup_teacher_temp_epochs=10,
	 group_student_temp=0.1,
	 loss_weights=[1.0, 0.5, 1.0],
	 lr=1e-4,
	 weight_decay=0.05,
	 clip_grad=3.0,
	 patch_embed_lr_mult=1.0,
	 saveckp_freq=10,

	 #other
	 local_group_knn_top_n = 5,
	 use_bn_in_head = False,
	 norm_before_pred=True,
	 group_bottleneck_dim = 1024,
	 norm_last_layer = False,
	 return_target=True
	
	)

	# Now you can proceed to call your training function
	logs = train_mugs(args)
