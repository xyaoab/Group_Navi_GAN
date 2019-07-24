import argparse
import os
import sys
import torch
import imageio
import numpy as np
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attrdict import AttrDict
torch.backends.cudnn.benchmark = True

from sgan.data.loader import data_loader


from sgan.various_length_models import TrajectoryDiscriminator,LateAttentionFullGenerator

from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

parser.add_argument('--model_path', type=str)

parser.add_argument('--plot_dir', default='./plots/interpretation/')

def get_discriminator(checkpoint, best=False):
    args = AttrDict(checkpoint['args'])
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)
    if best:
        discriminator.load_state_dict(checkpoint['d_best_state'])
    else:
        discriminator.load_state_dict(checkpoint['d_state'])
    discriminator.cuda()
    discriminator.train()
    return discriminator

# For late attent model by full state
def get_attention_generator(checkpoint, best=False):
    args = AttrDict(checkpoint['args'])
    generator = LateAttentionFullGenerator(
        goal_dim=(2,),
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        spatial_dim=2)
    if best:
        generator.load_state_dict(checkpoint['g_best_state'])
    else:
        generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(
        args, obs_traj, obs_traj_rel, pred_traj_gt, seq_start_end, goals, goals_rel, attention_generator, discriminator, plot_dir=None
    ):
    ade_outer, fde_outer = [], []
    total_traj = 0
    count = 1
    guid = 0
    attention_generator.eval()
    with torch.no_grad():
        D_real, D_fake = [], []
        #ade, fde = [], []
    

        pred_traj_fake_rel, _ = attention_generator(
        obs_traj, obs_traj_rel, seq_start_end, seq_len=attention_generator.pred_len, goal_input=goals_rel)


        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
        #ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
        #fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
        # Record the trajectories
        # For each batch, draw only the first sample
        if True:
            _plot_dir = plot_dir+'/'
            if not os.path.exists(_plot_dir):
                os.makedirs(_plot_dir)
            fig = plt.figure()
            goal_point = goals[0,0, :]

            whole_traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            whole_traj_fake = whole_traj_fake[:, 0, :]
            whole_traj_gt = torch.cat([obs_traj, pred_traj_gt],dim=0)
            whole_traj_gt = whole_traj_gt[:, seq_start_end[0][0]:seq_start_end[0][1], :]
            y_upper_limit = max([torch.max(whole_traj_fake[:, 1]).data, 
                                 torch.max(whole_traj_gt[:, :, 1]).data,
                                 goal_point[1].data]) + 0.1
            y_lower_limit = min([torch.min(whole_traj_fake[:, 1]).data, 
                                 torch.min(whole_traj_gt[:, :, 1]).data,
                                 goal_point[1].data]) - 0.1

            x_upper_limit = max([torch.max(whole_traj_fake[:, 0]).data, 
                                 torch.max(whole_traj_gt[:, :, 0]).data,
                                 goal_point[0].data]) + 0.1
            x_lower_limit = min([torch.min(whole_traj_fake[:, 0]).data, 
                                 torch.min(whole_traj_gt[:, :, 0]).data,
                                 goal_point[0].data]) - 0.1

            def plot_time_step(i):
                fig, ax = plt.subplots()
                ax.plot(goal_point[0].cpu().numpy(), goal_point[1].cpu().numpy(), 'gx')
                # plot last three point
                gt_points_x = whole_traj_gt[max(i-2, 0):i+1,:,0].cpu().numpy().flatten()
                gt_points_y = whole_traj_gt[max(i-2, 0):i+1,:,1].cpu().numpy().flatten()
                ax.plot(gt_points_x, gt_points_y, 'b.')

                fake_points_x = whole_traj_fake[max(i-2, 0):i+1,0].cpu().numpy()
                fake_points_y = whole_traj_fake[max(i-2, 0):i+1,1].cpu().numpy()
                if i >= args.obs_len:
                    ax.plot(fake_points_x, fake_points_y, 'r*')
                else:
                    ax.plot(fake_points_x, fake_points_y, 'g.')

                ax.set_ylim(y_lower_limit, y_upper_limit)
                ax.set_xlim(x_lower_limit, x_upper_limit)

                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                return image


            imageio.mimsave(_plot_dir+str(count)+'.gif', 
                            [plot_time_step(i) for i in range(args.obs_len+args.pred_len)],
                            fps=2)


            #ade_sum = evaluate_helper(ade, seq_start_end)
            #fde_sum = evaluate_helper(fde, seq_start_end)
           
    #ade_outer.append(ade_sum)
    #fde_outer.append(fde_sum)



    #ade = sum(ade_outer) / (4 * args.pred_len)
    #fde = sum(fde_outer) / (4)

    return 


def main(args):
    if os.path.isdir(args.model_path):
        
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
  
    else: 
        paths = [args.model_path]

    for path in paths:

        checkpoint = torch.load(path, map_location='cuda:0')
        attention_generator = get_attention_generator(checkpoint)
        discriminator = get_discriminator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        obs_traj = torch.ones([8, 7, 2], dtype=torch.float32).cuda()

        obs_traj_rel = torch.ones([8, 7, 2], dtype=torch.float32).cuda()
        pred_traj_gt = torch.ones([8, 7, 2], dtype=torch.float32).cuda()
        goals = torch.ones([1, 7, 2], dtype=torch.float32).cuda()
        goals_rel = torch.ones([1, 7, 2], dtype=torch.float32).cuda()
        
        for t in range(8):
            obs_traj[t,0,:] = torch.Tensor([1.2 - 0.1*t, 0])
            pred_traj_gt[t,0,:] = torch.Tensor([-0.5,0])#([0.2 - 0.1*t, - 0.01*t])
            obs_traj[t,1,:] = torch.Tensor([-0.7 + 0.1*t, 0.1])
            obs_traj[t,2,:] = torch.Tensor([-0.7 + 0.1*t, 0.15])
            obs_traj[t,3,:] = torch.Tensor([-0.7+ 0.1*t, 0.2])
            obs_traj[t,4,:] = torch.Tensor([1.0 - 0.1*t, -0.1])
            obs_traj[t,5,:] = torch.Tensor([1.0 - 0.1*t, -0.2])
            obs_traj[t,6,:] = torch.Tensor([1.0 - 0.1*t, -0.15])
            
            pred_traj_gt[t,1,:] = torch.Tensor([0.1 + 0.1*t, 0.1])
            pred_traj_gt[t,2,:] = torch.Tensor([0.1 + 0.1*t, 0.15])
            pred_traj_gt[t,3,:] = torch.Tensor([0.1 + 0.1*t, 0.2])
            pred_traj_gt[t,4,:] = torch.Tensor([0.2 - 0.1*t, -0.1])
            pred_traj_gt[t,5,:] = torch.Tensor([0.2 - 0.1*t, -0.2])
            pred_traj_gt[t,6,:] = torch.Tensor([0.2 - 0.1*t, -0.15])
                                      
        obs_traj_rel = obs_traj - obs_traj[0,:,:]
        goals[0,0,:] =  torch.Tensor([-0.5,0])
        goals[0,1,:] =  torch.Tensor([1.5,0.1])
        goals[0,2,:] =  torch.Tensor([1.5,0])
        goals[0,3,:] =  torch.Tensor([1.5,0.2])
        goals[0,4,:] =  torch.Tensor([1.5,-0.1])
        goals[0,5,:] =  torch.Tensor([1.5,-0.2])
        goals[0,6,:] =  torch.Tensor([1.5,-0.3])
    
        goals_rel = goals - obs_traj[0,:,:]
        seq_start_end = torch.Tensor([[0,7]]).to(torch.int64).cuda()
        
        
        evaluate(    
            _args, obs_traj, obs_traj_rel, pred_traj_gt, seq_start_end, goals, goals_rel, attention_generator, discriminator,
           plot_dir=args.plot_dir)
        
        #print(' Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(_args.pred_len, ade, fde))
    

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main(args)
