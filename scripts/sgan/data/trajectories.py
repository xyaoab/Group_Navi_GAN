import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset
import sys
logger = logging.getLogger(__name__)
path = '/data/xinjiey/Group_Navi_GAN/env/lib/python3.5/site-packages/libsvm'
sys.path.append(path)
from svmutil import *

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, goals_list, goals_rel_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    goals = torch.cat(goals_list, dim=0).permute(2, 0, 1)
    goals_rel = torch.cat(goals_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, goals, goals_rel
    ]

    return tuple(out)

def row_repeat( tensor, num_reps):
    """
    Inputs:
    -tensor: 2D tensor of any shape
    -num_reps: Number of times to repeat each row
    Outpus:
    -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
    """
    col_len = tensor.size(1)
    tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
    tensor = tensor.view(-1, col_len)
    return tensor
## for adding angle, velocity, position differnece
def seq_delta_collate(data):


    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, goals_list, goals_rel_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    
    

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    goals = torch.cat(goals_list, dim=0).permute(2, 0, 1)
    goals_rel = torch.cat(goals_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    
    delta_size = (seq_start_end[:,1] - seq_start_end[:,0]).max()
    # 4, batch, delta_size
    obs_delta = torch.zeros(4, obs_traj.size(1), delta_size)

    model = svm_load_model('//data/xinjiey/Group_Navi_GAN/spencer/group/social_relationships/groups_probabilistic_small.model')
    
    # adding delta
    for start, end in seq_start_end:
        end = end.item()
        start = start.item()
        num_ped = end - start
        end_pos = obs_traj_rel[-1, start:end, :]

        # r1,r1,r1, r2,r2,r2, r3,r3,r3 - r1,r2,r3, r1,r2,r3, r1,r2,r3
        end_pos_difference = row_repeat(end_pos, num_ped) - end_pos.repeat(num_ped, 1)
        end_displacement = obs_traj_rel[-1,start:end,:] - obs_traj_rel[-2,start:end,:] / 0.4
        end_speed = torch.sqrt(torch.sum(end_displacement**2, dim=1)).view(-1,1)

        end_speed_difference =  row_repeat(end_speed, num_ped) - end_speed.repeat(num_ped, 1)
        end_heading = torch.atan2(end_displacement[:,0], end_displacement[:,1]).view(-1,1)
        end_heading_difference =  row_repeat(end_heading, num_ped) - end_heading.repeat(num_ped, 1)
        # num_ped**2
        delta_distance = torch.sqrt(torch.sum(end_pos_difference**2, dim=1)).view(-1,1)
        # num_ped
        delta_speed = torch.abs(end_speed_difference)
        delta_heading = torch.abs(torch.atan2(torch.sin(end_heading_difference), torch.cos(end_heading_difference)))
       
        _x = torch.cat((delta_distance, delta_speed, delta_heading),1)
        _, _, prob = svm_predict([], _x.tolist(), model,'-b 1 -q')
        prob = torch.FloatTensor(prob)[:,0]
        #positive prob >0.5 consider group relationship 
        obs_delta[3, start:end, :num_ped] = (prob>=0.5).long().view(num_ped, num_ped)
        obs_delta[0, start:end, :num_ped] = delta_distance.view(num_ped, num_ped)
        obs_delta[1, start:end, :num_ped] = delta_speed.view(num_ped, num_ped)
        obs_delta[2, start:end, :num_ped] = delta_heading.view(num_ped, num_ped)

            
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, goals, goals_rel, obs_delta
    ]

    return tuple(out)

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        goal_list = []
        goal_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            # Zero padding, the dimension will be used to store exit point
            data = np.pad(data, [(0, 0), (0, 2)], mode='constant')
            # Inverse tracing to find the exit point of each ped
            exit_point_map = {}
            for i in range(len(data)-1, -1, -1):
                if data[i][1] not in exit_point_map:
                    exit_point_map[data[i][1]] = data[i][2:4]
                data[i][4:6] = exit_point_map[data[i][1]]

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_goal_rel = np.zeros((len(peds_in_curr_seq), 2, 1))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_goal = np.zeros((len(peds_in_curr_seq), 2, 1))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_goal = curr_ped_seq[0, -2:].reshape([2, 1])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:4])
                    curr_ped_seq_start = curr_ped_seq[:, 0].reshape([2, 1])
                    # Make coordinates relative
                    curr_ped_goal_rel = curr_ped_goal - curr_ped_seq_start
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # Relative coordinate should base on our position at first timestep
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq_start
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_goal[_idx, :, pad_front:pad_end] = curr_ped_goal
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_goal_rel[_idx, :, pad_front:pad_end] = curr_ped_goal_rel
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    goal_list.append(curr_goal[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    goal_list_rel.append(curr_goal_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        goal_list = np.concatenate(goal_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        goal_list_rel = np.concatenate(goal_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)

        self.goals = torch.from_numpy(goal_list).type(torch.float)
        self.goals_rel = torch.from_numpy(goal_list_rel).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.goals[start:end, :], self.goals_rel[start:end, :]
        ]
        return out
