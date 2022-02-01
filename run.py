import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo
from lib.load_data import load_data

from camera.camera_dict import camera_dict


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')


    # Curriculum Learning
    parser.add_argument(
        "--add_ie", default=5000, type=int, 
        help="step to start learning ie"
    )
    parser.add_argument(
        "--add_od", default=8000, type=int,
        help="step to start learning od"
    )
    parser.add_argument(
        "--add_prd", type=int, default=11000, 
        help="step to use prd loss"
    )

    parser.add_argument("--ray_o_noise_scale", type=float, default=1e-3, help="scale of offset distortion")
    parser.add_argument("--ray_d_noise_scale", type=float, default=1e-3, help="scale of direction distortion")
    parser.add_argument("--intrinsics_noise_scale", default=1.0, type=float)
    parser.add_argument("--distortion_noise_scale", default=1e-2, type=float)
    parser.add_argument("--grid_size", type=int, default=10)

    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise Exception("Boolean value expected.")
    
    parser.add_argument(
        "--multiplicative_noise", type=str2bool, nargs="?", const=True, 
        default=False, help="learn multiplicative noise"
    )


    return parser

def sample_rays_random(H, W, images, i_train, shuffled_image_idx, shuffled_ray_idx, i_batch, poses, model, camera_model, args, cfg, cfg_train):
    image_idx_curr_step = shuffled_image_idx[i_batch:i_batch + cfg_train.N_rand]
    h_list = shuffled_ray_idx[i_batch:i_batch + cfg_train.N_rand] % (H * W) // W
    w_list = shuffled_ray_idx[i_batch:i_batch + cfg_train.N_rand] % (H * W) % W
    
    select_coords = np.stack([w_list, h_list], -1)
    assert select_coords[:, 0].max() < W
    assert select_coords[:, 1].max() < H
    
    image_idx_curr_step_tensor = torch.from_numpy(
        image_idx_curr_step
    ).to(device)

    image_idx_curr_step_unique = np.unique(image_idx_curr_step)
    poses_unique = model.get_modified_poses(poses[image_idx_curr_step_unique], image_idx_curr_step_unique)
    poses = torch.zeros((cfg_train.N_rand, *poses_unique[0].shape))
    for i, c2w in zip(image_idx_curr_step_unique, poses_unique):
        poses[image_idx_curr_step == i] = c2w

    kps_list = torch.from_numpy(select_coords).cuda()

    index_train = np.array(i_train)[shuffled_image_idx[i_batch: i_batch + cfg_train.N_rand]]

    i_batch += cfg_train.N_rand
    if i_batch >= len(shuffled_ray_idx):
        print("Shuffle data after an epoch!")
        np.random.shuffle(shuffled_ray_idx)
        shuffled_image_idx = shuffled_ray_idx // (H * W)
        i_batch = 0
    

    rays_o, rays_d, viewdirs = dvgo.get_rays_kps_use_camera(
                H, 
                W, 
                camera_model, 
                poses, 
                ndc=cfg.data.ndc, 
                inverse_y=cfg.data.inverse_y,    
                flip_x=cfg.data.flip_x, 
                flip_y=cfg.data.flip_y, 
                kps_list=kps_list)
            
    target = images[index_train, kps_list[:,1], kps_list[:,0]] 


    return rays_o, rays_d, viewdirs, target, i_batch


def sample_rays_in_maskcache(H, W, images, i_train, shuffled_image_idx, shuffled_ray_idx, i_batch, poses, model, camera_model, args, cfg, cfg_train, render_kwargs):
    DEVICE = images[0].device 
    N_rand = cfg_train.N_rand

    rays_o_tr = torch.zeros([N_rand,3], device=DEVICE)
    rays_d_tr = torch.zeros([N_rand,3], device=DEVICE)
    viewdirs_tr = torch.zeros([N_rand,3], device=DEVICE)
    target_tr = torch.zeros([N_rand,3], device=DEVICE)
    
    top = 0
    while top < N_rand:
        image_idx_curr_step = shuffled_image_idx[i_batch:i_batch + N_rand]
        h_list = shuffled_ray_idx[i_batch:i_batch + N_rand] % (H * W) // W
        w_list = shuffled_ray_idx[i_batch:i_batch + N_rand] % (H * W) % W
        
        select_coords = np.stack([w_list, h_list], -1)
        assert select_coords[:, 0].max() < W
        assert select_coords[:, 1].max() < H
        
        image_idx_curr_step_tensor = torch.from_numpy(
            image_idx_curr_step
        ).to(device)

        image_idx_curr_step_unique = np.unique(image_idx_curr_step)
        poses_unique = model.get_modified_poses(poses[image_idx_curr_step_unique], image_idx_curr_step_unique)
        poses = torch.zeros((N_rand, *poses_unique[0].shape))
        for i, c2w in zip(image_idx_curr_step_unique, poses_unique):
            poses[image_idx_curr_step == i] = c2w


        kps_list = torch.from_numpy(select_coords).cuda()

        index_train = np.array(i_train)[shuffled_image_idx[i_batch: i_batch + N_rand]] 

        rays_o, rays_d, viewdirs, mask = dvgo.get_rays_kps_use_camera_in_maskcache_sampling(
            H, 
            W, 
            camera_model, 
            poses, 
            cfg.data.ndc, 
            cfg.data.inverse_y,    
            cfg.data.flip_x, 
            cfg.data.flip_y, 
            model, 
            render_kwargs, 
            kps_list)
        n = mask.sum()
        rays_o_tr[top:top+n].copy_(rays_o[:N_rand-top].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[:N_rand-top].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[:N_rand-top].to(DEVICE))
        mask = mask.to(DEVICE)
        target_tr[top:top+n].copy_(images[index_train[mask], kps_list[mask, 1], kps_list[mask, 0], :][:N_rand-top])

        top += n

        i_batch += N_rand
        if i_batch >= len(shuffled_ray_idx):
            print("Shuffle data after an epoch!")
            np.random.shuffle(shuffled_ray_idx)
            shuffled_image_idx = shuffled_ray_idx // (H * W)
            i_batch = 0


    return rays_o_tr, rays_d_tr, viewdirs_tr, target_tr, i_batch




@torch.no_grad()
def render_viewpoints_use_camera(model, render_poses, HW, camera_model, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW)

    if render_factor!=0:
        HW = np.copy(HW)
        #Ks = np.copy(Ks)
        HW //= render_factor
        #Ks[:, :2, :3] //= render_factor
        camera_model.get_intrinsic()[:, :2, :3] //= render_factor

    rgbs = []
    disps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, camera_model.get_intrinsic(), c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        
        keys = ['rgb_marched', 'disp']
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(16, 0), rays_d.split(16, 0), viewdirs.split(16, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        disp = render_result['disp'].cpu().numpy()

        rgbs.append(rgb)
        disps.append(disp)
        if i==0:
            print('Testing', rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        '''
        print('Testing psnr', [f'{p:.3f}' for p in psnrs])
        if eval_ssim: print('Testing ssim', [f'{p:.3f}' for p in ssims])
        if eval_lpips_vgg: print('Testing lpips (vgg)', [f'{p:.3f}' for p in lpips_vgg])
        if eval_lpips_alex: print('Testing lpips (alex)', [f'{p:.3f}' for p in lpips_alex])
        '''
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    return rgbs, disps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])

    return data_dict

def initialize_camera_model(H, W, focal, args):
    intrinsic_init = torch.tensor(
        [
            [focal, 0, W/2, 0],
            [0, focal, H/2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    )
    camera_kwargs = {
        "intrinsics": intrinsic_init,
        "args": args,
        "H": H,
        "W": W,
    }
    camera_model = camera_dict[args.camera_model](**camera_kwargs).cuda()
    return camera_model

def compute_bbox_by_cam_frustrm_camera_model(args, cfg, camera_model, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm_camera_model: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    
    H = camera_model.H
    W = camera_model.W

    K = camera_model.get_intrinsic_without_noise()

    for c2w in poses[i_train]:
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(H, W, K, 
                c2w, ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y, 
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))

    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path) 
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def gradient_descent_step(model, optimizer, render_result, target, global_step, psnr_lst, cfg_train):
    optimizer.zero_grad(set_to_none=True)
    loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
    psnr = utils.mse2psnr(loss.detach()).item()
    if cfg_train.weight_entropy_last > 0:
        pout = render_result['alphainv_cum'][...,-1].clamp(1e-6, 1-1e-6)
        entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
        loss += cfg_train.weight_entropy_last * entropy_last_loss
    if cfg_train.weight_rgbper > 0:
        rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
        rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
        loss += cfg_train.weight_rgbper * rgbper_loss
    if cfg_train.weight_tv_density>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
        loss += cfg_train.weight_tv_density * model.density_total_variation()
    if cfg_train.weight_tv_k0>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
        loss += cfg_train.weight_tv_k0 * model.k0_total_variation()
    loss.backward()
    optimizer.step()
    psnr_lst.append(psnr)

    return loss

def update_lr(optimizer, cfg_train):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (1/decay_steps)
    for i_opt_g, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = param_group['lr'] * decay_factor


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, camera_model, coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]
    H, W = HW[0]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None
    
    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) and reload_ckpt_path is None:
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
    model_kwargs['len_data'] = len(i_train)
    model = dvgo.DirectVoxGO(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        mask_cache_path=coarse_ckpt_path,
        **model_kwargs)
    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    model = model.to(device)

    # init optimizer
    optimizer = utils.create_optimizer_or_freeze_model(model, camera_model, cfg_train, global_step=0)

    # load checkpoint if there is
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start, camera_model = utils.load_checkpoint( 
                model, optimizer, camera_model, reload_ckpt_path, args.no_reload_optimizer)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    if data_dict['irregular_shape']:
        rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        raise NotImplementedError
    else:
        rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)


    shuffled_ray_idx = np.arange(len(i_train) * H * W)
    np.random.shuffle(shuffled_ray_idx)
    shuffled_image_idx = shuffled_ray_idx // (H * W)
    i_batch = 0

    # view-count-based learning rate TODO: implement with the initial noisy intrinsics?
    '''if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()'''

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            model.scale_volume_grid(model.num_voxels * 2)
            optimizer = utils.create_optimizer_or_freeze_model(model, camera_model, cfg_train, global_step=0)
            model.density.data.sub_(1)

        # Curriculum learning
        if global_step == start-1 and global_step < args.add_ie and camera_model is not None:
            camera_model.intrinsics_noise.requires_grad_(False)
            print("Deactivated learnable ie")

        if global_step == start-1 and global_step < args.add_od and camera_model is not None:
            camera_model.ray_o_noise.requires_grad_(False)
            camera_model.ray_d_noise.requires_grad_(False)
            print("Deactivate learnable od")

        if global_step == args.add_ie and camera_model is not None:
            camera_model.intrinsics_noise.requires_grad_(True)
            print("Activated learnable ie")

        if global_step == args.add_od and camera_model is not None:
            camera_model.ray_o_noise.requires_grad_(True)
            camera_model.ray_d_noise.requires_grad_(True)
            print("Activated learnable od")

        '''    
            #TODO for projected ray distance loss
            target_s = images[index_train, h_list, w_list]
            img_i = np.random.choice(index_train)
            img_i_train_idx = np.where(i_train == img_i)[0][0]
        '''

        if cfg_train.ray_sampler == 'in_maskcache':
            rays_o, rays_d, viewdirs, target, i_batch = sample_rays_in_maskcache(
                H, 
                W, 
                images, 
                i_train,
                shuffled_image_idx, 
                shuffled_ray_idx, 
                i_batch, 
                poses, 
                model, 
                camera_model, 
                args, 
                cfg,
                cfg_train, 
                render_kwargs)

        elif cfg_train.ray_sampler == 'random':
            rays_o, rays_d, viewdirs, target, i_batch = sample_rays_random(
                H, 
                W, 
                images, 
                i_train,
                shuffled_image_idx, 
                shuffled_ray_idx, 
                i_batch, 
                poses,
                model, 
                camera_model, 
                args, 
                cfg,
                cfg_train
            )

        else:
            raise NotImplementedError


        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)


        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        # gradient descent step
        loss = gradient_descent_step(model, optimizer, render_result, target, global_step, psnr_lst, cfg_train)
        
        # update lr
        update_lr(optimizer, cfg_train)
        
        
        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'MaskCache_kwargs': model.get_MaskCache_kwargs(),
                'model_state_dict': model.state_dict(),
                'camera_model_state_dict': camera_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'MaskCache_kwargs': model.get_MaskCache_kwargs(),
            'model_state_dict': model.state_dict(),
            'camera_model_state_dict': camera_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # initialize camera model
    camera_model = initialize_camera_model(data_dict['hwf'][0], data_dict['hwf'][1], data_dict['hwf'][2], args)

    # coarse geometry searching
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm_camera_model(args, cfg, camera_model, **data_dict)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
            xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
            data_dict=data_dict, stage='coarse',
            camera_model=camera_model)
    eps_coarse = time.time() - eps_coarse
    eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
    print('train: coarse geometry searching in', eps_time_str)

    # fine detail reconstruction
    eps_fine = time.time()
    coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
            model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
            thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path,
            camera_model=camera_model)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # Choose camera model
    args.camera_model = "pinhole_rot_noise_10k_rayo_rayd"

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # TODO Ks -> camera_model.get_intrinsic()
    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device) 
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
        camera_model = initialize_camera_model(data_dict['hwf'][0], data_dict['hwf'][1], data_dict['hwf'][2], args)
        camera_model = utils.load_camera_model(camera_model, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints_use_camera(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                camera_model=camera_model,
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints_use_camera(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                camera_model=camera_model,
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints_use_camera(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                camera_model=camera_model,
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    print('Done')

