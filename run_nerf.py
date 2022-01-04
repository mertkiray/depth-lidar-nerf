import os, sys

import lpips
import numpy as np
import imageio
import json
import random
import time

import open3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from discriminator import ESRDiscriminator, LSDiscriminator, weights_init_normal, DCDiscriminator, BasicDiscriminator
from vgg19_feature_model import Vgg19, prepare_images_vgg19, visualize_features, unnormalize_image
from external_models_resnet import Resnet
from preprocess.KITTI360.Kitti360Dataset import Kitti360Dataset
from run_nerf_helpers import *

from load_llff import load_llff_data, load_lidar_depth
from load_dtu import load_dtu_data

from loss import SigmaLoss


from data import RayDataset
from torch.utils.data import DataLoader

from utils.generate_renderpath import generate_renderpath

import cv2
# import time

# concate_time, iter_time, split_time, loss_time, backward_time = [], [], [], [], []
from utils.visualization import visualize_depths_as_image, visualize_depths_on_image, visualize_depths_masked_uv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(2)
np.random.seed(0)
#torch.manual_seed(0)
DEBUG = False

from torch.utils.tensorboard import SummaryWriter



def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def batchify_rays_feature_loss(rays_flat, chunk=1024*32, keep_keys=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    # ret_rgb_only = keep_keys and len(keep_keys) == 1 and keep_keys[0] == 'rgb_map'
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if keep_keys and k not in keep_keys:
                # Don't save this returned value to save memory
                continue
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret



def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, depths=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    #print(f'rays_o shape :{rays_o.shape}')
    #print(f'rays_d shape :{rays_d.shape}')

    #rays_o_pcd = open3d.geometry.PointCloud()
    #rays_o_pcd.points = open3d.utility.Vector3dVector(rays_o.to('cpu'))
    #open3d.io.write_point_cloud("rays_o_t2.ply", rays_o_pcd)

    #rays_d_pcd = open3d.geometry.PointCloud()
    #rays_d_pcd.points = open3d.utility.Vector3dVector(rays_d.to('cpu'))
    #open3d.io.write_point_cloud("rays_d_t2.ply", rays_d_pcd)

    #open3d.visualization.draw_geometries([rays_o_pcd, rays_d_pcd])

    #exit(0)

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1) # B x 8
    if depths is not None:
        rays = torch.cat([rays, depths.reshape(-1,1)], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_feature_loss(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  keep_keys=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays_feature_loss(rays, chunk, keep_keys=keep_keys, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    if keep_keys:
        k_extract = [k for k in k_extract if k in keep_keys]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, iteration=0, writer=None, coords=None):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    #for i, c2w in enumerate(tqdm(render_poses)):
    for i , c2w in enumerate(render_poses):
        #print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], retraw=True, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        #if i==0:
        #    print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            depth = depth.cpu().numpy()
            print("max:", np.nanmax(depth))
            # depth = depth / 5 * 255
            # depth_color = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
            # depth_color[np.isnan(depth_color)] = 0
            # imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth_color)
            imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth)



            if writer:
                depth_image, depth_image_world = visualize_depths_as_image(depth)

                writer.add_image('Images/rgb', rgb8, iteration, dataformats='HWC')
                writer.add_image('Images/depth', depth, iteration, dataformats='HW')
                writer.add_image('Images/depth_image', depth_image, iteration, dataformats='HWC')
                writer.add_image('Images/depth_image_world', depth_image_world, iteration, dataformats='HWC')

                if coords is not None:
                    masked_depth_image = visualize_depths_masked_uv(depth, coords)
                    writer.add_image('Images/masked_depth_image', masked_depth_image, iteration, dataformats='HWC')



            np.savez(os.path.join(savedir, '{:03d}.npz'.format(i)), rgb=rgb.cpu().numpy(), disp=disp.cpu().numpy(), acc=acc.cpu().numpy(), depth=depth)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def render_test_ray(rays_o, rays_d, hwf, ndc, near, far, use_viewdirs, N_samples, network, network_query_fn, **kwargs):
    H, W, focal = hwf
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    z_vals = near * (1.-t_vals) + far * (t_vals)

    z_vals = z_vals.reshape([rays_o.shape[0], N_samples])

    rgb, sigma, depth_maps = sample_sigma(rays_o, rays_d, viewdirs, network, z_vals, network_query_fn)

    return rgb, sigma, z_vals, depth_maps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.alpha_model_path is None:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model.parameters())
    else:
        alpha_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        print('Alpha model reloading from', args.alpha_model_path)
        ckpt = torch.load(args.alpha_model_path)
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        if not args.no_coarse:
            model = NeRF_RGB(D=args.netdepth, W=args.netwidth,
                        input_ch=input_ch, output_ch=output_ch, skips=skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model).to(device)
            grad_vars = list(model.parameters())
        else:
            model = None
            grad_vars = []
    

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRF_RGB(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if args.sigma_loss:
        render_kwargs_train['sigma_loss'] = SigmaLoss(args.N_samples, args.perturb, args.raw_noise_std)

    ##########################

    from torchsummary import summary
    print(f'NERF MODEL')
    print(summary(model, (1024, 90)))
    print(f'NERF MODEL FINE')
    print(summary(model_fine, (1024, 90)))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                sigma_loss=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    if network_fn is not None:
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    else:
        # rgb_map, disp_map, acc_map = None, None, None
        # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
        # noise = 0
        # alpha = network_query_fn(pts, viewdirs, network_fine.alpha_model)[...,3]
        if network_fine.alpha_model is not None:
            raw = network_query_fn(pts, viewdirs, network_fine.alpha_model)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        else:
            raw = network_query_fn(pts, viewdirs, network_fine)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)


    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if sigma_loss is not None and ray_batch.shape[-1] > 11:
        depths = ray_batch[:,8]
        ret['sigma_loss'] = sigma_loss.calculate_loss(rays_o, rays_d, viewdirs, near, far, depths, network_query_fn, network_fine)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_ray", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true', 
                        help='render the train set instead of render_poses path')  
    parser.add_argument("--render_mypath", action='store_true', 
                        help='render the test path')         
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    
    # debug
    parser.add_argument("--debug",  action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")
    parser.add_argument("--sigma_loss", action='store_true',
                        help="Use depth supervision by colmap - sigma loss.")
    parser.add_argument("--sigma_lambda", type=float, default=0.1,
                        help="Sigma lambda used for loss.")
    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")
    parser.add_argument("--depth_with_rgb", action='store_true',
                    help="single forward for both depth and rgb")
    parser.add_argument("--normalize_depth", action='store_true',
                    help="normalize depth before calculating loss")
    parser.add_argument("--depth_rays_prop", type=float, default=0.5,
                        help="Proportion of depth rays.")
    parser.add_argument("--feature_loss", action='store_true',
                        help="Use feature loss VGG or not.")
    parser.add_argument("--feature_start_iteration", type=int, default=1000,
                        help="Start of feature loss iteration")
    parser.add_argument("--feature_loss_every_n", type=int, default=15,
                        help="Calculate feature loss every n iteration")
    parser.add_argument("--feature_lambda", type=float, default=0.1,
                        help="Feature loss lambda")
    parser.add_argument("--nH", type=int, default=32,
                        help="Height of total image for feature loss")
    parser.add_argument("--nW", type=int, default=32,
                        help="Width of total image for feature loss")
    parser.add_argument("--gradH", type=int, default=16,
                        help="Height of grad image for feature loss Total ray grad = gradH * gradW")
    parser.add_argument("--gradW", type=int, default=16,
                        help="Width of grad image for feature loss Total ray grad = gradH * gradW")
    parser.add_argument("--feature_loss_type", type=str, default='vgg',
                        help='feature loss type: available: lpips, vgg')
    parser.add_argument("--lpips_spatial", action='store_true',
                        help="Create spatial image from lpips to understand where model learns.")
    parser.add_argument("--lpips_backbone", type=str, default='alex',
                        help="LPIPS BACKBONE Possible: alex, vgg, squeeze")
    parser.add_argument("--vgg_layers", type=str, nargs='*',
                        help="VGG Layers to use")
    parser.add_argument("--vgg_layer_weights", type=float, default=[1, 1], nargs='*',
                        help="VGG Layers weights for each layer")
    parser.add_argument("--vgg_loss_type", type=str, default='l2',
                        help="VGG feature loss type, l1 or l2")
    parser.add_argument("--gan_loss", action='store_true',
                        help="Use GAN Loss or not")
    parser.add_argument("--gan_lambda", type=float, default=0.1,
                        help="GAN Loss lambda")
    parser.add_argument("--gan_start_iteration", type=int, default=500,
                        help="GAN Loss start iteration")
    parser.add_argument("--gan_disc_lrate", type=float, default=5e-4,
                        help="GAN Discriminator learning rate")

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    writer = SummaryWriter('runs/' + args.expname)
    config_params = {}
    for config in open(args.config):
        dict = config.split('=')
        if len(dict) == 2:
            key, value = dict[0].strip().replace('\n' , ''), dict[1].strip().replace('\n' , '')
            config_params[key] = value

    writer.add_text('config',  str(config_params))

    if args.dataset_type == 'llff':

        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]

        if args.colmap_depth:
            depth_gts = load_lidar_depth(args.datadir, hwf=hwf, factor=args.factor )


        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                        (i not in i_test and i not in i_val)])

        # #TODO: For overfit content try delete after:
        #i_train = np.array([0])
        #i_test = np.array([0])
        #i_val = np.array([0])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    elif args.dataset_type == 'dtu':
        images, poses, hwf = load_dtu_data(args.datadir)
        print('Loaded DTU', images.shape, poses.shape, hwf, args.datadir)
        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                        (i not in i_test and i not in i_val)])

        near = 0.1
        far = 5.0
        if args.colmap_depth:
            depth_gts = load_lidar_depth(args.datadir, factor=args.factor, bd_factor=.75)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])
    elif args.render_mypath:
        # render_poses = generate_renderpath(np.array(poses[i_test]), focal)
        render_poses = generate_renderpath(np.array(poses[i_test])[3:4], focal, sc=1)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)


    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    print(render_poses.shape)
    np.save('render_poses.npy', render_poses)
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            if args.render_test:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test', start))
            elif args.render_train:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
            else:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            if args.render_test_ray:
                import pytransform3d.visualizer as pv
                import pytransform3d.transformations as pt
                dataset = Kitti360Dataset(kitti360Path='data/kitti/KITTI-360', seq=0)
                print(i_train)

                xxx = depth_gts[0]['coord']
                print(f'coords: {xxx}')

                # rays_o, rays_d = get_rays(H, W, focal, render_poses[0])
                index_pose = i_train[0]
                rays_o, rays_d = get_rays_by_coord_np(H, W, focal, poses[index_pose,:3,:4], depth_gts[index_pose]['coord'])
                rays_o, rays_d = torch.Tensor(rays_o).to(device), torch.Tensor(rays_d).to(device)
                rgb, sigma, z_vals, depth_maps = render_test_ray(rays_o, rays_d, hwf, network=render_kwargs_test['network_fine'], **render_kwargs_test)
                # sigma = sigma.reshape(H, W, -1).cpu().numpy()
                # z_vals = z_vals.reshape(H, W, -1).cpu().numpy()
                # np.savez(os.path.join(testsavedir, 'rays.npz'), rgb=rgb.cpu().numpy(), sigma=sigma.cpu().numpy(), z_vals=z_vals.cpu().numpy())
                # visualize_sigma(sigma[0, :].cpu().numpy(), z_vals[0, :].cpu().numpy(), os.path.join(testsavedir, 'rays.png'))
                # for k in range(20):
                #     visualize_weights(weights[k*100, :].cpu().numpy(), z_vals[k*100, :].cpu().numpy(), os.path.join(testsavedir, f'rays_weights_%d.png' % k))
                print("colmap depth:", depth_gts[index_pose]['depth'][0])
                print("Estimated depth:", depth_maps[0].cpu().numpy())
                print(depth_gts[index_pose]['coord'])



                pose_34 = poses[i_train[0], :, :]
                pose = np.eye(4)
                pose[:3, :4] = pose_34
                print('[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]')
                print(pose)
                print('[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]')
                print(pose.shape)
                print('rays shape')
                print(rays_d.shape)
                print(rays_o.shape)
                #
                fig = pv.figure()

                #pose = dataset.get_pose_of_frame(6290)
                cam2world = pose
                virtual_image_distance = 1

                sensor_size = np.array([W, H])
                intrinsic = [
                    [focal, 0, sensor_size[0] / 2], [0, focal, sensor_size[1] / 2.0], [0, 0, 1]
                ]
                intrinsic_matrix = np.array(intrinsic)

                fig.plot_transform(A2B=cam2world, strict_check=False)
                fig.plot_camera(
                    cam2world=cam2world, M=intrinsic_matrix, sensor_size=sensor_size,
                    virtual_image_distance=virtual_image_distance)


                #i = 1
                print(rays_d.shape)
                for i in range(500):
                    i = i * 50
                    ray_o = rays_o[i].cpu().numpy()
                    ray_d = rays_d[i].cpu().numpy()

                    fig.plot_vector(start=ray_o,
                                     direction=ray_d,
                                     c=(1.0, 0.5, 0.0))

                    P = np.zeros((2, 3))
                    colors = np.empty((2, 3))
                    P[0] = ray_o

                    # ray_o + (depth_gts[index_pose]['depth'][0] * rays_d)

                    print(depth_gts[index_pose]['depth'][i])
                    print(ray_o)
                    print(ray_d)
                    ray_x = [ray_o[0], ray_o[1], ray_o[2] * depth_gts[index_pose]['depth'][i]]
                    print(ray_x)
                    P[1] = ray_o + (depth_gts[index_pose]['depth'][i] * ray_d)
                    #P[1] = ray_x
                    colors[:, 0] = np.linspace(0, 1, len(colors))
                    colors[:, 1] = np.linspace(0, 1, len(colors))
                    fig.plot(P, [0,1,0])

                 # fig.plot_transform(A2B=np.eye(4))

                # default parameters of a camera in Blender


                #
                # # fig = pv.Figure()


                #pcd = dataset.load_velodyne_data(6290)
                #pcd = dataset.get_velodyne_points_in_camera_coord(pcd)

                #pcd = dataset.load_velodyne_data(5930)
                pcd = dataset.get_velodyne_points_visible_in_rec_camera(5930)

                print(pcd[0])

                pcd = pcd[:, :3]
                print(pcd.shape)

                print(f'rays_d : {rays_d}')

                #fig.scatter(pcd, s=0.1)
                pcd_our = open3d.geometry.PointCloud()
                pcd_our.points = open3d.utility.Vector3dVector(pcd)
                pcd_our.paint_uniform_color([1, 0, 0])

                # Come to world
                pcd_our.transform(pose)


                fig.add_geometry(pcd_our)


                #frame = fig.plot_basis(R=np.eye(3), s=0.5)


                fig.view_init()
                if "__file__" in globals():
                    fig.show()
                else:
                    fig.save_image("__open3d_rendered_image.jpg")









            else:
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
                disps[np.isnan(disps)] = 0
                print('Depth stats', np.mean(disps), np.max(disps), np.percentile(disps, 95))
                imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.percentile(disps, 95)), fps=30, quality=8)


            return

    # Prepare raybatch tensor if batching random rays
    if not args.colmap_depth:
        N_rgb = args.N_rand
    else:
        N_depth = int(args.N_rand * args.depth_rays_prop)
        N_rgb = args.N_rand - N_depth
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        if args.debug:
            print('rays.shape:', rays.shape)
        print('done, concats')
        print(rays.shape)
        print(images[:,None].shape)
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        if args.debug:
            print('rays_rgb.shape:', rays_rgb.shape)
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        rays_depth = None
        if args.colmap_depth:
            print('get depth rays')
            rays_depth_list = []
            for i in i_train:
                rays_depth = np.stack(get_rays_by_coord_np(H, W, focal, poses[i,:3,:4], depth_gts[i]['coord']), axis=0) # 2 x N x 3
                # print(rays_depth.shape)
                rays_depth = np.transpose(rays_depth, [1,0,2])
                depth_value = np.repeat(depth_gts[i]['depth'][:,None,None], 3, axis=2) # N x 1 x 3
                weights = np.repeat(depth_gts[i]['weight'][:,None,None], 3, axis=2) # N x 1 x 3
                rays_depth = np.concatenate([rays_depth, depth_value, weights], axis=1) # N x 4 x 3
                rays_depth_list.append(rays_depth)

            rays_depth = np.concatenate(rays_depth_list, axis=0)
            print('rays_weights mean:', np.mean(rays_depth[:,3,0]))
            print('rays_weights std:', np.std(rays_depth[:,3,0]))
            print('rays_weights max:', np.max(rays_depth[:,3,0]))
            print('rays_weights min:', np.min(rays_depth[:,3,0]))
            print('rays_depth.shape:', rays_depth.shape)
            rays_depth = rays_depth.astype(np.float32)
            print('shuffle depth rays')
            np.random.shuffle(rays_depth)

            max_depth = np.max(rays_depth[:,3,0])
        print('done')
        i_batch = 0

    if args.debug:
        return
    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        # rays_rgb = torch.Tensor(rays_rgb).to(device)
        # rays_depth = torch.Tensor(rays_depth).to(device) if rays_depth is not None else None
        raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size = N_rgb, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda')))
        raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), batch_size = N_depth, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))) if rays_depth is not None else None


    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    print('----------')
    writer.add_image('ImagesGT/color_image_gt', images[i_test[0]].cpu().numpy(), 0, dataformats='HWC')
    if args.colmap_depth:
        depth_image_gt, depth_on_image_gt =  visualize_depths_on_image(depth_gts[i_test[0]], images[i_test[0]].cpu().numpy())
        writer.add_image('ImagesGT/depth_image_gt', depth_image_gt, 0, dataformats='HWC')
        writer.add_image('ImagesGT/depth_on_image_gt', depth_on_image_gt, 0, dataformats='HWC')


    if args.feature_loss:
    #TODO: vgg19 creation
        if args.feature_loss_type == 'vgg':
            print('USING VGG FEATURE LOSS')
            feature_model = Vgg19(args.vgg_layers)
            #feature_model = Resnet(output_layer='layer1')
            feature_model = feature_model.to(device)
        #TODO: lpips creation
        elif args.feature_loss_type == 'lpips':
            print(f'USING LPIPS FEATURE LOSS : {args.lpips_backbone}')
            lpips_loss = lpips.LPIPS(net=args.lpips_backbone, spatial=args.lpips_spatial)
            lpips_loss = lpips_loss.to(device)
        else:
            print('FEATURE LOSS TYPE CAN BE vgg OR lpips')
            exit(-1)

    if args.gan_loss:
        discriminator = ESRDiscriminator(input_shape=[3, args.nH, args.nW])
        #discriminator = LSDiscriminator(input_shape=[3, args.nH, args.nW])
        #discriminator = DCDiscriminator(img_size=args.nH, n_feat=256)
        #discriminator = BasicDiscriminator(input_shape=[3, args.nH, args.nW])
        discriminator = discriminator.to(device)
        #discriminator.apply(weights_init_normal)
        print(f' DISCRIMINATOR: {discriminator}')
        print(f' DISCRIMINATOR OUTPUT SHAPE: {discriminator.output_shape}')

        #criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        criterion_GAN = torch.nn.MSELoss().to(device)
        optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=args.gan_disc_lrate, betas=(0.9, 0.999))

        gan_valid = Variable(torch.ones((1, 1)), requires_grad=False).to(device)
        gan_fake = Variable(torch.zeros((1, 1)), requires_grad=False).to(device)

        gan_noise_mean = 0.
        start_gan_noise_std = 0.1
        gan_noise_std = 0.1

        print(f'GAN VALID SHAPE: {gan_valid.shape}')
        print(f'GAN FAKE SHAPE: {gan_fake.shape}')

    # with torch.no_grad():
    #     normalized_training_images = prepare_images_vgg19(images[i_train])
    #     features = feature_model(normalized_training_images)
    #     # B, F, W, H
    #     # gt_content_features.shape = torch.Size([19, 512, 5, 22]
    #     gt_content_features_1_1 = features['conv1_1']
    #     gt_content_features_1_2 = features['conv1_2']

        # gt_content_features_1_1 = gt_content_features_1_1.to(device)
        # gt_content_features_1_2 = gt_content_features_1_2.to(device)
        #visualize_features(gt_content_features)

    # Ground Truth mean/std
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)

    consistency_keep_keys = ['rgb_map', 'rgb0']

    start = start + 1

    for i in trange(start, N_iters):
        #writer.add_scalar("Iteration", i, i)
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            try:
                batch = next(raysRGB_iter).to(device)
            except StopIteration:
                raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size = N_rgb, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda')))
                batch = next(raysRGB_iter).to(device)
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            if args.colmap_depth:
                # batch_depth = rays_depth[i_batch:i_batch+N_rand]
                try:
                    batch_depth = next(raysDepth_iter).to(device)
                except StopIteration:
                    raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), batch_size = N_depth, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda')))
                    batch_depth = next(raysDepth_iter).to(device)
                batch_depth = torch.transpose(batch_depth, 0, 1)
                batch_rays_depth = batch_depth[:2] # 2 x B x 3
                target_depth = batch_depth[2,:,0] # B
                ray_weights = batch_depth[3,:,0]

            # i_batch += N_rand
            # if i_batch >= rays_rgb.shape[0] or (args.colmap_depth and i_batch >= rays_depth.shape[0]):
            #     print("Shuffle data after an epoch!")
            #     rand_idx = torch.randperm(rays_rgb.shape[0])
            #     rays_rgb = rays_rgb[rand_idx]
            #     if args.colmap_depth:
            #         rand_idx = torch.randperm(rays_depth.shape[0])
            #         rays_depth = rays_depth[rand_idx]
            #     i_batch = 0


        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rgb], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # timer_0 = time.perf_counter()

        if args.colmap_depth:
            N_batch = batch_rays.shape[1]
            batch_rays = torch.cat([batch_rays, batch_rays_depth], 1) # (2, 2 * N_rand, 3)

        # timer_concate = time.perf_counter()


        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        # timer_iter = time.perf_counter()

        if args.colmap_depth and not args.depth_with_rgb:
            # _, _, _, depth_col, extras_col = render(H, W, focal, chunk=args.chunk, rays=batch_rays_depth,
            #                                     verbose=i < 10, retraw=True, depths=target_depth,
            #                                     **render_kwargs_train)
            rgb = rgb[:N_batch, :]
            disp = disp[:N_batch]
            acc = acc[:N_batch]
            depth, depth_col = depth[:N_batch], depth[N_batch:]
            extras = {x:extras[x][:N_batch] for x in extras}
            # extras_col = extras[N_rand:, :]

        elif args.colmap_depth and args.depth_with_rgb:
            depth_col = depth

        # timer_split = time.perf_counter()


        # Mert WIP
        # if i%args.i_print==0:
        #     with torch.no_grad():
        #         print(f'render poses shape: {render_poses[0:1,:,:].shape}')
        #         rgbsanan, dispsanan = render_path(render_poses[0:1,   :,:], hwf, args.chunk, render_kwargs_train)
        #
        #     print('=========')
        #     print(rgbsanan.shape)
        #     print('=========')
        #
        #     plt.imshow(rgbsanan.squeeze())
        #     plt.show()



        # layer_count = 0
        # for parm in render_kwargs_train['network_fine'].parameters():
        #     if parm.grad is not None:
        #         writer.add_histogram('NERF/layer_'+ str(layer_count), parm.grad.data.cpu().numpy(), i)
        #     layer_count += 1

        ## TODO: Close this while not experimenting this slows things down
        # if (args.feature_loss and i >= args.feature_start_iteration and i%args.feature_loss_every_n==0) or (args.gan_loss and i >= args.gan_start_iteration):
        #     layer_count = 0
        #     for parm in render_kwargs_train['network_fine'].parameters():
        #         #print(parm.grad)
        #         if parm.grad is not None:
        #             writer.add_histogram('NERF_FINE/layer_'+ str(layer_count), parm.grad.data.cpu().numpy(), i)
        #         layer_count += 1
        #
        # if (args.feature_loss and i >= args.feature_start_iteration and i % args.feature_loss_every_n == 0) or (
        #         args.gan_loss and i >= args.gan_start_iteration):
        #     layer_count = 0
        #     for parm in render_kwargs_train['network_fn'].parameters():
        #         # print(parm.grad)
        #         if parm.grad is not None:
        #             writer.add_histogram('NERF_COARSE/layer_' + str(layer_count), parm.grad.data.cpu().numpy(), i)
        #         layer_count += 1

        optimizer.zero_grad()

        ## TODO: GAN LOSS
        if args.gan_loss:
            optimizer_D.zero_grad()

        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        depth_loss = 0
        if args.depth_loss:

            #print(f'depth_col: {depth_col[0]} [] target_depth: {target_depth[0]}')

            if not args.no_ndc:
                depth_col = 1/(1-depth_col)
                #target_depth = 1 - (1/target_depth)

            # depth_loss = img2mse(depth_col, target_depth)
            if args.weighted_loss:
                if not args.normalize_depth:
                    depth_loss = torch.mean(((depth_col - target_depth) ** 2) * ray_weights)
                else:
                    #depth_loss = torch.mean((((depth_col - target_depth) / max_depth) ** 2) * ray_weights)
                    depth_loss = torch.mean((((depth_col - target_depth) / torch.max(target_depth)) ** 2) * ray_weights)
            elif args.relative_loss:
                depth_loss = torch.mean(((depth_col - target_depth) / (target_depth + 1e-16))**2)
            else:
                depth_loss = img2mse(depth_col, target_depth)
        sigma_loss = 0
        if args.sigma_loss:
            sigma_loss = extras_col['sigma_loss'].mean()
            # print(sigma_loss)
        trans = extras['raw'][...,-1]

        decay_steps = args.lrate_decay * 1000
        decay_rate = 0.1
        depth_importance = decay_rate ** (global_step / decay_steps)
        depth_importance = 1
        #writer.add_scalar("Train/depth_importance", depth_importance, i)

        loss = img_loss + args.depth_lambda * depth_importance * depth_loss + args.sigma_lambda * sigma_loss



        if (args.feature_loss and i >= args.feature_start_iteration and i%args.feature_loss_every_n==0) or (args.gan_loss and i >= args.gan_start_iteration):
            # TODO: RENDER AN IMAGE AND CALCULATE SEMANTIC LOSS
            img_semantic_id = random.choice(i_train)
            #img_semantic_idx = np.where(i_train == img_semantic_id)
            #img_semantic_id = [img_semantic_id]



            # TODO: CHECK THIS
            feature_pose = poses[img_semantic_id, :3, :4].squeeze(0)

            grad_rays, no_grad_rays, gt_coords = get_rays_cropped_feature_loss_new(H, W, focal, c2w=feature_pose,
                                                                                   nH=args.nH, nW=args.nW,
                                                                                   gradH=args.gradH, gradW=args.gradW)

            gt_image_new = images[img_semantic_id]
            gt_image_new = gt_image_new[gt_coords[2]:gt_coords[3] + 1, gt_coords[0]:gt_coords[1] + 1]
            gt_image_new = gt_image_new[None, ...]
            gt_image_new = gt_image_new.permute(0, 3, 1, 2)


            if i % args.i_print == 0:
                noise = torch.randn(1, 3, args.nH, args.nW) * gan_noise_std + gan_noise_mean
                print_image_gt = gt_image_new + noise
                print_image_gt = print_image_gt.permute(0,2,3,1)
                #print(f' PRINTING GT CROPPED IMAGE AFTER PERMUTE: {print_image_gt.shape}')
                writer.add_image('Images/gt_cropped_image', print_image_gt[0], i, dataformats='HWC')


            grad_extras_feature_loss = render_feature_loss(H, W, focal, chunk=args.chunk,
                            rays=(grad_rays[0].to(device), grad_rays[1].to(device)),
                            keep_keys=consistency_keep_keys,
                            **render_kwargs_train)[-1]
            # rgb0 is the rendering from the coarse network, while rgb_map uses the fine network
            if args.N_importance > 0:
                rgbs = torch.stack([grad_extras_feature_loss['rgb_map'], grad_extras_feature_loss['rgb0']], dim=0)
            else:
                rgbs = grad_extras_feature_loss['rgb_map'].unsqueeze(0)
                #print(f' RGBS RENDERED BY NERF SHAPE: {rgbs.shape}')
            rgbs = rgbs.permute(0,2,1).clamp(0, 1)

            with torch.no_grad():

                no_grad_extras_feature_loss = render_feature_loss(H, W, focal, chunk=args.chunk,
                                                          rays=(no_grad_rays[0].to(device), no_grad_rays[1].to(device)),
                                                          keep_keys=consistency_keep_keys,
                                                          **render_kwargs_train)[-1]
                # rgb0 is the rendering from the coarse network, while rgb_map uses the fine network
                if args.N_importance > 0:
                    no_grad_rgbs = torch.stack([no_grad_extras_feature_loss['rgb_map'], no_grad_extras_feature_loss['rgb0']], dim=0)
                else:
                    no_grad_rgbs = no_grad_extras_feature_loss['rgb_map'].unsqueeze(0)
                    # print(f' RGBS RENDERED BY NERF SHAPE: {rgbs.shape}')
                no_grad_rgbs = no_grad_rgbs.permute(0,2,1).clamp(0, 1)


            im_shape = list(gt_image_new.data.size())

            grad_positions = grad_rays[2]
            no_grad_positions = no_grad_rays[2]
            im_shape[0] = no_grad_rgbs.shape[0]

            mask = torch.rand(im_shape)
            acc_rgb = torch.empty(im_shape)

            mask[:,:, grad_positions[...,0], grad_positions[...,1]] =1
            mask[:,:, no_grad_positions[...,0], no_grad_positions[...,1]] = 0

            acc_rgb[:, :, no_grad_positions[..., 0], no_grad_positions[..., 1]] = no_grad_rgbs
            with torch.enable_grad():
                acc_rgb[:, :, grad_positions[..., 0], grad_positions[..., 1]] = rgbs

            if i % args.i_print == 0:
                ##TODO: DO NOT NEED THESE PERMUTES, CHANGE DATAFORMATS IN ADD IMAGE
                print_rgbs_nerf = acc_rgb.permute(0, 2, 3, 1)
                #print(f'PRINTING IMAGES RENDERED BY NERF: {print_rgbs_nerf.shape} ')
                writer.add_image('Images/mask', mask[0], i)
                writer.add_image('Images/rgb_accumulated', print_rgbs_nerf[0], i, dataformats='HWC')
                if args.N_importance > 0:
                    writer.add_image('Images/mask0', mask[1], i)
                    writer.add_image('Images/rgb_accumulated0', print_rgbs_nerf[1], i, dataformats='HWC')


            if args.feature_loss and i >= args.feature_start_iteration and i%args.feature_loss_every_n==0:
                feature_loss = 0
                feature_loss0 = 0
                if args.feature_loss_type == 'vgg':
                    # print(f'NORMALIZING 1 GT IMAGE WITH SHAPE: {gt_image_new.shape} ')
                    gt_image_normalized_new = prepare_images_vgg19(gt_image_new)

                    ## VGG REQUIRES SHAPE 3 x H x W  ==> satisfy that
                    # print(f'GT IMAGE NORMALIZED_SHAPE: {gt_image_normalized_new.shape}')
                    gt_image_features_new = feature_model(gt_image_normalized_new)

                    # print(f'NORMALIZING IMAGES RENDERED BY NERF: {rgbs.shape} ')
                    normalized_rgbs_nerf = prepare_images_vgg19(acc_rgb)
                    # print(f'NORMALED IMAGES RENDERED BY NERF: {normalized_rgbs_nerf.shape} ')

                    #print(f'EXTRACTING FEATURES RENDERED BY NERF: {normalized_rgbs_nerf.shape} ')
                    features_rgbs = feature_model(normalized_rgbs_nerf)


                    for i_vgg_loss, loss_layer in enumerate(args.vgg_layers):
                        if args.vgg_loss_type == 'l1':
                            feature_loss = feature_loss + torch.mean(torch.abs(features_rgbs[loss_layer][0] - gt_image_features_new[loss_layer])) * args.vgg_layer_weights[i_vgg_loss]
                            if args.N_importance > 0:
                                feature_loss0 = feature_loss0 + torch.mean(torch.abs(features_rgbs[loss_layer][1] - gt_image_features_new[loss_layer])) * args.vgg_layer_weights[i_vgg_loss]

                        elif args.vgg_loss_type == 'l2':
                            feature_loss = feature_loss + torch.mean((features_rgbs[loss_layer][0] - gt_image_features_new[loss_layer]) ** 2) * args.vgg_layer_weights[i_vgg_loss]
                            if args.N_importance > 0:
                                feature_loss0 = feature_loss0 + torch.mean((features_rgbs[loss_layer][1] - gt_image_features_new[loss_layer]) ** 2) * args.vgg_layer_weights[i_vgg_loss]
                        else:
                            print('VGG LOSS TYPE SHOULD BE L1 OR L2')
                            exit(-1)


                ## TODO: LPIPS GRADIENT 0 OLUYOR BAZEN
                ## TODO: SPATIAL MAPLERI RGB OLARAK YAZDIRMAYA BAK?
                if args.feature_loss_type == 'lpips':
                    feature_loss = lpips_loss.forward(gt_image_new, acc_rgb[0], normalize=True)

                    if args.N_importance > 0:
                        feature_loss0 = lpips_loss.forward(gt_image_new, acc_rgb[1], normalize=True)

                    if args.lpips_spatial:
                        if i % args.i_print == 0:
                            writer.add_image('Images/lpips_spatial', feature_loss[0,0,...].data, i, dataformats='HW')
                        feature_loss = feature_loss.mean()
                        if args.N_importance > 0:
                            if i % args.i_print == 0:
                                writer.add_image('Images/lpips_spatial0', feature_loss0[0,0,...].data, i, dataformats='HW')
                            feature_loss0 = feature_loss0.mean()

                feature_loss = feature_loss + feature_loss0

                #TODO: open for feature loss
                loss = loss + feature_loss * args.feature_lambda


            if args.gan_loss and i >= args.gan_start_iteration:

                noise = torch.randn(1, 3, args.nH, args.nW) * gan_noise_std + gan_noise_mean

                pred_nerf = discriminator(acc_rgb[None,0] + noise)
                gan_loss = criterion_GAN(pred_nerf, gan_valid)

                # if i % args.i_print == 0:
                #     print(f'NERF DISCRIMINATOR PRED :{pred_nerf}')

                gan_loss0 = 0
                if args.N_importance > 0:
                    noise = torch.randn(1, 3, args.nH, args.nW) * gan_noise_std + gan_noise_mean
                    pred_nerf0 = discriminator(acc_rgb[None,1] + noise)
                    gan_loss0 = criterion_GAN(pred_nerf0, gan_valid)

                loss = loss + 0.5 * (gan_loss + gan_loss0) * args.gan_lambda
                #loss = loss + gan_loss * args.gan_lambda

                if i%args.i_print==0:
                    writer.add_scalars("Train/NERF_GAN_PRED",{
                    'pred_nerf': pred_nerf,
                    'pred_nerf0' : pred_nerf0
                    }, i)


        # timer_loss = time.perf_counter()

        # TODO: What is this?
        if 'rgb0' in extras and not args.no_coarse:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # # #TODO: Only for testing for now
        # if args.feature_loss and i >= args.feature_start_iteration and i%args.feature_loss_every_n==0:
        #     loss = feature_loss + feature_loss0

        # # #TODO: Only for testing for now
        # if args.gan_loss and i >= args.gan_start_iteration:
        #     #loss = gan_loss + gan_loss0
        #     loss = 0.5 * (gan_loss + gan_loss0) * args.gan_lambda

        loss.backward()
        optimizer.step()



        ##TODO: TRAIN DISCRIMINATOR:
        if args.gan_loss and i >= args.gan_start_iteration:
            optimizer_D.zero_grad()

            noise_real = torch.randn(1, 3, args.nH, args.nW) * gan_noise_std + gan_noise_mean
            noise_fake = torch.randn(1, 3, args.nH, args.nW) * gan_noise_std + gan_noise_mean

            pred_real = discriminator(gt_image_new.contiguous() + noise_real)
            pred_fake = discriminator(acc_rgb[None,0].detach() + noise_fake)

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real , gan_valid)
            loss_fake = criterion_GAN(pred_fake, gan_fake)


            # if i % args.i_print == 0:
            #     print(f'DISCRIMINATOR GT PRED :{pred_real}')
            #     print(f'DISCRIMINATOR FAKE PRED :{pred_fake}')


            loss_fake0 = 0
            if args.N_importance > 0:
                noise_fake0 = torch.randn(1, 3, args.nH, args.nW) * gan_noise_std + gan_noise_mean
                pred_fake0 = discriminator(acc_rgb[None,1].detach() + noise_fake0)
                loss_fake0 = criterion_GAN(pred_fake0, gan_fake)

            # Total loss
            loss_fakes =  0.5 * (loss_fake + loss_fake0)
            loss_dis = loss_fakes + 0.5 * loss_real

            if i % args.i_print == 0:
                writer.add_scalars("Train/GAN_DISCRIMINATOR_PRED", {
                    'pred_real': pred_real,
                    'pred_fake': pred_fake,
                    'pred_fake0': pred_fake0
                }, i)

            loss_dis.backward()
            optimizer_D.step()




        # timer_backward = time.perf_counter()
        # print('\nconcate:',timer_concate-timer_0)
        # print('iter',timer_iter-timer_concate)
        # print('split',timer_split-timer_iter)
        # print('loss',timer_loss-timer_split)
        # print('backward',timer_backward-timer_loss)
        # concate_time.append(timer_concate-timer_0)
        # iter_time.append(timer_iter-timer_concate)
        # split_time.append(timer_split-timer_iter)
        # loss_time.append(timer_loss-timer_split)
        # backward_time.append(timer_backward-timer_loss)

        # if i%10 == 0:
        #     print('\nconcate:',np.mean(concate_time))
        #     print('iter',np.mean(iter_time))
        #     print('split',np.mean(split_time))
        #     print('loss',np.mean(loss_time))
        #     print('backward',np.mean(backward_time))
        #     print('total:',np.mean(concate_time)+np.mean(iter_time)+np.mean(split_time)+np.mean(loss_time)+np.mean(backward_time))

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if i%args.i_print==0:
            writer.add_scalar("lr", new_lrate, i)


        if args.gan_loss:
            # for param_group_d in optimizer_D.param_groups:
            #     param_group_d['lr'] = new_lrate
            #
            # writer.add_scalar("discriminator_lr", new_lrate, i)

            noise_decay_rate = 0.9
            gan_noise_std = start_gan_noise_std * (noise_decay_rate ** (global_step / 1000))
            writer.add_scalar("gan_noise_std", gan_noise_std, i)


        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train['network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if args.i_video > 0 and i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.nanmax(disps)), fps=30, quality=8)


            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0 and len(i_test) > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():

                coords = None
                if args.colmap_depth:
                    coords = depth_gts[i_test[0]]['coord']
                rgbs, disps = render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test,
                                          gt_imgs=images[i_test], savedir=testsavedir, iteration=i, writer=writer, coords=coords)
            print('Saved test set')

            filenames = [os.path.join(testsavedir, '{:03d}.png'.format(k)) for k in range(len(i_test))]

            test_loss = img2mse(torch.Tensor(rgbs), images[i_test])
            test_psnr = mse2psnr(test_loss)


            writer.add_scalar("Test/loss", test_loss, i)
            writer.add_scalar("Test/psnr", test_psnr, i)


        if i%args.i_print==0:

            if args.feature_loss and i >= args.feature_start_iteration and i%args.feature_loss_every_n==0:
                writer.add_scalar("Train/feature_loss", feature_loss * args.feature_lambda, i)
            if args.gan_loss and i >= args.gan_start_iteration:
                writer.add_scalars("Train/GAN",{
                'GAN': gan_loss,
                'DISCRIMINATOR' : loss_dis
                }, i)

            if not args.depth_loss:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            else:
                print(f'depth_col: {depth_col[0]} [] target_depth: {target_depth[0]}')
                print_depth_loss = depth_loss * args.depth_lambda
                writer.add_scalar("Train/depth_loss", print_depth_loss, i)
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Depth Loss: {depth_loss.item()}")
                print(f' depth_col: {depth_col[0]} []  target_depth: {target_depth[0]}')

            writer.add_scalar("Train/loss", loss, i)
            if 'rgb0' in extras and not args.no_coarse:
                writer.add_scalar("Train/img_loss0", img_loss0, i)
                if args.feature_loss and i >= args.feature_start_iteration and i%args.feature_loss_every_n==0:
                    writer.add_scalar("Train/feature_loss0", feature_loss0 * args.feature_lambda, i)


            writer.add_scalar("Train/img_loss", img_loss, i)
            writer.add_scalar("Train/loss", loss, i)
            writer.add_scalar("Train/psnr", psnr, i)


        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
