import os
import argparse
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from IPython import embed
import lpips
from dggt.models.vggt import VGGT
from dggt.utils.load_fn import load_and_preprocess_images
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import palette_10, concat_list, get_split_gs, gs_dict,get_gs_items,downsample_3dgs, get_gs_conf_flat, get_k_from_gs_map
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import math
def compute_lifespan_loss(gamma):
    return torch.mean(torch.abs(1 / (gamma + 1e-6)))
def compute_gs_conf_slot_loss(gs_conf_raw, cov_score, K):
    """
    K-slot gs_conf alignment loss using soft co-visibility targets.
    gs_conf_raw: [N, K] logits
    cov_score: [N]
    """
    device = gs_conf_raw.device
    N = gs_conf_raw.shape[0]
    desired_slots = 1.0 + (1.0 - cov_score) * (K - 1)  # [N]
    k_idx = torch.arange(K, device=device).float().unsqueeze(0)  # [1, K]
    raw_target = 1.0 - k_idx / desired_slots.unsqueeze(1)  # [N, K]
    soft_target = raw_target.clamp(0.0, 1.0)  # [N, K]

    # 关闭自动平均
    bce_all = F.binary_cross_entropy_with_logits(gs_conf_raw, soft_target, reduction="none")
    # 每个像素内部K个slot损失求和，再像素维度求平均
    loss_per_pixel = bce_all.sum(dim=1)
    return loss_per_pixel.mean()

def compute_covis_weighted_l1(rendered, target, cov_score):
    """
    Co-visibility weighted L1 rendering loss.
    Low co-visibility pixels get higher weight (2×) because they
    appear in fewer frames and need more Gaussian capacity.
    Args:
        rendered: [S, 3, H, W] rendered images
        target:  [S, 3, H, W] target images
        cov_score: [S, H, W] co-visibility score
    Returns:
        weighted L1 loss
    """
    # Weight: cov=1 → w=1.0, cov=0 → w=3.0
    weight = 1.0 + 2.0 * (1.0 - cov_score)  # [S, H, W]
    weight = weight.unsqueeze(1)  # [S, 1, H, W]
    pixel_loss = F.l1_loss(rendered, target, reduction='none')
    return (pixel_loss * weight).mean()
def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1, K = 4):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    if gamma0.dim() == 2:
        # Unfiltered: gamma0 [N, K] -> expand to [N*K]
        conf = torch.exp(sigma.flatten()*torch.concatenate([(t0-t)]*K)**2)
    else:
        # [MODIFIED] Post-filter: gamma0 [M], alpha [M], t [M] all 1D aligned
        conf = torch.exp(sigma * (t0 - t)**2)
    alpha_ = alpha * conf
    return alpha_.float()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="/home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini")
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/xxx')
    parser.add_argument('--sequence_length', type=int, default=4)#8,4
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50000)
    parser.add_argument('--save_image', type=int, default=100)
    parser.add_argument('--save_ckpt', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_splatformer', type=bool, default=False)
    parser.add_argument('--downsample_3dgs', type=bool, default=False)
    return parser.parse_args()
def main(args):
    
    dist.init_process_group(backend='nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.float32
    
    dataset = WaymoOpenDataset(args.image_dir, scene_names=[str(i).zfill(3) for i in range(0,10)], sequence_length=args.sequence_length, mode=1, views=1)
    sampler = DistributedSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)
    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.train()
    model = DDP(model, device_ids=[args.local_rank]) #, find_unused_parameters=True)
    model._set_static_graph()
    
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    for param in model.module.parameters():
        param.requires_grad = False
    for head_name in ["gs_head","instance_head","sky_model", "covis_head" ]: #, "gs_head","instance_head","sky_model", "semantic_head"
        for param in getattr(model.module, head_name).parameters():
            param.requires_grad = True
    optimizer = AdamW([
        {'params': model.module.gs_head.parameters(), 'lr': 4e-5},
        # {'params': model.module.semantic_head.parameters(), 'lr': 1e-4},
        {'params': model.module.instance_head.parameters(), 'lr': 4e-5},
        {'params': model.module.sky_model.parameters(), 'lr': 1e-4},
        {'params': model.module.covis_head.parameters(), 'lr': 4e-5},
    ], weight_decay=1e-4)
    warmup_iterations = 1000
    scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: min((step + 1) / warmup_iterations, 1.0) * 0.5 * (
        1 + math.cos(math.pi * step / args.max_epoch))
)
    for step in tqdm(range(args.max_epoch)):
        sampler.set_epoch(step)       
        for batch in dataloader:
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)
            if 'dynamic_mask' in batch:
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]
                use_depth = True
                if use_depth:
                    depth_map = predictions["depth"][0]
                    point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None,...]
                    # torch.Size([1, 4, 294, 518, 4, 3])
                    point_map = torch.from_numpy(point_map).to(device).float()
                else:      
                    point_map = predictions["world_points"]
                
                # torch.Size([1, 4, 294, 518, 4, 11])
                gs_map = predictions["gs_map"]
                # torch.Size([1, 4, 294, 518, 4])
                gs_conf = predictions["gs_conf"]
                # torch.Size([1, 4, 294, 518])
                dy_map = predictions["dynamic_conf"].squeeze(-1) #B,H,W,1
                semantic_logits = predictions["semantic_logits"]  #road, building, car, truck, person, bicycle, sky, vegetation
                K_gs = get_k_from_gs_map(gs_map)
                # torch.Size([1, 4, 294, 518])
                static_mask = torch.ones_like(bg_mask)
                # torch.Size([2436672, 3])
                static_points = point_map[static_mask].reshape(-1, 3)
                gs_dynamic_list = dy_map[static_mask].sigmoid()
                static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                # ============================================================
                # [MODIFIED] Per-pixel K-Gaussian selection via co-visibility.
                # Physically remove inactive Gaussians to save memory/compute.
                # ============================================================
                # Save raw gs_conf BEFORE filtering for slot loss
                static_gs_conf_raw_all = gs_conf[static_mask]  # [N] or [N, K]
                cov_score = predictions.get('covisibility_score')
                if cov_score is not None and K_gs > 1:
                    static_gs_conf_raw = static_gs_conf_raw_all  # alias
                    cov_score_static = cov_score[static_mask]  # [N]
                    active_count = torch.where(
                        cov_score_static > 0.75, 1,
                        torch.where(cov_score_static > 0.25, 2, K_gs)
                    ).long().clamp(1, K_gs)
                    N_pix = static_gs_conf_raw.shape[0]
                    _, sorted_idx = static_gs_conf_raw.sort(dim=1, descending=True)
                    active_k_mask = torch.zeros(N_pix, K_gs, dtype=torch.bool, device=device)
                    n_range = torch.arange(N_pix, device=device)
                    for k in range(K_gs):
                        active_k_mask[n_range, sorted_idx[:, k]] = (k < active_count)
                    active_mask = active_k_mask.reshape(-1)
                    # Physically remove inactive Gaussians
                    # static_points already [N*K, 3] from point_map K dim
                    # .contiguous() prevents CUDA illegal memory access from
                    # non-contiguous strides after boolean mask indexing.
                    static_points = static_points[active_mask].contiguous()
                    static_rgbs = static_rgbs[active_mask].contiguous()
                    static_opacity = static_opacity[active_mask].contiguous()
                    static_scales = static_scales[active_mask].contiguous()
                    static_rotations = static_rotations[active_mask].contiguous()
                    static_gs_conf = static_gs_conf_raw[active_k_mask].contiguous()  # [M]
                    gs_timestamps = timestamps[frame_idx].repeat_interleave(K_gs)[active_mask].contiguous()
                    gs_dynamic_list = gs_dynamic_list.repeat_interleave(K_gs)[active_mask].contiguous()
                else:
                    static_gs_conf = get_gs_conf_flat(gs_conf, static_mask)
                    if K_gs > 1:
                        # static_points already [N*K, 3] from point_map K dim
                        gs_dynamic_list = gs_dynamic_list.repeat_interleave(K_gs, dim=0)
                        frame_idx = frame_idx.repeat_interleave(K_gs, dim=0)
                    gs_timestamps = timestamps[frame_idx]
                static_opacity = static_opacity * (1 - gs_dynamic_list)
                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]  # [1, H, W, K, 3]
                    bg_mask_i = bg_mask[:, i]
                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)  # [N*K, 3]
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i], bg_mask_i)  # [N*K, C]
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()  # [N]
                    # ============================================================
                    # [MODIFIED] Per-pixel K-Gaussian selection via co-visibility.
                    # Physically remove inactive Gaussians for dynamic objects.
                    # ============================================================
                    if cov_score is not None and K_gs > 1:
                        gs_conf_i = gs_conf[:, i][bg_mask_i]  # [N, K]
                        cov_score_i = cov_score[:, i][bg_mask_i]  # [N]
                        active_count = torch.where(
                            cov_score_i > 0.75, 1,
                            torch.where(cov_score_i > 0.25, 2, K_gs)
                        ).long().clamp(1, K_gs)
                        N_pix = gs_conf_i.shape[0]
                        _, sorted_idx = gs_conf_i.sort(dim=1, descending=True)
                        active_k_mask = torch.zeros(N_pix, K_gs, dtype=torch.bool, device=device)
                        n_range = torch.arange(N_pix, device=device)
                        for k in range(K_gs):
                            active_k_mask[n_range, sorted_idx[:, k]] = (k < active_count)
                        active_mask = active_k_mask.reshape(-1)  # [N*K]
                        # Physically remove inactive Gaussians (dynamic)
                        # .contiguous() prevents non-contiguous strides from
                        # boolean mask indexing causing CUDA illegal memory access.
                        dynamic_point = dynamic_point[active_mask].contiguous()
                        dynamic_rgb = dynamic_rgb[active_mask].contiguous()
                        dynamic_opacity = dynamic_opacity[active_mask].contiguous()
                        dynamic_scale = dynamic_scale[active_mask].contiguous()
                        dynamic_rotation = dynamic_rotation[active_mask].contiguous()
                        gs_dynamic_list_i = gs_dynamic_list_i.repeat_interleave(K_gs)[active_mask].contiguous()
                        dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                    else:
                        gs_dynamic_list_i = torch.concatenate([gs_dynamic_list_i] * K_gs)
                        dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                    dynamic_points.append(dynamic_point)
                    dynamic_rgbs.append(dynamic_rgb)
                    dynamic_opacitys.append(dynamic_opacity)
                    dynamic_scales.append(dynamic_scale)
                    dynamic_rotations.append(dynamic_rotation)
                # Sync after ALL K-filtering (static + dynamic): GPU boolean
                # indexing kernels must complete before alpha_t reads the results.
                # Without this, async reassignment (tensor = tensor[mask]) can
                # race with subsequent CUDA operations.
                if K_gs > 1:
                    torch.cuda.synchronize()
                chunked_renders, chunked_alphas = [], []
                S = extrinsic.shape[0]
                for idx in range(S):
                    t0 = timestamps[idx]
                    # torch.Size([1218336])
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    # torch.Size([1218336, 3]) torch.Size([1218336, 3]) torch.Size([1218336]) torch.Size([1218336, 3]) torch.Size([1218336, 4])
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    if dynamic_points:
                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            static_gs_list,
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]#注释
                        )
                    
                    # torch.Size([3502352, 3])
                    # torch.Size([1789340, 4])
                    # torch.Size([1789340, 3])
                    # torch.Size([1789340])
                    # torch.Size([1, 4, 4])
                    # torch.Size([1, 3, 3])
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points, 
                        quats=rotation, 
                        scales=scales, 
                        opacities=opacity, 
                        colors=rgbs,
                        viewmats=extrinsic[idx][None], 
                        Ks=intrinsic[idx][None],
                        width=W, 
                        height=H, 
                    )
                    
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)
                renders = torch.cat(chunked_renders, dim=0)
                alphas = torch.cat(chunked_alphas, dim=0)
                bg_render = model.module.sky_model(images, extrinsic, intrinsic)
                renders = alphas * renders + (1 - alphas) * bg_render
                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]
                ####################### Loss ###########################
                # ---- Co-visibility weighted render L1 ----
                if cov_score is not None:
                    render_loss = compute_covis_weighted_l1(
                        rendered_image, target_image, cov_score[0]
                    )
                else:
                    render_loss = F.l1_loss(rendered_image, target_image)
                loss = render_loss
                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])
                loss += sky_mask_loss
                # ---- Lifespan regularization ----
                gs_conf_loss = compute_lifespan_loss(static_gs_conf)
                loss += 0.01 * gs_conf_loss
                # ---- K-slot gs_conf alignment ----
                if cov_score is not None and K_gs > 1:
                    cov_score_static = cov_score[static_mask]  # [N]
                    slot_loss = compute_gs_conf_slot_loss(
                        static_gs_conf_raw_all, cov_score_static, K_gs
                    )
                    loss += 0.0001 * slot_loss
                
                #dynamic mask loss
                if 'dynamic_mask' in batch:
                    dynamic_loss = binary_loss_fn(dy_map[0], dynamic_masks[0].float())
                else:
                    dynamic_loss =  binary_loss_fn(dy_map[0], torch.zeros_like(dy_map[0]))
                loss = loss + 0.05 * dynamic_loss  # loss +
                # #### Co-visibility loss ####
                if 'covisibility' in batch and cov_score is not None:
                    covis_gt = batch['covisibility'][0].to(device)  # [S, 1, H, W]
                    covis_gt = covis_gt[:, 0]  # [S, H, W]
                    covis_loss = F.l1_loss(cov_score[0], covis_gt)
                    loss = loss + 0.01 * covis_loss
                # #### semantic segmenation ####
                # if 'semantic_mask' in batch:
                #     gt_sem_mask = batch['semantic_mask'][0,:,0,...].to(device)
                #     #calculate loss
                #     semantic_loss = semantic_loss_fn(semantic_logits[0].permute(0, 3, 1, 2), gt_sem_mask.long())
                #     loss = loss + 0.01 * semantic_loss
                if step >= 0:
                    lpips_val = lpips_loss_fn(rendered_image, target_image)
                    loss += 0.05 * min(step / 1000, 1.0) * lpips_val.mean() # *
            loss.backward()
            # 新增梯度裁剪，保护所有模型参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if args.local_rank == 0 and step % 1 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[{step}/{args.max_epoch}] Total Loss: {loss.item():.4f} | LR: {lr:.6f}\n"
                f"  render_loss: {render_loss.item():.4f}\n"
                f"  sky_mask_loss: {sky_mask_loss.item():.4f}\n"
                f"  gs_conf (w0.01): {(0.01 * gs_conf_loss).item():.4f}\n"
                f"  slot_loss (w0.0001): {(0.0001 * slot_loss).item():.4f}\n"
                f"  dynamic_loss (w0.05): {(0.05 * dynamic_loss).item():.4f}\n"
                f"  covis_loss (w0.01): {(0.01 * covis_loss).item():.4f}\n"
                f"  lpips (w0.05): {(0.05 * min(step / 1000, 1.0) * lpips_val.mean()).item():.4f}"
            )
        if args.local_rank == 0 and step % args.save_image == 0:
            random_frame_idx = random.randint(0, rendered_image.shape[0] - 1)
            rendered = rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)
            target = target_image[random_frame_idx].detach().cpu().clamp(0, 1)
            dy_map_sigmoid = torch.sigmoid(dy_map[0, random_frame_idx]).detach().cpu()  # shape: (H, W)
            dy_map_rgb = dy_map_sigmoid.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
            sem_rgb = alphas[random_frame_idx, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()  # [3, H, W]
            combined = torch.cat([target, rendered, dy_map_rgb, sem_rgb], dim=-1) 
            T.ToPILImage()(combined).save(os.path.join(args.log_dir, "images", f"step_{step}_frame_{random_frame_idx}.png"))
        
        if args.local_rank == 0 and step > 0 and step % args.save_ckpt == 0:
            ckpt_path = os.path.join(args.log_dir, "ckpt", f"model_latest.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at step {step} to {ckpt_path}")
if __name__ == "__main__":
    args = parse_args()
    main(args)