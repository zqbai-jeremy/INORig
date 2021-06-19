from tqdm import tqdm
import time
import face_alignment
import face_alignment.detection.sfd as face_detector_module

# Internal libs
import data.BFM.utils as bfm_utils
import core_dl.module_util as dl_util
from networks.sub_nets import INORig
from demo_utils import *


def init_model(checkpoint_path):
    model = INORig(opt_step_size=1e-2)
    pre_checkpoint = dl_util.load_checkpoints(checkpoint_path)
    model.load_state(pre_checkpoint['net_instance'])
    model.cuda()
    model.eval()
    model.training = False

    bfm = model.opt_layer.bfm
    # MM_base_dir = './external/face3d/examples/Data/BFM'
    # bfm_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
    face_region_mask = bfm.face_region_mask.copy()
    # face_region_mask[bfm_info['nose_hole'].ravel()] = False
    model.face_region_mask = face_region_mask

    return model


if __name__ == '__main__':
    torch.manual_seed(7777777)
    np.random.seed(7777777)

    save_video_mesh = False
    rig_img_dir = './examples/case1'
    rig_out_dir = './out_dir/case1'
    src_vid_path = './examples/videos/clip1.mp4'
    vid_out_dir = './out_dir/case1_clip1'
    checkpoint_path = '/home/ziqianb/Documents/Face_NonRigidMVS/logs/Oct03_18-58-06_cs-guv-gpu02_PersonalizedRig_AdapResNetIDAdapMlpPCAMlpExpFeatLinear_OptAdapExpModel_AllMultiLevel_8Ep_ShadingDynAlbedo_NotAffectGeo_ESRC_SfsnetSH_2-7views/checkpoints/iter_039000.pth.tar'
    video_checkpoint_path = '/home/ziqianb/Documents/Face_NonRigidMVS/logs/Oct03_18-58-06_cs-guv-gpu02_PersonalizedRig_AdapResNetIDAdapMlpPCAMlpExpFeatLinear_OptAdapExpModel_AllMultiLevel_8Ep_ShadingDynAlbedo_NotAffectGeo_ESRC_SfsnetSH_2-7views/checkpoints/iter_039000.pth.tar'

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
    face_detector = face_detector_module.FaceDetector(device='cuda', verbose=False)
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    # Build Face Rig ---------------------------------------------------------------------------------------------------
    print('Initializing model ...')
    model = init_model(checkpoint_path)
    print('Preprocessing images ...')
    img, ori_img_rig, kpts, ori_imgs_rig, ss, ts = preprocess(rig_img_dir, fa, face_detector)
    print('Reconstructing ...')
    opt_verts, opt_verts_obj, opt_sp_verts, _, opt_vis_masks, _, albedo_list, colors_list, _, _, _ = \
        predict(model, img, ori_img_rig, kpts)
    sh_coeffs = model.opt_layer.sh_coeff
    print('Visualizing results ...')
    vis_list = visualize(opt_verts, opt_verts_obj, albedo_list, colors_list, sh_coeffs, ori_imgs_rig, ss, ts,
                         model.face_region_mask, model.opt_layer.bfm, None)
    print('Saving Reconstruction ...')
    face_full, face_valid = convert_to_output_formate(opt_verts, albedo_list, model.opt_layer.bfm, model, None)
    save_outputs(rig_out_dir, face_full, face_valid, vis_list)

    # Video Reconstruction and Facial Motion Transfer ------------------------------------------------------------------
    # Load video
    in_vid = cv2.VideoCapture(src_vid_path)
    n_total = int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(in_vid.get(cv2.CAP_PROP_FPS) + 0.5)
    H = int(in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(in_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # W = int(W // 2.5 + 0.5)
    # H = int(H // 2.5 + 0.5)
    print('Total number of frames: %d' % n_total)
    print('FPS: %d' % fps)

    # Create output video
    if not os.path.exists(vid_out_dir):
        os.mkdir(vid_out_dir)
    if not os.path.exists(os.path.join(vid_out_dir, 'mesh')) and save_video_mesh:
        os.mkdir(os.path.join(vid_out_dir, 'mesh'))
    recon_path = os.path.join(vid_out_dir, 'output_recon.webm')
    recon_vid = cv2.VideoWriter(recon_path, cv2.VideoWriter_fourcc('V', 'P', '8', '0'), fps, (W, H))
    exptrans_path = os.path.join(vid_out_dir, 'output_exptrans.webm')
    exptrans_vid = cv2.VideoWriter(exptrans_path, cv2.VideoWriter_fourcc('V', 'P', '8', '0'), fps, (W, H))
    clip_path = os.path.join(vid_out_dir, 'input_clip.webm')
    clip_vid = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc('V', 'P', '8', '0'), fps, (W, H))
    exptrans_albedo_path = os.path.join(vid_out_dir, 'output_exptrans_albedo.webm')
    exptrans_albedo_vid = cv2.VideoWriter(exptrans_albedo_path, cv2.VideoWriter_fourcc('V', 'P', '8', '0'), fps, (W, H))
    exptrans_colors_path = os.path.join(vid_out_dir, 'output_exptrans_colors.webm')
    exptrans_colors_vid = cv2.VideoWriter(exptrans_colors_path, cv2.VideoWriter_fourcc('V', 'P', '8', '0'), fps, (W, H))
    colors_path = os.path.join(vid_out_dir, 'output_colors.webm')
    colors_vid = cv2.VideoWriter(colors_path, cv2.VideoWriter_fourcc('V', 'P', '8', '0'), fps, (W, H))
    albedo_path = os.path.join(vid_out_dir, 'output_albedo.webm')
    albedo_vid = cv2.VideoWriter(albedo_path, cv2.VideoWriter_fourcc('V', 'P', '8', '0'), fps, (W, H))

    # Initially cache first 5 frames
    start_idx = 0
    end_idx = n_total
    model_video = init_model(video_checkpoint_path)
    sp_vert = opt_sp_verts[-1][-1].detach().clone()
    img_cache = []
    ori_img_cache = []
    kpts_cache = []
    yaw_cache = []
    s_cache = []
    t_cache = []
    frame_cache = []
    frame_step = int(n_total / 5)
    for idx in [0, 1 * frame_step, 2 * frame_step, 3 * frame_step, 4 * frame_step]: #[0, 20, 260, 320, 400]: #
        in_vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = in_vid.read()
        if not ret:
            raise Exception('Read frame error: %d!' % idx)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(np.asarray(frame), (W, H), interpolation=cv2.INTER_AREA)
        img, ori_img, kpts, s, t = preprocess_frame(face_detector, frame, 1.2, 256, fa, transforms)
        img_cache.append(img)
        ori_img_cache.append(ori_img)
        kpts_cache.append(kpts)
        s_cache.append(s)
        t_cache.append(t)
        frame_cache.append(frame)
    img_cache = torch.stack(img_cache, dim=0).unsqueeze(0).cuda()
    ori_img_cache = torch.stack(ori_img_cache, dim=0).unsqueeze(0).cuda()
    kpts_cache = torch.stack(kpts_cache, dim=0).unsqueeze(0).cuda()
    pose = tracking_init(img_cache, None, model_video)

    # Initialize cached frames
    yaws = pose[0, :, 1]
    yaw_cache, sort_idx = torch.sort(yaws)
    img_cache = img_cache[:, sort_idx, ...]
    ori_img_cache = ori_img_cache[:, sort_idx, ...]
    kpts_cache = kpts_cache[:, sort_idx, ...]
    """
    # Find cache frames
    for idx in tqdm(range(start_idx, end_idx, 10)):
        if idx % 20 == 0:
            print(yaw_cache)

        in_vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = in_vid.read()
        if not ret:
            print('Read frame error: %d!' % idx)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, ori_img, kpts, s, t = preprocess_frame(face_detector, frame, 1.2, 256, fa, transforms)
        img_in = torch.cat([img_cache, img.unsqueeze(0).unsqueeze(0).cuda()], dim=1)
        ori_img_in = torch.cat([ori_img_cache, ori_img.unsqueeze(0).unsqueeze(0).cuda()], dim=1)
        kpts_in = torch.cat([kpts_cache, kpts.unsqueeze(0).unsqueeze(0).cuda()], dim=1)
        pose = tracking_init(img.unsqueeze(0).unsqueeze(0).cuda(), None, model_video)
        # Update cache
        yaw = pose[0, -1:, 1]
        j = 0
        while j < yaw_cache.shape[0] and yaw_cache[j] < yaw:
            j += 1
        yaws = torch.cat([yaw_cache[:j], yaw, yaw_cache[j:]])
        img_cache = torch.cat([img_cache[:, :j], img_in[:, -1:], img_cache[:, j:]], dim=1)
        ori_img_cache = torch.cat([ori_img_cache[:, :j], ori_img_in[:, -1:], ori_img_cache[:, j:]], dim=1)
        kpts_cache = torch.cat([kpts_cache[:, :j], kpts_in[:, -1:], kpts_cache[:, j:]], dim=1)

        yaw_diff = yaws[1:] - yaws[:-1]
        min_idx = torch.argmin(yaw_diff).item()
        if min_idx == 0:
            cache_idx = [0] + [j for j in range(2, yaws.shape[0])]
        elif min_idx == yaw_diff.shape[0] - 1:
            cache_idx = [j for j in range(0, min_idx)] + [min_idx + 1]
        else:
            left_idx = min_idx - 1
            right_idx = min_idx + 1
            if yaw_diff[left_idx] < yaw_diff[right_idx]:
                cache_idx = [j for j in range(0, min_idx)] + [j for j in range(min_idx + 1, yaws.shape[0])]
            else:
                cache_idx = [j for j in range(0, right_idx)] + [j for j in range(right_idx + 1, yaws.shape[0])]

        yaw_cache = yaws[cache_idx]
        img_cache = img_cache[:, cache_idx, ...]
        ori_img_cache = ori_img_cache[:, cache_idx, ...]
        kpts_cache = kpts_cache[:, cache_idx, ...]
    """
    # for i in range(img_cache.shape[1]):
    #     plt.imshow(ori_img_cache[0, i, ...].cpu().numpy().transpose((1, 2, 0)))
    #     plt.show()

    # Process video
    pbar = tqdm(range(start_idx, end_idx))
    for idx in pbar:
        # print(yaw_cache)

        in_vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = in_vid.read()
        if not ret:
            raise Exception('Read frame error: %d!' % idx)

        # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(np.asarray(frame), (W, H), interpolation=cv2.INTER_AREA)
        img, ori_img, kpts, s, t = preprocess_frame(face_detector, frame, 1.2, 256, fa, transforms)
        img_in = torch.cat([img_cache, img.unsqueeze(0).unsqueeze(0).cuda()], dim=1)
        ori_img_in = torch.cat([ori_img_cache, ori_img.unsqueeze(0).unsqueeze(0).cuda()], dim=1)
        kpts_in = torch.cat([kpts_cache, kpts.unsqueeze(0).unsqueeze(0).cuda()], dim=1)
        start = time.time()
        with torch.no_grad():
            _, _, _, opt_verts, _, _, _, opt_vis_masks, _, opt_vis_masks_albedo, albedo_vid_list, _, _, _, _ \
                = model_video.forward(img_in, ori_img_in, kpts_in, None, None, False)
        recon_time = time.time() - start
        pose = model_video.opt_layer.pose[:, -1:, ...].detach().clone()
        ep_norm = model_video.opt_layer.ep_norms[3][-1][:, -1:, ...].detach().clone()
        start = time.time()
        verts, albedo = rig_face(model, ep_norm, sp_vert, pose, denormalize=True)
        rig_time = time.time() - start
        pbar.set_postfix({'recon_time': recon_time, 'rig_time': rig_time})

        _, _, _, H_img, W_img = ori_img_in.shape
        colors, _, sh_coeff = model_video.opt_layer.image_reconstruction(
            ori_img_in[:, -1:], opt_verts[-1][-1][:, -1:].detach(), opt_vis_masks_albedo[-1][-1][:, -1:],
            albedo_vid_list[-1][-1][-1:].detach(), model_video.opt_layer.sh_coeff[:, -1:], H_img, W_img, 1, True)

        bfm = model_video.opt_layer.bfm
        tri = np.zeros_like(bfm.model['tri'])
        tri[:, 0] = bfm.model['tri'][:, 2]
        tri[:, 1] = bfm.model['tri'][:, 1]
        tri[:, 2] = bfm.model['tri'][:, 0]
        shading = bfm_utils.light_sh_rgb_torch(verts, tri, sh_coeff, bfm)  # (N, V, 3, nver)
        exptrans_colors = shading * albedo

        # exptrans_colors, _ = model.opt_layer.sample_per_vert_feat(ori_img_rig, opt_verts_rig[-1][-1], 256, 256)  # (N, V, 3, nver)
        # exptrans_colors = exptrans_colors[:, [3, 4], ...].mean(dim=1, keepdim=True)

        # Albedo from target images
        # albedo = model.opt_layer.compute_albedo(model.opt_layer.sh_coeff, opt_verts_rig[-1][-1],
        #                                         ori_img_rig, 256, 256)
        # albedo = albedo[:, [3, 4], ...].mean(dim=1, keepdim=True)
        # exptrans_colors = shading * albedo

        # Save to video
        vert = opt_verts[-1][-1][0, -1, ...].detach().cpu().numpy() / s - t.reshape((1, 3))
        geo_vis = visualize_geometry(vert,
                                     np.copy(frame.astype(np.float32) / 255.0),
                                     model.opt_layer.bfm.model['tri'], model.face_region_mask)
        recon_vid.write(cv2.cvtColor((geo_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        geo_vis = visualize_geometry(vert,
                                     np.copy(frame.astype(np.float32) / 255.0),
                                     model.opt_layer.bfm.model['tri'], model.face_region_mask,
                                     colors=colors[0, 0, ...].detach().cpu().numpy().transpose((1, 0)))
        colors_vid.write(cv2.cvtColor((geo_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        geo_vis = visualize_geometry(vert,
                                     np.copy(frame.astype(np.float32) / 255.0),
                                     model.opt_layer.bfm.model['tri'], model.face_region_mask,
                                     colors=albedo_vid_list[-1][-1][-1, ...].detach().cpu().numpy().transpose((1, 0)))
        albedo_vid.write(cv2.cvtColor((geo_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        vert = verts[0, -1, ...].detach().cpu().numpy() / s - t.reshape((1, 3))
        geo_vis = visualize_geometry(vert,
                                     np.zeros_like(frame.astype(np.float32)) + np.array([[[159 / 255, 160 / 255, 164 / 255]]]).astype(np.float32),
                                     model.opt_layer.bfm.model['tri'], model.face_region_mask)
        exptrans_vid.write(cv2.cvtColor((geo_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        geo_vis = visualize_geometry(vert,
                                     np.zeros_like(frame.astype(np.float32)) + np.array([[[159 / 255, 160 / 255, 164 / 255]]]).astype(np.float32),
                                     model.opt_layer.bfm.model['tri'], model.face_region_mask,
                                     colors=exptrans_colors[0, 0, ...].detach().cpu().numpy().transpose((1, 0)))
        exptrans_colors_vid.write(cv2.cvtColor((geo_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        geo_vis = visualize_geometry(vert,
                                     np.zeros_like(frame.astype(np.float32)) + np.array([[[159 / 255, 160 / 255, 164 / 255]]]).astype(np.float32),
                                     model.opt_layer.bfm.model['tri'], model.face_region_mask,
                                     colors=albedo[0, 0, ...].detach().cpu().numpy().transpose((1, 0)))
        exptrans_albedo_vid.write(cv2.cvtColor((geo_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        clip_vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Update cache
        yaw = pose[0, -1:, 1]
        j = 0
        while j < yaw_cache.shape[0] and yaw_cache[j] < yaw:
            j += 1
        yaws = torch.cat([yaw_cache[:j], yaw, yaw_cache[j:]])
        img_cache = torch.cat([img_cache[:, :j], img_in[:, -1:], img_cache[:, j:]], dim=1)
        ori_img_cache = torch.cat([ori_img_cache[:, :j], ori_img_in[:, -1:], ori_img_cache[:, j:]], dim=1)
        kpts_cache = torch.cat([kpts_cache[:, :j], kpts_in[:, -1:], kpts_cache[:, j:]], dim=1)

        yaw_diff = yaws[1:] - yaws[:-1]
        min_idx = torch.argmin(yaw_diff).item()
        if min_idx == 0:
            cache_idx = [0] + [j for j in range(2, yaws.shape[0])]
        elif min_idx == yaw_diff.shape[0] - 1:
            cache_idx = [j for j in range(0, min_idx)] + [min_idx + 1]
        else:
            left_idx = min_idx - 1
            right_idx = min_idx + 1
            if yaw_diff[left_idx] < yaw_diff[right_idx]:
                cache_idx = [j for j in range(0, min_idx)] + [j for j in range(min_idx + 1, yaws.shape[0])]
            else:
                cache_idx = [j for j in range(0, right_idx)] + [j for j in range(right_idx + 1, yaws.shape[0])]

        yaw_cache = yaws[cache_idx]
        img_cache = img_cache[:, cache_idx, ...]
        ori_img_cache = ori_img_cache[:, cache_idx, ...]
        kpts_cache = kpts_cache[:, cache_idx, ...]

        # Save per-frame mesh
        if save_video_mesh:
            vert = opt_verts[-1][-1][0, -1, ...].detach().cpu().numpy()
            color = colors[0, 0, ...].detach().cpu().numpy().transpose((1, 0)).clip(0, 1)
            vert_valid, color_valid, tri_valid = \
                bfm_utils.filter_non_tight_face_vert(vert, color, tri, model.face_region_mask)
            mesh_tri = trimesh.base.Trimesh(vertices=vert_valid, faces=tri_valid, vertex_colors=color_valid)
            mesh_path = '%06d.ply' % idx
            mesh_tri.export(os.path.join(vid_out_dir, 'mesh', mesh_path))

    in_vid.release()
    recon_vid.release()
    exptrans_vid.release()
    clip_vid.release()
    exptrans_albedo_vid.release()
    colors_vid.release()
    # cv2.destroyAllWindows()
