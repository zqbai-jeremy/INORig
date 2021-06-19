import numpy as np
import torch
import torchvision
import os
from PIL import Image
import cv2
import trimesh
import matplotlib.pyplot as plt

# External libs
from external.face3d.face3d import mesh

# Internal libs
import data.BFM.utils as bfm_utils


# Visualization --------------------------------------------------------------------------------------------------------
def visualize_geometry(vert, back_ground, tri, face_region_mask=None, gt_flag=False, colors=None):
    """
    Visualize untextured mesh
    :param vert: mesh vertices. np.array: (nver, 3)
    :param back_ground: back ground image. np.array: (256, 256, 3)
    :param tri: mesh triangles. np.array: (ntri, 3) int32
    :param face_region_mask: mask for valid vertices. np.array: (nver, 1) bool
    :param gt_flag: Whether render with ESRC ground truth mesh. The normals of BFM (predicted mesh) point to the
                    opposite direction, thus need to multiply by -1.
    :return: image_t: rendered image. np.array: (3, 256, 256)
    """
    if gt_flag:
        sh_coeff = np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    else:
        sh_coeff = np.array((0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    if colors is None:
        colors = np.ones((vert.shape[0], 3), dtype=np.float) - 0.25
    else:
        sh_coeff = np.array((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    colors = mesh.light.add_light_sh(vert, tri, colors, sh_coeff)
    projected_vertices = vert.copy()  # using stantard camera & orth projection
    h, w, c = back_ground.shape
    image_vert = mesh.transform.to_image(projected_vertices, h, w)
    if face_region_mask is not None:
        image_vert, colors, tri = bfm_utils.filter_non_tight_face_vert(image_vert, colors, tri, face_region_mask)
    image_t = mesh.render.render_colors(image_vert, tri, colors, h, w, BG=back_ground)
    image_t = np.minimum(np.maximum(image_t, 0), 1)#.transpose((2, 0, 1))
    return image_t


def visualize(opt_verts, opt_verts_obj, albedo_list, colors_list, sh_coeffs, imgs, ss, ts, face_region_mask, bfm, V_out):
    vis_list = []
    N, V, nver, _ = opt_verts[-1][-1].shape
    V_out = V if V_out is None else V_out
    assert N == 1
    tri = np.zeros_like(bfm.model['tri'])
    tri[:, 0] = bfm.model['tri'][:, 2]
    tri[:, 1] = bfm.model['tri'][:, 1]
    tri[:, 2] = bfm.model['tri'][:, 0]
    for i in range(V_out):
        vert = opt_verts[-1][-1][0, i, ...].detach().cpu().numpy()
        albedo = albedo_list[-1][-1][i, ...].detach().cpu().numpy().transpose((1, 0))
        colors = colors_list[-1][-1][0, i, ...].detach().cpu().numpy().transpose((1, 0))
        sh_coeff = sh_coeffs[0, i, ...].detach().cpu().numpy().reshape(27, 1)
        cur_img_np = imgs[i].astype(np.float32) / 255.
        s = ss[i]
        t = ts[i]
        vert = vert / s - t.reshape((1, 3))
        # cur_img_np = np.ascontiguousarray(cur_img.numpy().transpose((1, 2, 0)))
        geo_vis = visualize_geometry(vert, np.copy(cur_img_np), tri, face_region_mask, gt_flag=True)
        albedo_vis = visualize_geometry(vert, np.copy(cur_img_np), tri, face_region_mask, gt_flag=True, colors=albedo)
        colors_vis = visualize_geometry(vert, np.copy(cur_img_np), tri, face_region_mask, gt_flag=True, colors=colors)
        light_colors = bfm_utils.add_light_sh_rgb(vert, tri, np.ones((vert.shape[0], 3), dtype=np.float) * 0.75, sh_coeff)
        # maxn = np.amax(light_colors[face_region_mask.ravel(), :])
        # minn = np.amin(light_colors[face_region_mask.ravel(), :])
        # light_colors = (light_colors - minn) / (maxn - minn)
        light_vis = visualize_geometry(vert, np.copy(cur_img_np), tri, face_region_mask, gt_flag=True, colors=light_colors)
        vert = opt_verts_obj[-1][-1][0, i, ...].detach().cpu().numpy() * 1.5e-3
        obj_geo_vis = visualize_geometry(vert, np.ones_like(cur_img_np), tri, face_region_mask, gt_flag=True)
        vis_list.append((cur_img_np[50:-50, 50:-50, :], colors_vis[50:-50, 50:-50, :], geo_vis[50:-50, 50:-50, :],
                         albedo_vis[50:-50, 50:-50, :], light_vis[50:-50, 50:-50, :], obj_geo_vis[50:-50, 50:-50, :]))
    return vis_list


# IO -------------------------------------------------------------------------------------------------------------------
def convert_to_output_formate(opt_verts, albedo_list, bfm, model, V_out):
    # Crop valid face region
    tri = np.zeros_like(bfm.model['tri'])
    tri[:, 0] = bfm.model['tri'][:, 2]
    tri[:, 1] = bfm.model['tri'][:, 1]
    tri[:, 2] = bfm.model['tri'][:, 0]

    N, V, nver, _ = opt_verts[-1][-1].shape
    V_out = V if V_out is None else V_out
    face_full = []
    face_valid = []
    for i in range(N):
        for j in range(V_out):
            vert = opt_verts[-1][-1][i, j, :, :].detach().cpu().numpy()
            albedo = albedo_list[-1][-1][i * V + j, ...].detach().cpu().numpy().transpose((1, 0))
            face_full.append((vert, tri, albedo))
            vert_valid, albedo_valid, tri_valid = \
                bfm_utils.filter_non_tight_face_vert(vert, albedo, tri, model.face_region_mask)
            face_valid.append((vert_valid, tri_valid, albedo_valid))

    # Return predicted full mesh (BFM topology) and cropped valid mesh
    # Results are in normalized image space
    # (x-axis to right, y-axis to up, right hand coord, camera center at z-axis facing -z)
    # (can be directly moved to image space by just adding a 2D translation)
    return face_full, face_valid


def save_outputs(out_dir, face_full, face_valid, vis_list=None, file_names=('face_full', 'face_valid'),
                 use_albedo=True, save_mesh=True):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_mesh_dir = os.path.join(out_dir, 'mesh')
    if not os.path.exists(out_mesh_dir):
        os.mkdir(out_mesh_dir)
    out_vis_dir = os.path.join(out_dir, 'visualization')
    if not os.path.exists(out_vis_dir):
        os.mkdir(out_vis_dir)

    V = len(face_full)
    if save_mesh:
        for i in range(V):
            vert, tri, albedo = face_full[i]
            if use_albedo:
                mesh = trimesh.base.Trimesh(vertices=vert, faces=tri, vertex_colors=albedo)
            else:
                mesh = trimesh.base.Trimesh(vertices=vert, faces=tri)
            mesh_path = 'view%d_%s.ply' % (i, file_names[0])
            mesh.export(os.path.join(out_mesh_dir, mesh_path))

        for i in range(V):
            vert, tri, albedo = face_valid[i]
            if use_albedo:
                mesh = trimesh.base.Trimesh(vertices=vert, faces=tri, vertex_colors=albedo)
            else:
                mesh = trimesh.base.Trimesh(vertices=vert, faces=tri)
            mesh_path = 'view%d_%s.ply' % (i, file_names[1])
            mesh.export(os.path.join(out_mesh_dir, mesh_path))

    if vis_list is not None:
        output_names = ('input', 'recon', 'geo', 'alb', 'lit', 'geo_nopose')
        for i in range(V):
            for j in range(len(output_names)):
                vis = vis_list[i][j]
                vis_path = 'view%d_%s.jpg' % (i, output_names[j])
                plt.imsave(os.path.join(out_vis_dir, vis_path), vis)


# Image / Frame Preprocess ---------------------------------------------------------------------------------------------
def load_img_2_tensors(image_path, fa, face_detector, transform_func=None):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_img = cv2.copyMakeBorder(
        img,
        top=50,
        bottom=50,
        left=50,
        right=50,
        borderType=cv2.BORDER_DEFAULT
    )
    img, s, t = get_square_face_image(face_detector, ori_img, 1.2, 256)
    assert img.shape[0] == img.shape[1] == 256
    ori_img_tensor = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32) / 255.0)  # (C, H, W)
    img_tensor = ori_img_tensor.clone()
    if transform_func:
        img_tensor = transform_func(img_tensor)

    # Get 2D landmarks on image
    kpts_list = fa.get_landmarks(img)
    kpts = kpts_list[0]
    kpts_tensor = torch.from_numpy(kpts)                                                    # (68, 2)

    return img_tensor, ori_img_tensor, kpts_tensor, ori_img, s, t


def preprocess(img_dir, fa, face_detector):
    """
    Propare data for inferencing.
    img_dir: directory of input images. str.
    fa: face alignment model. From https://github.com/1adrianb/face-alignment
    face_detector: face detector model. From https://github.com/1adrianb/face-alignment
    """
    transform_func = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    img_list = sorted(os.listdir(img_dir))

    img_tensors = []
    ori_img_tensors = []
    kpts_tensors = []
    ori_imgs = []
    ss = []
    ts = []
    for image_name in img_list:
        if '.jpg' not in image_name:
            continue
        image_path = os.path.join(img_dir, image_name)
        img_tensor, ori_img_tensor, kpts_tensor, ori_img, s, t = \
            load_img_2_tensors(image_path, fa, face_detector, transform_func)
        img_tensors.append(img_tensor)
        ori_img_tensors.append(ori_img_tensor)
        kpts_tensors.append(kpts_tensor)
        ori_imgs.append(ori_img)
        ss.append(s)
        ts.append(t)
    img_tensors = torch.stack(img_tensors, dim=0).unsqueeze(0)                  # (1, V, C, H, W)
    ori_img_tensors = torch.stack(ori_img_tensors, dim=0).unsqueeze(0)          # (1, V, C, H, W)
    kpts_tensors = torch.stack(kpts_tensors, dim=0).unsqueeze(0)                # (1, V, 68, 2)

    return img_tensors.cuda(), ori_img_tensors.cuda(), kpts_tensors.cuda(), ori_imgs, ss, ts


def get_square_face_image(face_detector, img, scale, size):
    """
    Crop a square face region based on face_detector (SFD in https://github.com/1adrianb/face-alignment)
    :param face_detector: SFD face detector
    :param img: raw RGB image read by opencv
    :param scale: side length of the square = scale * max(detected bounding box width, detected bounding box height)
    :param size: resize to this size, (size, size)
    :return: square face image, pixel value in [0, 1], np.array(H, W, 3)
    """
    d = face_detector.detect_from_image(img[..., ::-1].copy())
    idx = 0
    if len(d) > 1:
        for i, face in enumerate(d):
            if face[-1] > d[idx][-1]:
                idx = i
    d = d[idx]
    center = [d[3] - (d[3] - d[1]) / 2.0, d[2] - (d[2] - d[0]) / 2.0]
    center[0] += (d[3] - d[1]) * 0.06
    l = max(d[2] - d[0], d[3] - d[1]) * scale
    x_s = int(center[1] - (l / 2) + 0.5)
    y_s = int(center[0] - (l / 2) + 0.5)
    x_e = int(center[1] + (l / 2) + 0.5)
    y_e = int(center[0] + (l / 2) + 0.5)
    t = [img.shape[1] / 2. - center[1], center[0] - img.shape[0] / 2., 0]
    s = size / (x_e - x_s)
    img = Image.fromarray(img).crop((x_s, y_s, x_e, y_e))
    img = cv2.resize(np.asarray(img), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return img, s, np.array(t)


def preprocess_frame(face_detector, ori_img, scale, size, fa, transform_func):
    # Crop img
    img, s, t = get_square_face_image(face_detector, ori_img, scale, size)
    assert img.shape[0] == img.shape[1] == size
    ori_img_tensor = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32) / 255.0)  # (C, H, W)
    img_tensor = ori_img_tensor.clone()
    if transform_func:
        img_tensor = transform_func(img_tensor)

    # Get 2D landmarks on image
    kpts_list = fa.get_landmarks(ori_img)
    kpts = kpts_list[0]
    ori_center = np.asarray([ori_img.shape[1], ori_img.shape[0]]).reshape((1, 2)) / 2
    center = np.asarray([ori_center[0, 0] - t[0], ori_center[0, 1] + t[1]]).reshape((1, 2))
    kpts = (kpts - center) * s + 128
    kpts_tensor = torch.from_numpy(kpts).float()  # (68, 2)
    # plt.imshow(img / 255.)
    # plt.scatter(kpts[:, 0], kpts[:, 1], s=50)
    # plt.show()

    return img_tensor, ori_img_tensor, kpts_tensor, s, t


# Inference ------------------------------------------------------------------------------------------------------------
def predict(model, img, ori_img, kpts):
    # Network forward
    with torch.no_grad():
        pose, sp_norm, ep_norm, \
        opt_verts, opt_verts_obj, opt_sp_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, _, albedo_list, \
        colors_list, raw_step_sizes, adap_B_uv_list, delta_vert_list = \
            model.forward(img, ori_img, kpts, None, None, False)

    return opt_verts, opt_verts_obj, opt_sp_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, albedo_list, \
        colors_list, raw_step_sizes, adap_B_uv_list, delta_vert_list


def rig_face(model, ep_norm, sp_vert, pose, denormalize=False):
    N, V, _, _ = ep_norm.shape

    # Process params
    pose = pose.view(N * V, 6)
    pitch, yaw, roll, s, tx, ty = \
        pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3], pose[:, 4], pose[:, 5]      # in degree
    if denormalize:
        pitch, yaw, roll, s, tx, ty = \
            bfm_utils.denormalize_pose_params(pitch, yaw, roll, s, tx, ty)

    # Process vertices
    nver = int(model.opt_layer.bfm.nver)
    ep_vert = model.opt_layer.exp_model(ep_norm, None, sp_vert[:, 0, :, :], 2, False)        # (N, V, nver, 3)
    vert = sp_vert + ep_vert

    # Transform by pose
    vert = vert.view(N * V, nver, 3)
    # vert_obj = vert.clone()
    angles = torch.stack([pitch, yaw, roll], dim=1)
    zeros = torch.zeros_like(tx)
    t = torch.stack([tx, ty, zeros], dim=1)
    vert = model.opt_layer.bfm_torch.transform(vert, s, angles, t)              # (N * V, nver, 3)

    # Dynamic Albedo
    albedo = model.opt_layer.albedo_model(None, ep_norm, ep_vert, 2, False)              # (N * V, 3, nver)

    return vert.view(N, V, nver, 3), albedo.view(N, V, 3, nver)


def tracking_init(img, kpts, model, denormalize=False):
    N, V, _, _, _ = img.shape
    model.regressor.eval()
    with torch.no_grad():
        pose = model.regressor.forward(img).view(N * V, 6)
    if denormalize:
        pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm = \
            pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3], pose[:, 4], pose[:, 5]
        pitch, yaw, roll, s, tx, ty = \
            bfm_utils.denormalize_pose_params(pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm)
        return torch.stack([pitch, yaw, roll, s, tx, ty], dim=1).view(N, V, 6)
    else:
        return pose.view(N, V, 6)
