import os
import cv2
import numpy as np
import pywt
import random
from pathlib import Path
from scipy.ndimage import gaussian_filter, maximum_filter, gaussian_filter1d, gaussian_laplace, gaussian_gradient_magnitude
from matplotlib import pyplot as plt

# Classic Harris and Multi-Scale Harris Definitions
# 亂調
def harris_corner(
        img_gray,
        window_size=9,
        sigma=3,
        k_value=0.04,
        thresh_rel=0.001,
        max_corners=10000,
        min_dist=0
    ):
    I = img_gray.astype(np.float32)
    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    Sx2 = cv2.GaussianBlur(np.square(Ix), (window_size, window_size), sigma)
    Sy2 = cv2.GaussianBlur(np.square(Iy), (window_size, window_size), sigma)
    Sxy = cv2.GaussianBlur(Ix * Iy, (window_size, window_size), sigma)

    detM = (Sx2 * Sy2) - np.square(Sxy)
    traceM = Sx2 + Sy2
    R = detM - k_value * (traceM ** 2)

    alpha = thresh_rel  
    threshold = np.mean(R) + alpha * np.std(R)
    local_max = np.where(R <= threshold, 0, 1).astype(np.uint8)

    for i in np.arange(9):
        if i == 4:
            continue
        kernel = np.zeros(9)
        kernel[4] = 1
        kernel[i] = -1
        kernel = kernel.reshape(3, 3)
        tmp_result = np.where(cv2.filter2D(R, -1, kernel) < 0, 0, 1).astype(np.uint8)
        local_max &= tmp_result

    # 找出來的點
    locs = np.argwhere(local_max == 1)
    kps = [cv2.KeyPoint(float(x), float(y), float(window_size), -1, float(R[y, x])) for (y, x) in locs]

    if isinstance(max_corners, int) and len(kps) > max_corners:
        kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:max_corners]

    if min_dist > 0:
        kps = spatial_nms(kps, img_gray.shape, min_dist)
    print(f"[Harris Corner] Found {len(kps)} local features!")
    return kps, R


def multiscale_harris(
        img_gray,
        num_levels=4,
        scale_factor=2,
        sigma_d=1.0,           
        sigma_i=1.5,            
        k_value=0.04,
        thresh_rel=0.001,
        max_corners=10000,
        min_dist=0
    ):
    img0 = img_gray.astype(np.float32)
    pyramid = [img0]

    for _ in range(1, num_levels):
        blur = cv2.GaussianBlur(pyramid[-1], (0, 0), sigma_d)
        h, w = blur.shape
        down = cv2.resize(blur, (w // scale_factor, h // scale_factor),
                          interpolation=cv2.INTER_NEAREST)
        pyramid.append(down)

    R_levels = []     
    keypoints = []

    for lvl, I in enumerate(pyramid):  
        # 計算 Ix, Iy
        I_blur = cv2.GaussianBlur(I, (0, 0), sigma_i)
        Ix = cv2.Scharr(I_blur, cv2.CV_32F, 1, 0)  
        Iy = cv2.Scharr(I_blur, cv2.CV_32F, 0, 1)

        Sx2 = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma_i)
        Sy2 = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma_i)
        Sxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma_i)
        R   = (Sx2 * Sy2 - Sxy * Sxy) - k_value * (Sx2 + Sy2) ** 2
        R_levels.append(R)

    global_max = max(r.max() for r in R_levels)
    thresh = thresh_rel * global_max

    for lvl, R in enumerate(R_levels):
        mask = np.where(R > thresh, R, 0)
        dil = cv2.dilate(R, np.ones((3, 3), np.uint8))
        locs = np.argwhere((R == dil) & (mask > 0))

        scale = (scale_factor ** lvl)
        for (y, x) in locs:
            keypoints.append(
                cv2.KeyPoint(float(x * scale),
                             float(y * scale),
                             3.0 * scale,
                             -1,
                             float(R[y, x]))
            )

    if isinstance(max_corners, int) and len(keypoints) > max_corners:
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_corners]
    if min_dist > 0:
        keypoints = spatial_nms(keypoints, img_gray.shape, min_dist)

    h0, w0 = img_gray.shape
    R_comb = np.zeros((h0, w0), np.float32)
    for lvl, R in enumerate(R_levels):
        h, w = R.shape
        R_up = cv2.resize(R, (w0, h0), interpolation=cv2.INTER_LINEAR)
        R_comb = np.maximum(R_comb, R_up)
    print(f"[Multiscale Corner] Found {len(keypoints)} local features!")
    return keypoints, R_comb
 

# 點點最小距離函數
def spatial_nms(kps,img_shape,min_dist):
    h,w = img_shape[:2]
    grid = np.zeros((h//min_dist+1, w//min_dist+1), bool)
    kept = []
    for kp in sorted(kps, key=lambda k:-k.response):
        x, y = kp.pt
        gx, gy = int(x//min_dist), int(y//min_dist)
        if grid[max(0,gy-1):gy+2, max(0,gx-1):gx+2].any():
            continue
        kept.append(kp)
        grid[gy, gx] = True
    return kept

# Cylindrical Projection and Focal Loader
def load_focal_lengths(file_path):
    focals=[]
    for line in open(file_path):
        if line.strip() and '.jpg' not in line:
            try: focals.append(float(line))
            except: pass
    return focals


def cylindrical_projection(img,f):
    h,w = img.shape[:2]
    cx, cy = w/2.0, h/2.0
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_rel, y_rel = x-cx, y-cy
    theta = np.arctan2(x_rel, f)
    h_     = y_rel/np.sqrt(x_rel**2+f**2)
    map_x = f*theta + cx
    map_y = f*h_    + cy
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Orientation and MSOP Descriptor
def assign_orientations(gray, kps, sigma=1.0):
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma, sigmaY=sigma)
    Ix = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    for kp in kps:
        x, y = kp.pt
        ix, iy = int(round(x)), int(round(y))
        if ix<0 or ix>=gray.shape[1] or iy<0 or iy>=gray.shape[0]:
            kp.angle = 0
        else:
            ori = np.arctan2(Iy[iy,ix], Ix[iy,ix])
            kp.angle = np.degrees(ori)


def compute_msop_descriptor(gray, kp, spacing=5):
    x, y = kp.pt
    scale = kp.size / 3.0
    patch_size = 8
    # 對齊
    M = cv2.getRotationMatrix2D((x,y), kp.angle, 1.0/(spacing*scale))
    M[0,2] += patch_size/2 - x
    M[1,2] += patch_size/2 - y
    patch = cv2.warpAffine(gray, M, (patch_size,patch_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    mu, sigma = patch.mean(), patch.std()
    norm_patch = np.zeros_like(patch) if sigma<1e-7 else (patch-mu)/sigma
    # 小波變換  多頻
    LL, (LH,HL,HH) = pywt.dwt2(norm_patch, 'haar')
    desc = np.hstack([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])
    n = np.linalg.norm(desc)
    return desc/n if n>1e-7 else desc


# Detection and Description
def detect_and_describe(gray, type_of_detector='harris'):
    if type_of_detector == 'harris':
        kps, R = harris_corner(gray)
        Rs = [(R, 0)]
    else:
        kps, R = multiscale_harris(gray)
        Rs = [(R, 0)]
    assign_orientations(gray, kps)
    des = np.array([compute_msop_descriptor(gray, kp) for kp in kps], dtype=np.float32)
    return kps, des, Rs
def solve_homography(src_pts, dst_pts):
    """
    src_pts, dst_pts: shape (4, 2) 或 (N, 2)，至少 4 對點
    回傳 3×3 homography，若無法計算則回傳 None
    """
    if src_pts.shape[0] < 4:
        return None
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1,  0,  0,  0,  u*x,  u*y,  u])
        A.append([ 0,  0,  0, -x, -y, -1,  v*x,  v*y,  v])
    A = np.asarray(A, dtype=np.float64)

    # 利用 SVD 取最後一個特徵向量
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # 規一化，使 H[2,2] = 1
    if abs(H[2, 2]) < 1e-8:
        return None
    return H / H[2, 2]
# ---- Matching using Translation-only RANSAC and save the image of the matching ----
#  作對照組時改use_ransac參數如果你不要用ransac
def match_keypoints(
        des1, des2, kps1, kps2,
        ratio          = 1.5,
        use_ransac     = False,
        ransac_thresh  = 5.0,
        max_iters      = 10000,
        confidence     = 0.9,
        save_match_img = None,
        img1           = None,
        img2           = None
    ):
    # bf   = cv2.BFMatcher(cv2.NORM_L2)
    # raw  = bf.knnMatch(des1, des2, k=2)
    # # good = []
    # # for m, n in raw:
    # #     if m.distance < ratio * n.distance:
    # #         good.append(m)
    # good = [m for m, n in raw if m.distance < ratio * n.distance]
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # # 1) dire1 ratio-test
    # raw12  = bf.knnMatch(des1, des2, k=2)
    # good12 = [m for m,n in raw12 if m.distance < ratio * n.distance]

    # # 2) dire2 ratio-test
    # raw21  = bf.knnMatch(des2, des1, k=2)
    # good21 = [m for m,n in raw21 if m.distance < ratio * n.distance]

    # # 3) mutual check
    # idx21 = {(m.trainIdx, m.queryIdx) for m in good21}
    # mutual = [m for m in good12 if (m.queryIdx, m.trainIdx) in idx21]
    raw  = bf.match(des1, des2)        # 單向即可，crossCheck=True 會自動驗證雙向最佳


   # 取距離最小的前30
    good = sorted(raw, key=lambda m: m.distance)[:]

    # 5) 至少要有 4 个匹配
    if len(good) < 4:
        return None
    pts1 = np.float32([ kps1[m.queryIdx].pt for m in good ])
    pts2 = np.float32([ kps2[m.trainIdx].pt for m in good ])

    if use_ransac:
        best_inlier_idx, best_score = [], 0
        n_samples = 1
        for it in range(max_iters):
            idx = random.randrange(len(good))
            dx, dy = pts1[idx] - pts2[idx]
            diffs = pts1 - (pts2 + np.array([dx, dy]))
            d2    = np.sum(diffs**2, axis=1)
            inliers = np.where(d2 < ransac_thresh**2)[0]
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inlier_idx = inliers
                inlier_ratio = best_score / len(good)
                if inlier_ratio>0:
                    max_iters = min(
                        max_iters,
                        int(np.ceil(np.log(1 - confidence)/np.log(1 - inlier_ratio)))
                    )
            if it >= max_iters:
                break
        if best_score < 4:
            return None
        dx, dy = np.median(pts1[best_inlier_idx] - pts2[best_inlier_idx], axis=0)
        inlier_matches = [good[i] for i in best_inlier_idx]
    else:
        dx, dy = np.median(pts1 - pts2, axis=0)
        inlier_matches = good
    dx, dy = int(round(dx)), int(round(dy))

    H = np.array([[1,0,dx],[0,1,dy],[0,0,1]], dtype=np.float64)
    if save_match_img and img1 is not None and img2 is not None:
        vis = cv2.drawMatches(img1, kps1, img2, kps2, inlier_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        Path(save_match_img).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving match image: {save_match_img}")
        cv2.imwrite(str(save_match_img), vis)
    return H, inlier_matches

# Save Response , Keypoints and the gradient mapping 
def save_response_and_keypoints(
        cyl_img, gray, kps, Rs,
        output_folder, base_name):
    h, w = gray.shape
    R_comb = np.zeros((h,w), np.float32)
    for R,_ in Rs:
        R_up = cv2.resize(R, (w,h), interpolation=cv2.INTER_LINEAR)
        R_comb = np.maximum(R_comb, R_up)
    R_nonneg = np.clip(R_comb, 0, None)
    mn, mx = R_nonneg.min(), R_nonneg.max()
    R_vis = ((R_nonneg - mn)/(mx - mn + 1e-12)*255).astype(np.uint8)
    R_vis_color = cv2.applyColorMap(R_vis, cv2.COLORMAP_JET)
    kp_vis = cyl_img.copy()
    for kp in kps:
        x, y = map(int, kp.pt)
        cv2.circle(kp_vis, (x,y), 3, (0,0,255), 1)
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_response.jpg"), R_vis_color)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_keypoints.jpg"), kp_vis)
    print(f"Saved: {base_name} response & keypoints")

def blend_images(pano, warped):
    mask_p = np.sum(pano, axis=2) > 0
    mask_w = np.sum(warped, axis=2) > 0
    pano[mask_w & ~mask_p] = warped[mask_w & ~mask_p]
    overlap = mask_p & mask_w
    pano[overlap] = (
        (pano[overlap].astype(np.float32) +
         warped[overlap].astype(np.float32)) * 0.5
    ).astype(np.uint8)
    return pano

def backward_warp(src, H, dst_h, dst_w, off_x, off_y):
    h_src, w_src = src.shape[:2]

    # 在 H 上加上 canvas 偏移
    H_off = H.copy()
    H_off[0, 2] += off_x
    H_off[1, 2] += off_y
    H_inv = np.linalg.inv(H_off)

    xs, ys   = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
    x_flat   = xs.ravel()
    y_flat   = ys.ravel()
    v        = np.stack([x_flat, y_flat, np.ones_like(x_flat)])  # 3xN

    u = H_inv @ v
    u /= u[2]  

    # 越界 mask
    mask_oob = (
        (u[0] < 0)      | (u[0] >= w_src-1) |
        (u[1] < 0)      | (u[1] >= h_src-1)
    )

    v_x = np.delete(v[0], mask_oob).astype(int)
    v_y = np.delete(v[1], mask_oob).astype(int)
    u_x = np.delete(u[0], mask_oob)
    u_y = np.delete(u[1], mask_oob)

    # bilinear interpolation
    ix, iy = u_x.astype(int), u_y.astype(int)
    fx, fy = u_x - ix,      u_y - iy
    wa = (1 - fx)*(1 - fy)
    wb = (1 - fx)*fy
    wc = fx*(1 - fy)
    wd = fx*fy

    warped = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for c in range(3):
        Ia = src[iy,   ix,   c]
        Ib = src[iy+1, ix,   c]
        Ic = src[iy,   ix+1, c]
        Id = src[iy+1, ix+1, c]
        vals = wa*Ia + wb*Ib + wc*Ic + wd*Id
        warped[v_y, v_x, c] = np.clip(vals, 0, 255).astype(np.uint8)

    valid = np.zeros((dst_h, dst_w), dtype=bool)
    valid[v_y, v_x] = True

    return warped, valid

# path = os.path.join(folder, f"prtn{i:02d}.jpg")
###################

# stitch
def stitch_images(choose_example , folder, folder_for_res, pano_txt, n=18, do_end_to_end_alignment=False  , used_ransac=True):
    focals = load_focal_lengths(pano_txt)
    imgs, grays = [], []
    harris_kps, harris_des, harris_Rs = [], [], []
    multi_kps,  multi_des,  multi_Rs  = [], [], []
    Hs_harris = [np.eye(3)]
    Hs_multi  = [np.eye(3)]

    # --------- 1. 讀檔 & 描述子 ---------
    for i in range(n):
        if choose_example:
            
            path = os.path.join(folder, f"prtn{i:02d}.jpg")
            img  = cv2.imread(path)
            img = crop_by_ratio(img, axis='vertical', keep='second')
        # using our photo
        else:
            path = os.path.join(folder, f"IMG_{5558+i}.jpeg")
            img  = cv2.imread(path)
            img = crop_by_ratio(img, axis='vertical', keep='second')
        if img is None:
            print(f"Cannot read {path}")
            continue
        # example 在做cylindrical
        if choose_example:
            cyl  = cylindrical_projection(
                img, focals[i] if i < len(focals) else focals[-1]
            )
            gray = cv2.cvtColor(cyl, cv2.COLOR_BGR2GRAY)

            kh, dh, Rh = detect_and_describe(gray, 'harris')
            km, dm, Rm = detect_and_describe(gray, 'multiscale')

            imgs.append(cyl)       # 注意這裡用 cyl 而不是原圖
        # 不是別做
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kh, dh, Rh = detect_and_describe(gray, 'harris')
            km, dm, Rm = detect_and_describe(gray, 'multiscale')
            imgs.append(img)

        grays.append(gray)
        harris_kps.append(kh); harris_des.append(dh); harris_Rs.append(Rh)
        multi_kps.append(km);  multi_des.append(dm);  multi_Rs.append(Rm)

    # --------- 2. Pairwise homography & 存 Keypoints/Matches ---------
    for i in range(1, len(imgs)):
        use_ransac, ransac_thresh = used_ransac, 5.0

        # Harris pipeline
        res_h = match_keypoints(
            harris_des[i-1], harris_des[i],
            harris_kps[i-1], harris_kps[i],
            use_ransac=use_ransac,
            ransac_thresh=ransac_thresh,
            save_match_img=os.path.join(
                folder_for_res,
                f"match_{'ransac' if use_ransac else 'no_ransac'}_harris_{i-1}_{i}.jpg"
            ),
            img1=imgs[i-1], img2=imgs[i]
        )
        Hs_harris.append(Hs_harris[-1] @ (res_h[0] if res_h else np.eye(3)))

        # Multiscale pipeline
        res_m = match_keypoints(
            multi_des[i-1], multi_des[i],
            multi_kps[i-1], multi_kps[i],
            use_ransac=use_ransac,
            ransac_thresh=ransac_thresh,
            save_match_img=os.path.join(
                folder_for_res,
                f"match_{'ransac' if use_ransac else 'no_ransac'}_multiscale_{i-1}_{i}.jpg"
            ),
            img1=imgs[i-1], img2=imgs[i]
        )
        Hs_multi.append(Hs_multi[-1] @ (res_m[0] if res_m else np.eye(3)))

        # 存 response heatmap
        save_response_and_keypoints(
            imgs[i-1], grays[i-1],
            harris_kps[i-1], harris_Rs[i-1],
            folder_for_res, f"harris_prtn{i-1:02d}"
        )
        save_response_and_keypoints(
            imgs[i-1], grays[i-1],
            multi_kps[i-1],  multi_Rs[i-1],
            folder_for_res, f"multiscale_prtn{i-1:02d}"
        )

    # 最後一張 keypoints
    if imgs:
        last = len(imgs)-1
        save_response_and_keypoints(
            imgs[last], grays[last],
            harris_kps[last], harris_Rs[last],
            folder_for_res, f"harris_prtn{last:02d}"
        )
        save_response_and_keypoints(
            imgs[last], grays[last],
            multi_kps[last],  multi_Rs[last],
            folder_for_res, f"multiscale_prtn{last:02d}"
        )

        # --------- End-to-End ---------
    if do_end_to_end_alignment and len(imgs) > 1:
        for Hs, des_list, kps_list in [
            (Hs_harris, harris_des, harris_kps),
            (Hs_multi,  multi_des,   multi_kps)
        ]:
            # 最後一張 → 第一張
            res_loop = match_keypoints(
                des_list[-1], des_list[0],
                kps_list[-1], kps_list[0],
                use_ransac=True,
                ransac_thresh=5.0,
                save_match_img=None, img1=None, img2=None
            )
            if not res_loop:
                continue
            H_last2first, _ = res_loop

            # 2) 累積的最後→第一
            H_seq = Hs[-1]

            H_err = H_last2first @ np.linalg.inv(H_seq)
            # y方向誤差
            dy_err = H_err[1, 2]     
            steps  = len(imgs) - 1
            # 平均到每一張
            dy_step = dy_err / steps  

        
            for j in range(1, len(Hs)):
                Hs[j][1, 2] += dy_step * j
    # --------- End Alignment ---------



    # --------- 3. 計算 Canvas 範圍 ---------
    def compute_canvas(Hs):
        corners = []
        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            pts  = np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]]).T
            tr   = H @ pts; tr /= tr[2]
            corners.append(tr[:2].T)
        all_c = np.vstack(corners)
        return all_c[:,0].min(), all_c[:,1].min(), all_c[:,0].max(), all_c[:,1].max()

    min_x1, min_y1, max_x1, max_y1 = compute_canvas(Hs_harris)
    min_x2, min_y2, max_x2, max_y2 = compute_canvas(Hs_multi)
    min_x, min_y = min(min_x1,min_x2), min(min_y1,min_y2)
    max_x, max_y = max(max_x1,max_x2), max(max_y1,max_y2)

    pano_w, pano_h = int(np.ceil(max_x-min_x)), int(np.ceil(max_y-min_y))
    off_x, off_y   = -min_x, -min_y
    print("Wait pationtly, stitching images...")

    # def make_panorama(Hs, blur_size: int = 31):
    #     """
    # 參數
    # ----
    # Hs : list[np.ndarray]
    #     每張影像到全景座標系的 3×3 homography
    # blur_size : int, optional
    #     產生 feather 權重時的高斯核大小 (奇數，越大過渡越平滑)
    # 全域依賴
    # ----------
    # imgs      : list[np.ndarray]
    # pano_h/w  : int
    # off_x/y   : float 或 int
    # backward_warp(img, H, h, w, ox, oy) -> (warped_img, valid_mask)
    # """
    #     # 1. 累加用大畫布 (float32 以免溢位)
    #     acc   = np.zeros((pano_h, pano_w, 3), np.float32)   # 色彩累加
    #     w_acc = np.zeros((pano_h, pano_w),     np.float32)   # 權重累加

    #     for img, H in zip(imgs, Hs):
    #         warped, valid = backward_warp(img, H, pano_h, pano_w, off_x, off_y)
    #         if not valid.any():
    #             continue

    #         # 2. 建立 feather 權重 (0‒1)，邊緣漸淡
    #         mask = valid.astype(np.float32)
    #         mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    #         mask /= mask.max() + 1e-6          # 正規化

    #         # 3. 累加 (像素 × 權重) 與 權重本身
    #         acc   += warped * mask[..., None]
    #         w_acc += mask

    #     # 4. 取平均並轉回 uint8
    #     pano = acc / np.maximum(w_acc[..., None], 1e-6)
    #     pano = np.clip(pano, 0, 255).astype(np.uint8)
    #     return pano
    
    def make_panorama(Hs):
        pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
        for img, H in zip(imgs, Hs):
            warped, valid = backward_warp(img, H, pano_h, pano_w, off_x, off_y)
            if not valid.any():
                continue
            pano[valid] = warped[valid]
        return pano



    return make_panorama(Hs_harris), make_panorama(Hs_multi)

# 切圖片
def crop_by_ratio(img, axis='vertical', keep='first', ratio=0.9):
    
    if not (0 < ratio < 1):
        raise ValueError("ratio must be between 0 and 1")

    h, w = img.shape[:2]
    if axis == 'vertical':
        cut = int(w * ratio)
        if keep == 'first':
            return img[:, :cut]
        elif keep == 'second':
            return img[:, w-cut:]
        else:
            raise ValueError("keep must be 'first' or 'second'")
    elif axis == 'horizontal':
        cut = int(h * ratio)
        if keep == 'first':
            return img[:cut, :]
        elif keep == 'second':
            return img[h-cut:, :]
        else:
            raise ValueError("keep must be 'first' or 'second'")
    else:
        raise ValueError("axis must be 'vertical' or 'horizontal'")





if __name__=='__main__':

    # example
    folder = 'parrington 2'
    output_folder = 'parrington2_res'
    pano_txt = os.path.join(folder, 'pano.txt')
    os.makedirs(output_folder, exist_ok=True)
    pano , pano2 = stitch_images(True,folder, output_folder, pano_txt , do_end_to_end_alignment=True , used_ransac=True , n=18)
    cv2.imwrite(os.path.join(output_folder, 'panorama.jpg'), pano)
    cv2.imwrite(os.path.join(output_folder, 'panorama_multiscale.jpg'), pano2)
    print('Done')


    # our photo
    folder = 'target'
    output_folder = 'target_res'
    pano_txt = os.path.join(folder, 'focal.txt')
    os.makedirs(output_folder, exist_ok=True)
    # 對照組可以把do_end_to_end_alignment 還有 used_ransac 設成False/true
    pano_our , pano2_our = stitch_images(False,folder, output_folder, pano_txt , do_end_to_end_alignment=True , used_ransac=False , n=8)
    cv2.imwrite(os.path.join(output_folder, 'panorama.jpeg'), pano_our)
    cv2.imwrite(os.path.join(output_folder, 'panorama_multiscale.jpeg'), pano2_our)
    print("Done")
    
