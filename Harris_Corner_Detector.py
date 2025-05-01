import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt

# Ensure results directory exists
# and global prefix for filenames

dir_res = "res2"
current_prefix = ""
os.makedirs(dir_res, exist_ok=True)


def save_figure(fig_name):
    # Uses current_prefix to avoid overwriting
    global current_prefix
    path = os.path.join(dir_res, f"{current_prefix}{fig_name}")
    plt.savefig(path)
    plt.close()


def cylindrical_projection(img, f):
    h, w = img.shape[:2]
    cyl = np.zeros_like(img)
    cx, cy = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            x_, y_ = x - cx, y - cy
            theta = np.arctan(x_ / f)
            h_ = y_ / np.sqrt(x_**2 + f**2)
            x_c = f * theta + cx
            y_c = f * h_ + cy
            if 0 <= x_c < w and 0 <= y_c < h:
                cyl[y, x] = img[int(y_c), int(x_c)]
    return cyl


def adaptive_nms(corners, response_map, desired_kp=500, initial_radius=20, min_radius=1):
    corners = sorted(corners, key=lambda pt: response_map[pt[0], pt[1]], reverse=True)
    for r in range(initial_radius, min_radius - 1, -1):
        selected, occupied = [], np.zeros_like(response_map, dtype=bool)
        for y, x in corners:
            if not occupied[y, x]:
                selected.append((y, x))
                y0, y1 = max(0, y - r), min(response_map.shape[0], y + r + 1)
                x0, x1 = max(0, x - r), min(response_map.shape[1], x + r + 1)
                occupied[y0:y1, x0:x1] = True
            if len(selected) >= desired_kp:
                return selected
    return selected


def harris_features(gray, k=0.04, window_size=3, threshold=0.01, adaptive_nums=300):
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    Ixx = gaussian_filter(Ix**2, sigma=1)
    Iyy = gaussian_filter(Iy**2, sigma=1)
    Ixy = gaussian_filter(Ix*Iy, sigma=1)
    detM = Ixx * Iyy - Ixy**2
    traceM = Ixx + Iyy
    R = detM - k * (traceM**2)
    R_norm = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)
    Rmax = maximum_filter(R, size=window_size)
    corners = (R == Rmax) & (R > threshold * R.max())
    kps = np.argwhere(corners)
    selected = adaptive_nms(kps, R_norm, desired_kp=adaptive_nums)
    return selected, R


def sift_assign_orientation(gray, keypoints, patch_size=16, num_bins=36):
    orientations = []
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    for y, x in keypoints:
        x0, y0 = int(x), int(y)
        half = patch_size // 2
        if y0-half<0 or y0+half>=gray.shape[0] or x0-half<0 or x0+half>=gray.shape[1]: continue
        mag = np.hypot(Ix[y0-half:y0+half, x0-half:x0+half], Iy[y0-half:y0+half, x0-half:x0+half])
        theta = np.arctan2(Iy[y0-half:y0+half, x0-half:x0+half], Ix[y0-half:y0+half, x0-half:x0+half])
        hist = np.zeros(num_bins, dtype=np.float32)
        for i in range(patch_size):
            for j in range(patch_size):
                bin_idx = int(((theta[i,j]+np.pi)/(2*np.pi))*num_bins)%num_bins
                hist[bin_idx] += mag[i,j]
        ori = (np.argmax(hist)/num_bins)*2*np.pi - np.pi
        orientations.append((x0, y0, ori))
    return orientations


def sift_describe_patch(gray, orientations, patch_size=16, grid=4, bins=8):
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    descs, cell = [], patch_size//grid
    for x, y, ang in orientations:
        x0, y0 = int(x), int(y); half=patch_size//2
        if y0-half<0 or y0+half>=gray.shape[0] or x0-half<0 or x0+half>=gray.shape[1]: continue
        patchIx = Ix[y0-half:y0+half, x0-half:x0+half]
        patchIy = Iy[y0-half:y0+half, x0-half:x0+half]
        mag = np.hypot(patchIx, patchIy)
        theta = np.arctan2(patchIy, patchIx) - ang
        vec=[]
        for i in range(grid):
            for j in range(grid):
                hist = np.zeros(bins)
                for dy in range(cell):
                    for dx in range(cell):
                        idx = i*cell+dy; jdx = j*cell+dx
                        b = int(((theta[idx,jdx]+np.pi)/(2*np.pi))*bins)%bins
                        hist[b] += mag[idx,jdx]
                vec.extend(hist)
        v = np.array(vec); descs.append(v/np.linalg.norm(v))
    return np.array(descs)


def plot_keypoints(img, points):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ys,xs=zip(*points)
    plt.scatter(xs, ys, s=10, c='red'); plt.axis('off'); plt.title('Keypoints')
    save_figure('keypoints.png')


def show_keypoints_with_orientations(img, orientations, scale=8):
    vis=img.copy()
    for x,y,ang in orientations:
        x1,y1=int(x),int(y)
        x2,y2=int(x+scale*np.cos(ang)),int(y+scale*np.sin(ang))
        cv2.arrowedLine(vis,(x1,y1),(x2,y2),(0,255,0),1,tipLength=0.3)
    plt.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.title('Oriented Keypoints')
    save_figure('oriented_keypoints.png')


def show_descriptor_vector(descs, index=0):
    v=descs[index]
    plt.figure(figsize=(10,2)); plt.bar(range(len(v)),v); plt.axis('off')
    save_figure(f'descriptor_{index}.png')


def visualize_keypoint_patches(gray, points, num=2, step=2):
    Ix=np.gradient(gray,axis=1); Iy=np.gradient(gray,axis=0)
    for i,(y,x) in enumerate(points[:num]):
        if y<8 or x<8 or y+8>=gray.shape[0] or x+8>=gray.shape[1]: continue
        p=gray[y-8:y+8,x-8:x+8]; px,py=Ix[y-8:y+8,x-8:x+8][::step,::step],Iy[y-8:y+8,x-8:x+8][::step,::step]
        h,w=px.shape; X,Y=np.meshgrid(range(w),range(h))
        plt.imshow(p,cmap='gray'); plt.quiver(X,Y,px,-py,scale=50); plt.axis('off')
        save_figure(f'patch_{i}.png')


def load_focal_lengths(fp):
    f=[]
    for line in open(fp):
        try: f.append(float(line.strip()))
        except: pass
    return f


if __name__=='__main__':
    image_dir = 'parrington 2'
    focal_file = os.path.join(image_dir, 'pano.txt')
    focals = load_focal_lengths(focal_file)
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    # Ensure matching counts
    num = min(len(files), len(focals))
    if len(files) != len(focals):
        print(f"Warning: {len(files)} images but {len(focals)} focal lengths; processing first {num} items.")

    for idx in range(num):

        current_prefix = f"img{idx}_"
        name = files[idx]
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        f = focals[idx]

        # Cylindrical projection and grayscale
        cyl = cylindrical_projection(img, f)
        gray = cv2.cvtColor(cyl, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Harris features and response map
        kps, R = harris_features(gray)
        plt.imshow(R, cmap='jet'); plt.axis('off'); save_figure('response_map.png')

        # Plot keypoints
        plot_keypoints(cyl, kps)

        # Orientation assignment and plot
        ors = sift_assign_orientation(gray, kps)
        show_keypoints_with_orientations(cyl, ors)

        # Descriptor and plot
        descs = sift_describe_patch(gray, ors)
        if descs.size:
            show_descriptor_vector(descs, 0)

        # Visualize patches
        visualize_keypoint_patches(gray, kps)
    print("done , all file save in res2")
