import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt

def cylindrical_projection(img, f):
    h, w = img.shape[:2]
    cyl = np.zeros_like(img)
    center_x = w // 2
    center_y = h // 2

    for y in range(h):
        for x in range(w):
            x_ = x - center_x
            y_ = y - center_y

            theta = np.arctan(x_ / f)
            h_ = y_ / np.sqrt(x_**2 + f**2)

            x_c = f * theta + center_x
            y_c = f * h_ + center_y

            if 0 <= x_c < w and 0 <= y_c < h:
                cyl[y, x] = img[int(y_c), int(x_c)]

    return cyl

def adaptive_nms(corners, response_map, desired_kp=500, initial_radius=20, min_radius=1):
    # corners = [tuple(map(int, pt)) for pt in corners]
    print(type(response_map))
    print(response_map)
    corners = sorted(corners, key=lambda pt: response_map[pt[0], pt[1]], reverse=True)

    for r in range(initial_radius, min_radius - 1, -1):
        selected = []
        occupied = np.zeros_like(response_map, dtype=bool)
        for y, x in corners:
            if not occupied[y, x]:
                selected.append((y, x))
                y_min = max(0, y - r)
                y_max = min(response_map.shape[0], y + r + 1)
                x_min = max(0, x - r)
                x_max = min(response_map.shape[1], x + r + 1)
                occupied[y_min:y_max, x_min:x_max] = True

            if len(selected) >= desired_kp:
                return selected
    return selected

def harris_features(gray, k=0.04, window_size=3, threshold=0.01, adaptive_nums=300):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of gradients
    Ixx = gaussian_filter(Ix**2, sigma=1)
    Iyy = gaussian_filter(Iy**2, sigma=1)
    Ixy = gaussian_filter(Ix*Iy, sigma=1)

    # Harris response
    det_M = Ixx * Iyy - Ixy**2
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M**2)

    """
    # Thresholding
    R_max = np.max(R)
    corners = np.argwhere(R > threshold * R_max)
    """
    # Normalize and threshold
    R_norm = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)
    # Local maximum filtering
    R_max = maximum_filter(R, size=window_size)
    corners = (R == R_max) & (R > threshold * R.max())

    # Extract coordinates
    keypoints = np.argwhere(corners)
    selected_keypoints = adaptive_nms(keypoints, R_norm, desired_kp=adaptive_nums)
    
    return selected_keypoints, R

def sift_assign_orientation(gray, keypoints, patch_size=16, num_bins=36):
    orientations = []
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Sobel gradients
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    for y, x in keypoints:
        # Extract patch around keypoint
        x0, y0 = int(x), int(y)
        half = patch_size // 2

        if y0 - half < 0 or y0 + half >= gray.shape[0] or x0 - half < 0 or x0 + half >= gray.shape[1]:
            continue

        mag = np.sqrt(Ix[y0-half:y0+half, x0-half:x0+half]**2 +
                      Iy[y0-half:y0+half, x0-half:x0+half]**2)
        theta = np.arctan2(Iy[y0-half:y0+half, x0-half:x0+half],
                           Ix[y0-half:y0+half, x0-half:x0+half])

        # Histogram of orientations
        hist = np.zeros(num_bins, dtype=np.float32)
        for i in range(patch_size):
            for j in range(patch_size):
                angle = theta[i, j]
                magnitude = mag[i, j]
                bin = int(((angle + np.pi) / (2 * np.pi)) * num_bins) % num_bins
                hist[bin] += magnitude

        dominant_orientation = (np.argmax(hist) / num_bins) * 2 * np.pi - np.pi
        orientations.append((x0, y0, dominant_orientation))

    return orientations

def sift_describe_patch(gray, keypoints, patch_size=16, grid=4, bins=8):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    descriptors = []

    cell_size = patch_size // grid

    for x, y, angle in keypoints:
        x0, y0 = int(x), int(y)
        half = patch_size // 2

        if y0 - half < 0 or y0 + half >= gray.shape[0] or x0 - half < 0 or x0 + half >= gray.shape[1]:
            continue

        patch_Ix = Ix[y0 - half:y0 + half, x0 - half:x0 + half]
        patch_Iy = Iy[y0 - half:y0 + half, x0 - half:x0 + half]

        mag = np.sqrt(patch_Ix**2 + patch_Iy**2)
        theta = np.arctan2(patch_Iy, patch_Ix) - angle  # rotate to align

        # 128-dimensional descriptor
        descriptor = []

        for i in range(grid):
            for j in range(grid):
                hist = np.zeros(bins)
                for dy in range(cell_size):
                    for dx in range(cell_size):
                        y_idx = i * cell_size + dy
                        x_idx = j * cell_size + dx

                        angle_bin = int(((theta[y_idx, x_idx] + np.pi) / (2 * np.pi)) * bins) % bins
                        hist[angle_bin] += mag[y_idx, x_idx]
                descriptor.extend(hist)

        descriptor = np.array(descriptor)
        descriptor = descriptor / np.linalg.norm(descriptor)  # normalize
        descriptors.append(descriptor)

    return np.array(descriptors)

def show_keypoints_with_orientations(image, keypoints, scale=8):
    img_show = image.copy()
    for x, y, angle in keypoints:
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + scale * np.cos(angle))
        y2 = int(y + scale * np.sin(angle))
        cv2.arrowedLine(img_show, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints with Orientation")
    plt.axis("off")
    plt.show()

def show_descriptor_vector(descriptor, index=0):
    desc = descriptor[index]
    plt.figure(figsize=(10, 2))
    plt.bar(np.arange(len(desc)), desc)
    plt.title(f"Descriptor Vector #{index}")
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def draw_harris_points(img, points):
    for y, x in points:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    return img

def load_focal_lengths(file_path):
    output = list()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.strip() != '':
                output.append(float(line.strip()))
    return output

# Visualize result
def plot_keypoints(img, points):
    import matplotlib.pyplot as plt
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    ys, xs = zip(*points)
    plt.scatter(xs, ys, s=10, c='red')
    plt.title("Keypoints after Adaptive NMS")
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    image_dir = "test1"
    focal_file = "test1/pano.txt"

    focals = load_focal_lengths(focal_file)
    image_files = sorted(list(filter(lambda x: '.jpg' in x, os.listdir(image_dir))))

    for idx, image_name in enumerate(image_files):
        if idx == 10:
            img = cv2.imread(os.path.join(image_dir, image_name))
            f = focals[idx]
            print(os.path.join(image_dir, image_name), f)
            cyl_img = cylindrical_projection(img, f)
            gray_image = cv2.cvtColor(cyl_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            keypoints, R_map = harris_features(gray_image)
            """
            out = draw_harris_points(cyl_img.copy(), keypoints)
            cv2.imshow("Harris Features", out)
            cv2.waitKey(0)
            """
            """
            show_keypoints_with_orientations(img, orientation)
            orientations = assign_orientation(img, keypoints)
            descriptor = describe_patch(img, orientations)
            show_descriptor_vector(descriptor)
            """
            # selected_keypoints = adaptive_nms(keypoints, R_map, desired_kp=300)
            plot_keypoints(img, keypoints)



