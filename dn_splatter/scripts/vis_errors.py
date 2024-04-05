"""Visualize the L2 loss between two images"""
import os
import shutil

import cv2
import numpy as np
from matplotlib import pyplot as plt


def error_level_curve(error_img, percentile=0.5):
    total_error = np.sum(error_img)
    threshold = percentile * total_error

    error_lin = np.ravel(error_img)
    indices = np.argsort(-error_lin)
    error_sorted = error_lin[indices]
    error_cumsum = np.cumsum(error_sorted)
    threshold_index = np.searchsorted(error_cumsum, threshold)

    mask_lin = np.zeros(error_lin.shape, dtype=bool)
    mask_lin[indices[:threshold_index]] = 1

    return np.reshape(mask_lin, error_img.shape)


def match_sizes(img1, img2):
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
        print("Warning: images have different sizes, cropping to smallest")

    img1_cropped = img1[:min_height, :min_width]
    img2_cropped = img2[:min_height, :min_width]
    return img1_cropped, img2_cropped


def multi_error_curve(error_img):
    err_90 = error_level_curve(error_img, percentile=0.9)
    err_50 = error_level_curve(error_img, percentile=0.3)

    img = np.ones(err_90.shape + (3,), dtype=np.uint8) * 255
    img[err_90, :] = [255, 255, 0]
    img[err_50, :] = [255, 0, 0]
    return img


def remove_common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            break
    else:
        i += 1
    return s1[i:], s2[i:]


def find_largest_error_area(error_image, window_size, rel_min_edge_dist=0.15):
    import numpy as np

    rows, cols = error_image.shape
    wr, wc = window_size

    edge_x = int(rel_min_edge_dist * cols)
    edge_y = int(rel_min_edge_dist * rows)

    max_error = 0
    max_position = (0, 0)

    for r in range(edge_y, rows - edge_y - wr):
        for c in range(edge_x, cols - edge_x - wc):
            # Sum the absolute errors within the current window
            current_sum = np.sum(error_image[r : r + wr, c : c + wc])

            # Update the maximum error and position if the current sum is larger
            if current_sum > max_error:
                max_error = current_sum
                max_position = (r, c)

    # Return the position of the window with the largest error sum
    return max_position


def add_zoomed_in_area_in_place(image, error_area, window_size, zoom_rel_size=0.45):
    h, w, c = image.shape
    ay, ax = error_area
    ah, aw = window_size
    zoomed_h_target = int(h * zoom_rel_size)
    zoom_factor = round(zoomed_h_target / ah)

    zoomed_h = zoom_factor * ah
    zoomed_w = zoom_factor * aw

    area = image[ay : ay + ah, ax : ax + aw, ...]
    zoomed = cv2.resize(area, (zoomed_w, zoomed_h), interpolation=cv2.INTER_NEAREST)

    border_color = (0, 0, 255)
    cv2.rectangle(image, (ax, ay), (ax + aw, ay + ah), border_color, 2)

    zoomed_x0 = 0
    zoomed_y0 = h - zoomed_h
    zoomed_x1 = zoomed_x0 + zoomed_w
    zoomed_y1 = zoomed_y0 + zoomed_h
    cv2.rectangle(
        image, (zoomed_x0, zoomed_y0), (zoomed_x1, zoomed_y1), border_color, 4
    )
    image[zoomed_y0:zoomed_y1, zoomed_x0:zoomed_x1, ...] = zoomed


def compute_error_images(gt_img, other_images):
    gt_image = cv2.imread(gt_img)

    result = {
        "name": os.path.basename(gt_img).rpartition(".")[0],
        "gt": gt_image,
        "variants": [],
    }

    prev_error = None

    img_size = gt_image.shape[:2]

    for i, f2 in enumerate(other_images):
        image2 = cv2.imread(f2)
        image1, image2 = match_sizes(gt_image, image2)
        img_size = image1.shape[:2]

        l2_err = np.sum((image1.astype(float) - image2.astype(float)) ** 2, axis=-1)

        mse = np.mean(l2_err) / 3
        psnr = 20 * np.log10(255) - 10 * np.log10(mse)

        variant = {
            "path": f2,
            "image": image2,
            "l2": l2_err,
            "contributions": multi_error_curve(l2_err),
            "psnr": psnr,
        }

        if prev_error is not None:
            l2_err, prev_error = match_sizes(l2_err, prev_error)
            err_diff = l2_err - prev_error
            variant["l2_diff"] = err_diff
            img_size = err_diff.shape[:2]

        prev_error = l2_err

        result["variants"].append(variant)

    # TODO: hacky: should find out why the image sizes don't match in the first place
    result["gt"] = result["gt"][: img_size[0], : img_size[1], ...]
    for variant in result["variants"]:
        for k, v in variant.items():
            if isinstance(v, np.ndarray):
                variant[k] = v[: img_size[0], : img_size[1], ...]

    return result


def visualize(gt_img, other_images):
    error_images = compute_error_images(gt_img, other_images)

    n_rows = max(len(other_images), 2)
    n_cols = 2
    if len(other_images) > 1:
        n_cols = 4
    plt.figure(figsize=(6 * n_cols, 6 * n_rows))

    # Display the first image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(cv2.cvtColor(error_images["gt"], cv2.COLOR_BGR2RGB))
    plt.title("GT")
    plt.axis("off")  # Hide the axis

    for i, v in enumerate(error_images["variants"]):
        if i > 0:
            folder_tag, _ = remove_common_prefix(
                os.path.abspath(v["path"]), os.path.abspath(other_images[0])
            )
            folder_tag = folder_tag.split(os.path.sep)[0] + ":"
        else:
            folder_tag = ""

        if len(other_images) > 1:
            row_offs = i * n_cols + 1
        else:
            row_offs = 1

        plt.subplot(n_rows, n_cols, row_offs + 1)
        plt.imshow(cv2.cvtColor(v["image"], cv2.COLOR_BGR2RGB))
        plt.title(folder_tag + os.path.basename(v["path"]))
        plt.axis("off")

        plt.subplot(n_rows, n_cols, row_offs + 2)
        magn = 3 * 255**2
        plt.imshow(v["l2"], cmap="jet", vmin=0, vmax=magn)
        # plt.colorbar()  # Add a colorbar to interpret the error magnitudes
        plt.title("L2 Error (PSNR = %g)" % v["psnr"])
        plt.axis("off")

        plt.subplot(n_rows, n_cols, row_offs + 3)
        plt.imshow(v["contributions"])
        # plt.spy(error_level_curve(l2_err, percentile=0.9))
        # plt.colorbar()  # Add a colorbar to interpret the error magnitudes
        plt.title("Error contribution areas 30% / 90%")
        plt.axis("off")

        if "l2_diff" in v:
            plt.subplot(n_rows, n_cols, row_offs)
            m = magn * 0.1
            plt.imshow(v["l2_diff"], cmap="jet", vmin=-m, vmax=m)
            plt.title("Error diff")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def apply_colormap(img, vmin, vmax):
    img = (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


def save(gt_img, other_images, output_folder, photo_format="png"):
    error_images = compute_error_images(gt_img, other_images)

    name = error_images["name"]
    name_and_path = os.path.join(output_folder, name)
    # print('writing to ' + name_and_path)
    cv2.imwrite(name_and_path + "_0_gt." + photo_format, error_images["gt"])

    compilation_image = []

    n_variants = len(error_images["variants"])
    for i, v in enumerate(error_images["variants"]):
        v_name = name_and_path + "_%d" % (i + 1)
        err_name = name_and_path + "_err_%d" % (i + 1)
        cv2.imwrite(v_name + "_pred." + photo_format, v["image"])
        l2_contrib = cv2.cvtColor(v["contributions"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(err_name + "_l2_contributions.png", l2_contrib)
        compilation_image.append(v["image"])

        if i + 1 == n_variants:
            compilation_image.append(l2_contrib)
        error_target = v["l2"]

        if "l2_diff" in v:
            magn = 3 * 255**2
            m = magn * 0.1
            err_diff_image = apply_colormap(v["l2_diff"], vmin=-m, vmax=m)
            cv2.imwrite(err_name + ("_vs_%d" % i) + "_l2_diff.png", err_diff_image)
            compilation_image.append(err_diff_image)
            error_target = -v["l2_diff"]

    error_window_size = (48, 64)
    error_area = find_largest_error_area(error_target, error_window_size)

    compilation_image.insert(-n_variants, error_images["gt"])
    for img in compilation_image[:-n_variants]:
        add_zoomed_in_area_in_place(img, error_area, error_window_size)
    compilation_image = np.hstack(compilation_image)
    cv2.imwrite(
        os.path.join(output_folder, "hstack_" + name + "." + photo_format),
        compilation_image,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("input_folder", type=str, default="renders", nargs="?")
    parser.add_argument("other_folders", type=str, nargs="*")
    parser.add_argument("--auto_baseline", action="store_true")
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default=None,
        help="Store images into this folder",
    )
    parser.add_argument("-f", "--photo-format", type=str, default="png")
    args = parser.parse_args()

    all_folders = [args.input_folder] + args.other_folders
    if args.auto_baseline:
        rec_path = os.path.abspath(os.path.join(args.input_folder, "..", "..", ".."))
        rec_name = os.path.basename(rec_path)
        baseline_rec_path = os.path.join(
            rec_path, "..", "..", "baseline", rec_name, "splatfacto"
        )
        print(baseline_rec_path)

        if os.path.exists(baseline_rec_path):
            all_folders.insert(
                0,
                os.path.join(
                    baseline_rec_path, os.listdir(baseline_rec_path)[0], "renders"
                ),
            )
        else:
            print("No baseline %s -> skipping" % baseline_rec_path)

    if args.output_folder is not None:
        target = os.path.join(args.output_folder, "errors")
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target)

    pred_dir = os.path.join(args.input_folder, "pred", "rgb")
    for pred_fn in os.listdir(pred_dir):
        gt_fn = os.path.join(args.input_folder, "gt", "rgb", pred_fn)
        other_images = [os.path.join(d, "pred", "rgb", pred_fn) for d in all_folders]
        if args.output_folder is None:
            visualize(gt_fn, other_images)
        else:
            save(gt_fn, other_images, target, photo_format=args.photo_format)
        quit()
