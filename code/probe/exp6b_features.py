"""Frozen dense image features, pooled per piece; run on one GPU per shard."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image

from common import WORKSPACE, load_rgb
from exp6_pool_io import load_piece_masks
from exp6b_common import ROOT, Pool, rows_by_image, save_npz

PROJECTION_SEED = 1020
NATIVE_DIM = {"sam_vith": 256, "remoteclip": 1024, "dinov3": 1024}


class Encoder:
    def __init__(self, backbone, device):
        self.backbone, self.device = backbone, device
        if backbone == "sam_vith":
            from torchange.models.segment_any_change.segment_anything import sam_model_registry
            from torchange.models.segment_any_change.segment_anything.utils.transforms import ResizeLongestSide

            self.model = sam_model_registry["vit_h"](
                checkpoint=str(WORKSPACE / "models/SAM/sam_vit_h_4b8939.pth"))
            self.resize = ResizeLongestSide(1024)
        elif backbone == "remoteclip":
            import open_clip

            model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=None)
            model.load_state_dict(torch.load(
                WORKSPACE / "models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", map_location="cpu",
                weights_only=True))
            self.model = model.visual
        else:
            from transformers import DINOv3ViTModel

            directory = WORKSPACE / "models/dinov3/dinov3-vitl16-pretrain-sat493m"
            self.model = DINOv3ViTModel.from_pretrained(directory, local_files_only=True)
            self.preprocessing = json.loads((directory / "preprocessor_config.json").read_text())
        self.model = self.model.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, image):
        h, w = image.shape[:2]
        if self.backbone == "sam_vith":
            # All current datasets are square. Keep the full 64x64 SAM grid.
            if h != w or h % 64:
                raise ValueError("SAM pooling currently requires square images with side divisible by 64")
            tensor = torch.from_numpy(self.resize.apply_image(image)).permute(2, 0, 1).to(self.device)
            return self.model.image_encoder(self.model.preprocess(tensor)[None])[0].float()
        if self.backbone == "remoteclip":
            if h % 256 or w % 256:
                raise ValueError("RemoteCLIP requires image sides divisible by 256")
            # Each 256px tile is encoded once. No candidate-dependent backbone calls.
            tile_rows = []
            for y in range(0, h, 256):
                tile_cols = []
                for x in range(0, w, 256):
                    tile = Image.fromarray(image[y:y + 256, x:x + 256])
                    tensor = self.preprocess(tile)[None].to(self.device)
                    output = self.model.forward_intermediates(
                        tensor, indices=1, normalize_intermediates=True,
                        intermediates_only=True, output_fmt="NCHW")
                    tile_cols.append(output["image_intermediates"][0][0].float())
                tile_rows.append(torch.cat(tile_cols, dim=2))
            return torch.cat(tile_rows, dim=1)
        if h % 16 or w % 16:
            raise ValueError("DINOv3 requires image sides divisible by 16")
        cfg = self.preprocessing
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().to(self.device)
        tensor *= cfg["rescale_factor"]
        mean = tensor.new_tensor(cfg["image_mean"])[:, None, None]
        std = tensor.new_tensor(cfg["image_std"])[:, None, None]
        tensor = (tensor - mean) / std
        tokens = self.model(pixel_values=tensor[None]).last_hidden_state
        tokens = tokens[:, 1 + self.model.config.num_register_tokens:]
        return tokens[0].reshape(h // 16, w // 16, -1).permute(2, 0, 1).float()


def coverage_weights(rows, masks, shape, grid):
    """Exact average pooling on aligned grid cells, including sub-cell masks."""
    h, w = shape
    gh, gw = grid
    if h % gh or w % gw:
        raise ValueError("Pixel and feature grids must have integer cell sizes")
    sy, sx = h // gh, w // gw
    piece = np.zeros((len(rows), gh * gw), np.float32)
    context = np.empty_like(piece)
    cell_x, cell_y = np.arange(gw) * sx, np.arange(gh) * sy
    for i, row in enumerate(rows):
        bbox, local = masks[row["piece_id"]]
        if bbox != row["bbox"] or int(local.sum()) != row["area"]:
            raise ValueError(f'{row["image"]}/{row["piece_id"]}: mask metadata mismatch')
        x0, y0, _, _ = bbox
        yy, xx = np.nonzero(local)
        bins = ((yy + y0) // sy) * gw + (xx + x0) // sx
        piece[i] = np.bincount(bins, minlength=gh * gw) / (sy * sx)
        x0, y0, x1, y1 = row["crop"]
        wx = np.maximum(0, np.minimum(cell_x + sx, x1) - np.maximum(cell_x, x0)) / sx
        wy = np.maximum(0, np.minimum(cell_y + sy, y1) - np.maximum(cell_y, y0)) / sy
        context[i] = (wy[:, None] * wx[None, :]).ravel()
    if np.any(piece.sum(1) <= 0) or np.any(context.sum(1) <= 0):
        raise ValueError("Empty piece or context pooling weights")
    piece /= piece.sum(1, keepdims=True)
    context /= context.sum(1, keepdims=True)
    return piece, context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=tuple(NATIVE_DIM), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max_dim", type=int, default=256,
                        help="Fixed Gaussian projection cap; 0 keeps the native dimensions")
    parser.add_argument("--pool_batch", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.shards < 1 or not 0 <= args.shard < args.shards
            or args.max_dim < 0 or args.pool_batch < 1):
        parser.error("Invalid shard, max_dim or pool_batch")
    pool = Pool(args.dataset)
    native = NATIVE_DIM[args.backbone]
    dim = min(native, args.max_dim) if args.max_dim else native
    config = dict(backbone=args.backbone, native_dim=native, dim=dim,
                  projection_seed=PROJECTION_SEED,
                  projection="gaussian_1_over_sqrt_dim" if dim < native else "none",
                  extraction_version=1, layout="f1,f2,c1,c2",
                  remoteclip_tokens="last_ln_post_no_proj", dinov3_tokens="last_norm_no_cls_register",
                  pooling="exact_grid_cell_coverage")
    out = ROOT / "features" / args.backbone / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{args.shard}.npz"
    if path.exists():
        raise FileExistsError(f"{path}: move the previous shard before recomputing")
    selected = pool.image_id % args.shards == args.shard
    indices = np.flatnonzero(selected)
    temporary = out / f"shard_{args.shard}.features.npy"
    if len(indices):
        vectors = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(len(indices), 4, dim))
    else:
        vectors = np.empty((0, 4, dim), np.float16)
    encoder = Encoder(args.backbone, args.device)
    projection = None
    if dim < native:
        # CPU generator: identical matrix across backbones of the same width and all datasets.
        projection = np.random.default_rng(PROJECTION_SEED).standard_normal(
            (native, dim), dtype=np.float32) / np.sqrt(dim)
        projection = torch.from_numpy(projection.astype(np.float32)).to(args.device)
    cursor, images_done = 0, 0
    with torch.inference_mode():
        for name, rows in rows_by_image(args.dataset):
            if pool.image_index[name] % args.shards != args.shard:
                continue
            image1, image2 = [load_rgb(args.dataset, name, view) for view in ("t1", "t2")]
            if image1.shape != image2.shape:
                raise ValueError(f"{name}: image shape mismatch")
            maps = [encoder(image1), encoder(image2)]
            if maps[0].shape != maps[1].shape or maps[0].shape[0] != native:
                raise ValueError(f"{name}: unexpected encoder feature shape")
            grid = tuple(maps[0].shape[1:])
            flat = [m.flatten(1).T for m in maps]
            if projection is not None:
                flat = [m @ projection for m in flat]
            masks = load_piece_masks(args.dataset, name)
            for start in range(0, len(rows), args.pool_batch):
                batch = rows[start:start + args.pool_batch]
                weights = coverage_weights(batch, masks, image1.shape[:2], grid)
                piece, context = [torch.from_numpy(a).to(args.device) for a in weights]
                values = torch.stack((piece @ flat[0], piece @ flat[1],
                                      context @ flat[0], context @ flat[1]), dim=1)
                vectors[cursor:cursor + len(batch)] = values.cpu().numpy().astype(np.float16)
                cursor += len(batch)
            images_done += 1
            if images_done % 25 == 0:
                print(f"shard {args.shard}: {images_done} images, {cursor} pieces", flush=True)
    if cursor != len(indices):
        raise ValueError("Feature rows do not match the selected shard")
    if len(indices):
        vectors.flush()
    save_npz(path, image=np.asarray(pool.images)[pool.image_id[indices]],
             piece_id=pool.piece_id[indices], features=vectors, config=json.dumps(config),
             dataset=args.dataset, shard=args.shard, shards=args.shards)
    del vectors
    temporary.unlink(missing_ok=True)
    print(f"saved {path}: {cursor} pieces, D={dim}", flush=True)


if __name__ == "__main__":
    main()
