import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops

class DepthVideo:
    # Height filter settings (camera-relative)
    # Set to None to disable, or (below_cam, above_cam) in meters
    # e.g., (0.5, 0.5) means ±0.5m from camera height
    HEIGHT_FILTER = (0.5, 0.5)  # Set to (0.5, 0.5) to enable

    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        
    def to(self, device="cuda"):
        self.tstamp = self.tstamp.to(device=device)
        self.images = self.images.to(device=device)
        self.dirty = self.dirty.to(device=device)
        self.red = self.red.to(device=device)
        self.poses = self.poses.to(device=device)
        self.disps = self.disps.to(device=device)
        self.disps_sens = self.disps_sens.to(device=device)
        self.disps_up = self.disps_up.to(device=device)
        self.intrinsics = self.intrinsics.to(device=device)

        self.fmaps = self.fmaps.to(device=device)
        self.nets = self.nets.to(device=device)
        self.inps = self.inps.to(device=device)

        return self

    def __del__(self):
        # delete all tensors
        del self.tstamp
        del self.images
        del self.dirty
        del self.red
        del self.poses
        del self.disps
        del self.disps_sens
        del self.disps_up
        del self.intrinsics
        del self.fmaps
        del self.nets
        del self.inps

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8].cuda()
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def compute_height_mask(self, ix=None):
        """
        Compute mask of pixels within camera-relative height range.

        For a horizontal camera, computes which pixels have depths that
        place them within HEIGHT_FILTER range of the camera height.

        Args:
            ix: frame indices (None = all frames up to counter)

        Returns:
            mask: (N, H, W) boolean tensor, True = within height range
        """
        if self.HEIGHT_FILTER is None:
            return None

        below_cam, above_cam = self.HEIGHT_FILTER

        if ix is None:
            ix = torch.arange(self.counter.value, device=self.disps.device)
        elif not isinstance(ix, torch.Tensor):
            ix = torch.tensor(ix, device=self.disps.device)

        # get intrinsics (fy, cy are what matter for vertical)
        fy = self.intrinsics[0, 1]
        cy = self.intrinsics[0, 3]

        ht, wd = self.disps.shape[1:]

        # create row indices
        v = torch.arange(ht, device=self.disps.device, dtype=torch.float)

        # compute max depth per row that keeps point in height range
        # for horizontal camera: Y_cam = (v - cy) * d / fy
        # want -above_cam <= -Y_cam <= below_cam
        v_offset = v - cy

        # max depth based on height limits
        max_depth = torch.full((ht,), float('inf'), device=self.disps.device)

        # below optical center (v > cy): looking down, limit by below_cam
        below_mask = v_offset > 0.5  # small epsilon to avoid division issues
        max_depth[below_mask] = below_cam * fy / v_offset[below_mask]

        # above optical center (v < cy): looking up, limit by above_cam
        above_mask = v_offset < -0.5
        max_depth[above_mask] = above_cam * fy / (-v_offset[above_mask])

        # at optical center: all depths valid (already inf)

        # expand to (H, W) - same limit for all columns
        max_depth_map = max_depth.view(ht, 1).expand(ht, wd)

        # get depths from disparities (depth = 1/disp)
        disps_selected = self.disps[ix]
        depths = torch.where(disps_selected > 0.001, 1.0 / disps_selected,
                           torch.full_like(disps_selected, float('inf')))

        # mask: depth <= max_depth
        height_mask = depths <= max_depth_map.unsqueeze(0)

        return height_mask

    def get_height_weight_mask(self, ix=None):
        """
        Get weight multiplier for BA based on height validity.

        Returns 1.0 for valid points, 0.0 for out-of-range points.
        This can be used to reduce influence of out-of-range points in BA.
        """
        mask = self.compute_height_mask(ix)
        if mask is None:
            return None
        return mask.float()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")

        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:
            # optimization: only clone poses that are actually needed
            # instead of cloning all poses up to counter.value
            if len(ii) <= 10:
                # for small queries, determine the required pose range
                all_indices = torch.cat([ii, jj])
                max_idx = all_indices.max().item() + 1
                # only clone the poses we need (plus small buffer for safety)
                poses = self.poses[:max_idx].clone()
            else:
                # for large queries, clone all (original behavior)
                poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)
