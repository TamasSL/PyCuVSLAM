import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops

from cuda_timer import CudaTimer
from functools import partial

if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast


class FactorGraph:
    # maximum number of inactive factors to retain (prevents O(N²) memory growth)
    # reduced from 10000 to 2000 for better performance
    MAX_INACTIVE_FACTORS = 2000
    # maximum number of bad edges to track
    MAX_BAD_EDGES = 1000
    # sliding window size for distance computation (prevents O(N²) distance matrix)
    # reduced from 100 to 50 for better performance
    PROXIMITY_WINDOW = 50

    def __init__(self, video, update_op, device="cuda", corr_impl="volume", max_factors=-1, upsample=False):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # cached edge sets for O(1) deduplication instead of O(N^2) tensor comparison
        self._active_edges = set()
        self._inactive_edges = set()

        # cache for inactive factor concatenation in update() - invalidated when inactive factors change
        self._inac_cache_valid = False
        self._inac_cache_t0 = -1
        self._cached_ii_inac_filtered = None
        self._cached_jj_inac_filtered = None
        self._cached_target_inac_filtered = None
        self._cached_weight_inac_filtered = None

    def __filter_repeated_edges(self, ii, jj):
        """remove duplicate edges - O(N) set-based deduplication"""
        if len(ii) == 0:
            return ii, jj

        # use set lookup for O(1) per-edge check instead of O(N^2) tensor comparison
        combined_edges = self._active_edges | self._inactive_edges

        if len(combined_edges) == 0:
            return ii, jj

        # convert to list for iteration (single GPU->CPU transfer)
        ii_list = ii.tolist()
        jj_list = jj.tolist()

        # filter edges not in existing sets
        keep_indices = [
            idx for idx, (i, j) in enumerate(zip(ii_list, jj_list))
            if (i, j) not in combined_edges
        ]

        if len(keep_indices) == len(ii_list):
            return ii, jj
        elif len(keep_indices) == 0:
            return ii[:0], jj[:0]  # return empty tensors with same device/dtype

        keep_mask = torch.tensor(keep_indices, device=ii.device, dtype=torch.long)
        return ii[keep_mask], jj[keep_mask]

    def _invalidate_inac_cache(self):
        """invalidate the inactive factor cache when inactive factors change"""
        self._inac_cache_valid = False
        self._cached_ii_inac_filtered = None
        self._cached_jj_inac_filtered = None
        self._cached_target_inac_filtered = None
        self._cached_weight_inac_filtered = None

    def _update_edge_sets(self):
        """rebuild edge sets from current tensors (call after rm_factors/rm_keyframe)"""
        # only rebuild active edges (small, bounded by max_factors)
        if len(self.ii) > 0:
            self._active_edges = set(zip(self.ii.tolist(), self.jj.tolist()))
        else:
            self._active_edges = set()

        # for inactive edges, only rebuild if size is small enough
        # otherwise, accept slightly stale data for deduplication (safe - just may re-add some edges)
        if len(self.ii_inac) > 0 and len(self.ii_inac) <= 500:
            self._inactive_edges = set(zip(self.ii_inac.tolist(), self.jj_inac.tolist()))
        elif len(self.ii_inac) == 0:
            self._inactive_edges = set()
        # else: keep existing _inactive_edges (may be stale but safe)

        # invalidate inactive factor cache
        self._invalidate_inac_cache()

    def _prune_inactive_factors(self):
        """prune oldest inactive factors to prevent unbounded growth"""
        n_inac = len(self.ii_inac)
        if n_inac <= self.MAX_INACTIVE_FACTORS:
            return

        # keep only the most recent factors (highest frame indices tend to be more recent)
        # sort by max(ii, jj) to prioritize recent frames
        max_indices = torch.maximum(self.ii_inac, self.jj_inac)
        _, keep_order = torch.topk(max_indices, self.MAX_INACTIVE_FACTORS, largest=True)

        self.ii_inac = self.ii_inac[keep_order]
        self.jj_inac = self.jj_inac[keep_order]
        self.target_inac = self.target_inac[:, keep_order]
        self.weight_inac = self.weight_inac[:, keep_order]

        # update the inactive edge set (small enough after pruning)
        self._inactive_edges = set(zip(self.ii_inac.tolist(), self.jj_inac.tolist()))

        # invalidate cache
        self._invalidate_inac_cache()

    def _prune_bad_edges(self):
        """prune oldest bad edges to prevent unbounded growth"""
        n_bad = len(self.ii_bad)
        if n_bad <= self.MAX_BAD_EDGES:
            return

        # keep only most recent bad edges
        max_indices = torch.maximum(self.ii_bad, self.jj_bad)
        _, keep_order = torch.topk(max_indices, self.MAX_BAD_EDGES, largest=True)

        self.ii_bad = self.ii_bad[keep_order]
        self.jj_bad = self.jj_bad[keep_order]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self._prune_bad_edges()  # prevent unbounded growth
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """add edges to factor graph"""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if (
            self.max_factors > 0
            and self.ii.shape[0] + ii.shape[0] > self.max_factors
            and self.corr is not None
            and remove
        ):

            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii, 0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj, c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # update edge set cache for O(1) deduplication
        self._active_edges.update(zip(ii.tolist(), jj.tolist()))

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)


    @autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)
            # prune inactive factors to prevent O(N²) memory growth
            self._prune_inactive_factors()

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]

        # rebuild edge set caches after removal
        self._update_edge_sets()


    @autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        t = self.video.counter.value
        # with self.video.get_lock():
        self.video.images[ix : t - 1] = self.video.images[ix + 1 : t].clone()
        self.video.poses[ix : t - 1] = self.video.poses[ix + 1 : t].clone()
        self.video.disps[ix : t - 1] = self.video.disps[ix + 1 : t].clone()
        self.video.disps_sens[ix : t - 1] = self.video.disps_sens[ix + 1 : t].clone()
        self.video.intrinsics[ix : t - 1] = self.video.intrinsics[ix + 1 : t].clone()

        self.video.nets[ix : t - 1] = self.video.nets[ix + 1 : t].clone()
        self.video.inps[ix : t - 1] = self.video.inps[ix + 1 : t].clone()
        self.video.fmaps[ix : t - 1] = self.video.fmaps[ix + 1 : t].clone()
        self.video.tstamp[ix: t - 1] = self.video.tstamp[ix + 1 : t].clone()

        # invalidate cache since we're modifying inactive factors
        self._invalidate_inac_cache()

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:, ~m]
            self.weight_inac = self.weight_inac[:, ~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)
        # edge sets are rebuilt in rm_factors, but indices changed so rebuild again
        self._update_edge_sets()

    @autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # motion features
        with autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
        
        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive and len(self.ii_inac) > 0:
                # use cached filtered inactive factors if available and t0 matches
                if self._inac_cache_valid and self._inac_cache_t0 == t0:
                    ii_inac_f = self._cached_ii_inac_filtered
                    jj_inac_f = self._cached_jj_inac_filtered
                    target_inac_f = self._cached_target_inac_filtered
                    weight_inac_f = self._cached_weight_inac_filtered
                else:
                    # compute and cache filtered inactive factors
                    m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                    ii_inac_f = self.ii_inac[m]
                    jj_inac_f = self.jj_inac[m]
                    target_inac_f = self.target_inac[:, m]
                    weight_inac_f = self.weight_inac[:, m]
                    # cache for subsequent calls with same t0
                    self._inac_cache_valid = True
                    self._inac_cache_t0 = t0
                    self._cached_ii_inac_filtered = ii_inac_f
                    self._cached_jj_inac_filtered = jj_inac_f
                    self._cached_target_inac_filtered = target_inac_f
                    self._cached_weight_inac_filtered = weight_inac_f

                if len(ii_inac_f) > 0:
                    ii = torch.cat([ii_inac_f, self.ii], 0)
                    jj = torch.cat([jj_inac_f, self.jj], 0)
                    target = torch.cat([target_inac_f, self.target], 1)
                    weight = torch.cat([weight_inac_f, self.weight], 1)
                else:
                    ii, jj, target, weight = self.ii, self.jj, self.target, self.weight
            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1


    @autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            # print("Global BA Iteration #{}".format(step+1))
            with CudaTimer("backend", enabled=False):
                with autocast(enabled=False):
                    coords1, mask = self.video.reproject(self.ii, self.jj)
                    motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                    motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

                s = 8
                # pre-compute batch assignments to avoid GPU sync in loop
                ii_min = self.ii.min().item()
                jj_max = self.jj.max().item()

                for i in range(ii_min, jj_max+1, s):
                    v = (self.ii >= i) & (self.ii < i + s)

                    # use sum() which can be checked without sync in most cases
                    if not v.any():
                        continue

                    iis = self.ii[v]
                    jjs = self.jj[v]

                    ht, wd = self.coords0.shape[0:2]

                    with autocast(enabled=True):
                        corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                        net, delta, weight, damping, upmask = \
                            self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                        if self.upsample:
                            self.video.upsample(torch.unique(iis), upmask)

                    self.net[:,v] = net
                    self.target[:,v] = coords1[:,v] + delta.float()
                    self.weight[:,v] = weight.float()
                    self.damping[torch.unique(iis)] = damping

                damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP

                if use_inactive and len(self.ii_inac) > 0:
                    # limit inactive factors to most recent ones for performance
                    # filter to recent frames only (within PROXIMITY_WINDOW of current)
                    min_frame = max(0, t - self.PROXIMITY_WINDOW)
                    m = (self.ii_inac >= min_frame) | (self.jj_inac >= min_frame)
                    if m.any():
                        ii = torch.cat([self.ii_inac[m], self.ii], 0)
                        jj = torch.cat([self.jj_inac[m], self.jj], 0)
                        target = torch.cat([self.target_inac[:, m], self.target], 1)
                        weight = torch.cat([self.weight_inac[:, m], self.weight], 1)
                    else:
                        ii, jj, target, weight = self.ii, self.jj, self.target, self.weight
                else:
                    ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

                damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP
                target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
                weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
                
                self.age += 1

                # dense bundle adjustment
                self.video.ba(target, weight, damping, ii, jj, 1, t, 
                    itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

                self.video.dirty[:t] = True

    def add_neighborhood_factors(self, t0, t1, r=3):
        """add edges between neighboring frames within radius r"""

        ii, jj = torch.meshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(t0, t1, device=self.device),
            indexing="ij",
        )

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(
        self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False
    ):
        """add edges to the factor graph based on distance

        Uses a sliding window to limit distance computation to O(W²) instead of O(N²)
        where W is PROXIMITY_WINDOW and N is total frame count.
        """

        t = self.video.counter.value

        # apply sliding window to limit distance computation complexity
        # only compute distances for frames within the window
        window_t0 = max(t0, t - self.PROXIMITY_WINDOW)
        window_t1 = max(t1, t - self.PROXIMITY_WINDOW)

        ix = torch.arange(window_t0, t)
        jx = torch.arange(window_t1, t)

        if len(ix) == 0 or len(jx) == 0:
            return

        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta).cpu()
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        # only consider edges within the window for NMS suppression
        # use set-based lookup for O(1) checks instead of iterating all edges
        existing_edges = self._active_edges | self._inactive_edges

        # also include bad edges but limit to recent ones
        if len(self.ii_bad) > 0:
            # only consider bad edges within the window
            bad_mask = (self.ii_bad >= window_t0) & (self.jj_bad >= window_t1)
            ii_bad_window = self.ii_bad[bad_mask]
            jj_bad_window = self.jj_bad[bad_mask]
            if len(ii_bad_window) > 0:
                existing_edges = existing_edges | set(zip(ii_bad_window.tolist(), jj_bad_window.tolist()))

        # vectorized NMS suppression - only for edges within window
        ii1_list = []
        jj1_list = []
        for (i, j) in existing_edges:
            if window_t0 <= i < t and window_t1 <= j < t:
                ii1_list.append(i)
                jj1_list.append(j)

        if len(ii1_list) > 0:
            ii1_np = np.array(ii1_list)
            jj1_np = np.array(jj1_list)
            # pre-compute suppression radius per edge
            edge_diffs = np.abs(ii1_np - jj1_np)
            max_radii = np.maximum(np.minimum(edge_diffs - 2, nms), 0)

            for idx, (i, j) in enumerate(zip(ii1_np, jj1_np)):
                r = max_radii[idx]
                if r == 0:
                    continue
                for di in range(-nms, nms + 1):
                    for dj in range(-nms, nms + 1):
                        if abs(di) + abs(dj) <= r:
                            i1 = i + di
                            j1 = j + dj

                            if (window_t0 <= i1 < t) and (window_t1 <= j1 < t):
                                d[(i1 - window_t0) * (t - window_t1) + (j1 - window_t1)] = np.inf

        es = []
        for i in range(window_t0, t):
            if self.video.stereo:
                es.append((i, i))
                if (i - window_t0) * (t - window_t1) + (i - window_t1) < len(d):
                    d[(i - window_t0) * (t - window_t1) + (i - window_t1)] = np.inf

            for j in range(max(i - rad - 1, window_t0), i):
                es.append((i, j))
                es.append((j, i))
                if (i - window_t0) * (t - window_t1) + (j - window_t1) < len(d):
                    d[(i - window_t0) * (t - window_t1) + (j - window_t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k] > thresh:
                continue

            if self.max_factors > 0:
                if len(es) > self.max_factors:
                    break

            i = ii[k]
            j = jj[k]

            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (window_t0 <= i1 < t) and (window_t1 <= j1 < t):
                            d[(i1 - window_t0) * (t - window_t1) + (j1 - window_t1)] = np.inf

        if len(es) == 0:
            return

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
