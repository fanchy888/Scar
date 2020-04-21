import numpy as np


MID_DIST = 1e-10


class LocalOutlierFactor:
    def __init__(self, samples, k=10):
        self.samples = np.array(samples)
        self.k = min(k, len(samples) - 1)
        self._dk = []
        self._lrd = {}
        self._nn = {}
        self.dist = []
        self._distance()

    @property
    def dk(self):
        if not self._dk:
            for i, _ in enumerate(self.samples):
                self._dk.append(self._k_distance(i))
        return self._dk

    def _distance(self):
        for i, x in enumerate(self.samples):
            self.dist.append(np.linalg.norm(self.samples - x, 2, axis=1).tolist())

    def _k_distance(self, i):
        k_dist = max(sorted(self.dist[i])[self.k], MID_DIST)
        return k_dist

    @property
    def nn(self):
        if not self._nn:
            for i, _ in enumerate(self.samples):
                self._nn[i] = [j for j, d in enumerate(self.dist[i]) if j != i and d <= self.dk[i]]
        return self._nn

    def _reach_distance(self, i, j):
        return max(self.dist[i][j], self.dk[j])

    def local_reachable_density(self, i):
        if i not in self._lrd:
            sum_lrd = 0
            for n in self.nn[i]:
                sum_lrd += self._reach_distance(i, n)
            self._lrd[i] = len(self.nn[i]) / sum_lrd
        return self._lrd[i]

    def local_outlier_factor(self, i):
        lrd_o = 0
        lrd_p = self.local_reachable_density(i)
        for n in self.nn[i]:
            lrd_o += self.local_reachable_density(n)

        lof = lrd_o / len(self.nn[i]) / lrd_p
        return lof

    def decision(self, threshold=5, pct=0):
        lofs = []
        for i, _ in enumerate(self.samples):
            lofs.append(self.local_outlier_factor(i))

        lofs = np.array(lofs)
        if pct:
            n = len(self.samples)
            k = min(int(n * pct), n)
            return np.argsort(lofs)[-k:]
        else:

            return np.where(lofs > threshold)[0]



