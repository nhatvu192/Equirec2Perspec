import cv2
import numpy as np
from scipy import interpolate
import Equirec2Perspec as EP


class Perspective:
    def __init__(self, img):
        self._img = img
        self.height, self.width, _ = self._img.shape

    def GetEquirectangular(self, FOV, THETA, PHI, final_height, final_width):
        f = 0.5 * self.width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        K = np.array(
            [
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1],
            ],
            np.float32,
        )
        K_inv = np.linalg.inv(K)

        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = EP.xyz2lonlat(xyz)
        XY = EP.lonlat2XY(lonlat, shape=(final_height, final_width)).astype(np.float32)

        result = np.full((final_height, final_width, 3), np.nan)

        for i in range(XY.shape[0]):
            for j in range(XY.shape[1]):
                x = int(XY[i, j, 0])
                y = int(XY[i, j, 1])
                result[y - 1, x - 1] = self._img[i, j]

        xrange = np.arange(final_width)
        yrange = np.arange(final_height)

        mask = np.ma.masked_invalid(result[..., 0]).mask

        xx, yy = np.meshgrid(xrange, yrange)

        result = interpolate.griddata(
            (xx[~mask], yy[~mask]),
            result[~mask].reshape(-1, 3),
            (xx, yy),
        )

        result = np.where(np.isnan(result), 0, result)

        return result.astype(np.uint8)
