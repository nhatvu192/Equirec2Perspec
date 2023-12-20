import cv2
import numpy as np
from scipy import interpolate
import Equirec2Perspec as EP


class Perspective:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.img_height, self.img_width, _ = self.img.shape

    def GetCanvasPixelIndex(
        self,
        FOV: float,
        THETA: float,
        PHI: float,
        canvas_height: int,
        canvas_width: int,
    ) -> np.ndarray:
        """
        Given field of view, left/right and up/down viewing angle, the size of the
        equirectangular canvas, and the size of the input perspective image,
        return a 2D numpy array of shape (height, width, 2) where height and width
        are the height and width of the perspective image such that each element (i,j)
        contains the corresponding pixel index in the equirectangular canvas.

        The steps performed in this function are as followed.
        1. Normalize the distance from the center of the sphere to the perspective
        image plane to 1 and center the perspective image onto the origin of x and y
        coordinates

        2. Rotate the plane by THETA and PHI.

        3. Find the latitude and longitude coordinate of the projection of each
        of the pixel on the perspective plane onto the sphere.

        4. Convert the latitude and longitude to pixel coordinate in the equirectangular

        Args:
            FOV: angle of fiew of view

            THETA: left/right viewing angle

            PHI: up/down viewing angle

            canvas_height: height of the equirectangular canvas to draw on

            canvas_width: width of the equirectangular canvas to draw on

        Returns:
            a numpy array of shape (height, width, 2) where height and width
            are the height and width of self.img and each pixel contains the
            corresponding pixel in the equirectangular canvas.
        """

        f = 0.5 * self.img_width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (self.img_width - 1) / 2.0
        cy = (self.img_height - 1) / 2.0
        K = np.array(
            [
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1],
            ],
            np.float32,
        )
        K_inv = np.linalg.inv(K)

        x = np.arange(self.img_width)
        y = np.arange(self.img_height)
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
        return EP.lonlat2XY(lonlat, shape=(canvas_height, canvas_width)).astype(
            np.float32
        )

    def GetEquirectangular(
        self,
        FOV: float,
        THETA: float,
        PHI: float,
        canvas_height: int,
        canvas_width: int,
    ):
        XY = self.GetCanvasPixelIndex(FOV, THETA, PHI, canvas_height, canvas_width)

        result = np.full((canvas_height, canvas_width, 3), np.nan)

        for i in range(XY.shape[0]):
            for j in range(XY.shape[1]):
                x = int(XY[i, j, 0])
                y = int(XY[i, j, 1])
                result[y - 1, x - 1] = self.img[i, j]

        xrange = np.arange(canvas_width)
        yrange = np.arange(canvas_height)

        mask = np.ma.masked_invalid(result[..., 0]).mask

        xx, yy = np.meshgrid(xrange, yrange)

        result = interpolate.griddata(
            (xx[~mask], yy[~mask]),
            result[~mask].reshape(-1, 3),
            (xx, yy),
        )

        result = np.where(np.isnan(result), 0, result)

        return result.astype(np.uint8)
