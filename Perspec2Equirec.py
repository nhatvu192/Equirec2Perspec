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
    ) -> np.ndarray:
        """
        Given field of view, left/right and up/down viewing angle, the size of the
        equirectangular canvas, and the the input perspective image, paint the
        perspective image on to the corresponding location in the equirectangular
        canvas.

        The steps performed in this function are as follow.
        1. For each pixel in the perspective image, compute a corresponding
        pixel in the equirectangular image.

        2. Initialize an empty canvas

        3. Fill the canvas using the mapping from step 1.

        4. Perform interpolation on the filled canvas.

        5. Fill the remaining missing values with 0.

        Args:
            FOV: angle of fiew of view

            THETA: left/right viewing angle

            PHI: up/down viewing angle

            canvas_height: height of the equirectangular canvas to draw on

            canvas_width: width of the equirectangular canvas to draw on

        Returns:
            A numpy array of size (canvas_height, canvas_width, 3) with the
            perspective image drawn at the appropriate location and 0 everywhere.
        """
        XY = self.GetCanvasPixelIndex(FOV, THETA, PHI, canvas_height, canvas_width)

        canvas = np.full((canvas_height, canvas_width, 3), np.nan)

        for perspective_index_y in range(XY.shape[0]):
            for perspective_index_x in range(XY.shape[1]):
                canvas_index_x = int(XY[perspective_index_y, perspective_index_x, 0])
                canvas_index_y = int(XY[perspective_index_y, perspective_index_x, 1])
                canvas[canvas_index_y, canvas_index_x] = self.img[
                    perspective_index_y, perspective_index_x
                ]

        xrange = np.arange(canvas_width)
        yrange = np.arange(canvas_height)

        mask = np.ma.masked_invalid(canvas[..., 0]).mask

        xx, yy = np.meshgrid(xrange, yrange)

        canvas = interpolate.griddata(
            (xx[~mask], yy[~mask]),
            canvas[~mask].reshape(-1, 3),
            (xx, yy),
        )

        canvas = np.where(np.isnan(canvas), 0, canvas)

        return canvas.astype(np.uint8)
