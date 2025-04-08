# Copyright (c) 2025 ICHIRO ITS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np

def rotation(angle: float = 0) -> np.ndarray:
    """
    A 2D homogeneous rotation (3x3 matrix)
    """
    # fmt: off
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
    # fmt: on

def translation(x: float = 0, y: float = 0) -> np.ndarray:
    """
    A 2D homogeneous translation (3x3 matrix)
    """
    # fmt: off
    return np.array([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]], dtype=np.float32)
    # fmt: on

def frame(x: float = 0, y: float = 0, angle: float = 0) -> np.ndarray:
    """
    A 2D transformation (rotation and translation, 3x3 matrix)
    """
    return translation(x, y) @ rotation(angle)

def frame_inv(T: np.ndarray) -> np.ndarray:
    """
    Inverts a 3x3 2D matrix
    """
    R = T[:2, :2]  # Rotation
    t = T[:2, 2:]  # Translation
    upper = np.hstack((R.T, -R.T @ t))
    lower = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.vstack((upper, lower))

def apply(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a matrix transformation to a point
    """
    return (T @ [*point, 1.0])[:2]
