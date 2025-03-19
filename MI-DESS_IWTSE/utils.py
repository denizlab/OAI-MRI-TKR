# ==============================================================================
# Copyright (C) 2023 Haresh Rengaraj Rajamohan, Tianyu Wang, Kevin Leung, 
# Gregory Chang, Kyunghyun Cho, Richard Kijowski & Cem M. Deniz 
#
# This file is part of OAI-MRI-TKR
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
import random
import numpy as np
import math

def angle_axis_to_rotation_matrix(angle,axis):

    '''
    :param angle: angle of rotation
    :param axis: Axis 0f rotation
    :return: Rotation Matrix
    '''

    A = np.outer(axis,axis)
    B = np.zeros((3,3),dtype=float)
    B[0,1] = -1*axis[-1]
    B[0,2] = axis[-2]
    B[1,2] = axis[0]
    B = B - B.transpose()
    R = math.cos(angle)+np.eye(3)+ math.sin(angle)*B + (1-math.cos(angle))*A
    return R


def generating_random_rotation_matrix():
    '''
    :return: random rotation matrix, axis of rotation, angle of rotation
    random rotation matrix is selected in such a way that the axis of rotation is uinformally distributed on a unit
    sphere and the angle is uniformally distributed.
    '''

    guassian_vector = [random.gauss(mu=0, sigma=1), random.gauss(mu=0, sigma=1), random.gauss(mu=0, sigma=1)]
    axis_of_rotation = list(np.array(guassian_vector) / np.linalg.norm(x=np.array(guassian_vector)))
    angle_of_rotation = 2 * math.pi * random.uniform(0, 1)
    rotation_matrix = angle_axis_to_rotation_matrix(angle=angle_of_rotation, axis=axis_of_rotation)
    return rotation_matrix, axis_of_rotation, angle_of_rotation
