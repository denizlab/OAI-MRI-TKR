import random
import numpy as np
import utils
import math
import h5py
from scipy import ndimage as nd


class Random_Rotation:


    def __init__(self,image):
        self.image = image

    def RandomRotation(self,output_shape):
        '''
        The function generated a rotation matrix randomly such that the rotation axis is uniformly distributed
        on a unit sphere and the angle is uniformally distributed.
        :param output_shape: shape of the output image
        :return:Randomly rotated 3D image
        '''


        rotation_matrix =utils.generating_random_rotation_matrix()
        Output_image =self.__rotation__(output_shape, rotation_matrix)
        return Output_image

    def __rotation__(self,Output_shape,rotation_matrix):
        '''
        :param Output_shape: shape of the rotated image
        :param rotation_matrix:
        :return: Rotated 3D image based on rotation matrix
        '''

        image = self.image
        h,w,d = image.shape
        coordinate_j, coordinate_i, coordinate_k = np.meshgrid(
            np.array(range(Output_shape[1])), np.array(range(Output_shape[0])),
            np.array(range(Output_shape[2])))

        center_target = np.array([int(sh/2) for sh in list(Output_shape)]).reshape(3,1)
        center_source = np.array([int(h/2),int(w/2),int(d/2)]).reshape(3,1)
        coordinate_init = np.array([coordinate_j.flatten(), coordinate_i.flatten(), coordinate_k.flatten()])
        Rotation_matrix = rotation_matrix
        coordinate = coordinate_init - np.matlib.repmat(center_target,1,coordinate_init.shape[1]) + np.matlib.repmat(np.matmul(Rotation_matrix,center_source),1,coordinate_init.shape[1])


        mapped_to_source_coordinate = np.linalg.solve(Rotation_matrix,coordinate)
        output_coordinate_value = nd.map_coordinates(input=image,coordinates=mapped_to_source_coordinate,cval = -1000,order = 4,mode = 'constant')

        Output_image = -1000*np.ones(shape=Output_shape,dtype=float)

        for k in range(coordinate_init.shape[1]):
            Output_image[coordinate_init[0,k],coordinate_init[1,k],coordinate_init[2,k]] = output_coordinate_value[k]

        return Output_image

    def __random_rotation_matrix_fixing_rotation_axis__(self,axis = 0):

        '''
        :param axis: axis to fix
        :return: randomly rotated 3D image such that the axis of rotation is fixed
        '''

        axis_of_rotation = [0.0, 0.0, 0.0]
        axis_of_rotation[axis] = 1.0
        angle_of_rotation = 2 * math.pi * random.uniform(0, 1)
        Rotation_matrix = utils.angle_axis_to_rotation_matrix(angle=angle_of_rotation,axis=axis_of_rotation)
        return Rotation_matrix

    def RandomRotation_x_axis(self,Output_shape):

        '''
        :param Output_shape: shape of the output image
        :return: rotated 3D image along the x axis
        '''

        h, w, d = self.image.shape
        rotation_matrix = self.__random_rotation_matrix_fixing_rotation_axis__(axis = 0 )
        Output_image = self.__rotation__(Output_shape, rotation_matrix)
        return Output_image

    def RandomRotation_y_axis(self,Output_shape):
        '''
        :param Output_shape: shape of the output image
        :return: rotated 3D image along the y axis
        '''

        h, w, d = self.image.shape
        rotation_matrix = self.__random_rotation_matrix_fixing_rotation_axis__(axis=1)
        Output_image = self.__rotation__(Output_shape, rotation_matrix)
        return Output_image

    def RandomRotation_z_axis(self,Output_shape):
        '''
        :param Output_shape: shape of the output image
        :return: rotated 3D image along the z axis
        '''
        h, w, d = self.image.shape
        rotation_matrix = self.__random_rotation_matrix_fixing_rotation_axis__(axis=2)
        print("rotation_matrix: ",rotation_matrix)
        Output_image = self.__rotation__(Output_shape, rotation_matrix)
        return Output_image


class RandomCrop:

    '''
    Randomly Crop 3D image
    '''

    def __init__(self,image):
        self.image = image
        self.h,self.w,self.d = image.shape

    def __functional__(self,size):
        '''
        :param size: crop size
        :return: cropped 3D image
        '''

        h, w, d = self.image.shape
        crop_h, crop_w, crop_d = size
        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        k = random.randint(0, d - crop_d)
        crop_image = self.image[i:i + crop_h, j:j + crop_w, k:k + crop_d]
        return crop_image

    def crop_along_hieght_width(self,crop_size):

        crop = (crop_size[0],crop_size[2],self.d)
        self.crop_image = self.__functional__(size=crop)
        return self.crop_image

    def crop_along_hieght_width_depth(self,crop_size):
        self.crop_image = self.__functional__(size=crop_size)
        return self.crop_image


class CenterCrop:

    '''
    CenterCrop Images
    '''

    def __init__(self,image):
        self.image = image
        self.h,self.w,self.d = image.shape

    def __functional__(self,size):
        '''
        :param size: crop Size
        :return: Center Crop Images
        '''
        crop_h = int((self.h-size[0])/2)
        crop_w = int((self.w-size[1])/2)
        crop_d = int((self.d-size[2])/2)
        return self.image[crop_h:crop_h+size[0],crop_w:crop_w+size[1],crop_d:crop_d+size[2]]

    def crop(self,size):
        return self.__functional__(size)


class RandomFlip:

    def __init__(self,image,p=0.5):
        self.image = image
        self.p = p
        self.h,self.w,self.d = image.shape

    def horizontal_flip(self,p=-1):
        '''
        :param p: probability of flip
        :return: randomly horizontaly flipped image
        '''
        if p == -1:
            p = self.p

        integer = random.randint(0, 1)
        if integer <= p:
            output_image = self.image[:, -1:0:-1, :]
        else:
            output_image = self.image

        return output_image

    def vertical_flip(self,p=-1):

        '''
        :param p: probability of flip
        :return: randomly vertically flipped image
        '''

        if p == -1:
            p = self.p

        integer = random.randint(0, 1)
        if integer <= p:
            output_image = np.flipud(self.image)
        else:
            output_image = self.image

        return output_image
    
    def horizontal_flip(self,p=-1):

        '''
        :param p: probability of flip
        :return: randomly vertically flipped image
        '''

        if p == -1:
            p = self.p

        integer = random.randint(0, 1)
        if integer <= p:
            output_image = np.fliplr(self.image)
        else:
            output_image = self.image

        return output_image
