#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.io import loadmat

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

#load project modules
import model_keras
from model_cfg import *

#Information
__author__ = "Tobias Grundmann, Adrian Schneuwly, Johannes Oswald"
__copyright__ = "Copyright 2016, 3D Vision, ETH Zurich, CVP Group"
__credits__ = ["Martin Oswald", "Pablo Speciale"]
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Finished"


def load_pc(fname):
    """
    This function loads a .mat file density grid from a tango tablet
    :param fname(str): filename of density grid, .mat file
    :return: numpy ndarray with density grid data as float32 type
    """
    f = loadmat(fname)
    data = f["data"].astype(np.float32)
    return data


def voxilize(np_pc, rot=None):
    """
    This function converts a tango tablet matrix into a voxnet voxel volume
    :param n_pc: numpy ndarray with density grid data from load_pc
    :param rot: ability to roate picture rot times and take rot recongnitions
    :return: voxillized version of density grid that is congruent with voxnet size
    """
    p = 80

    max_dist = 0.0
    for it in range(0, 3):
        min = np.amin(np_pc[:, it])
        max = np.amax(np_pc[:, it])
        dist = max - min

        if dist > max_dist:
            max_dist = dist

        np_pc[:, it] = np_pc[:, it] - dist / 2 - min

        cls = 29

        vox_sz = dist / (cls - 1)

        np_pc[:, it] = np_pc[:, it] / vox_sz

    for it in range(0, 3):
        np_pc[:, it] = np_pc[:, it] + (cls-1) / 2

    np_pc = np.rint(np_pc).astype(np.uint32)

    # fill vox array
    vox = np.zeros([30, 30, 30])
    for (pc_x, pc_y, pc_z) in np_pc:
        if random.randint(0, 100) < 80:
            vox[pc_x, pc_y, pc_z] = 1

    if rot is not None:
        a = 1

    np_vox = np.zeros([1, 1, 32, 32, 32])
    np_vox[0, 0, 1:-1, 1:-1, 1:-1] = vox

    return np_vox


def voxel_scatter(np_vox):
    """
    This dunction
    :param np_vox(ndarray): numpy ndarray of 5 dimensions with voxel volume at [~, ~, x, y, z]
    :return: numpy ndarray of num points by 3 that can be plotted by matplotlib scatter plot
    """
    vox_scat = np.zeros([0,3], dtype=np.uint32)

    for x in range(0, np_vox.shape[2]):
        for y in range(0, np_vox.shape[3]):
            for z in range(0, np_vox.shape[4]):
                if np_vox[0, 0, x, y, z] == 1.0:
                    arr_tmp = np.zeros([1,3], dtype=np.uint32)
                    arr_tmp[0,:] = (x, y ,z)
                    vox_scat = np.concatenate((vox_scat, arr_tmp))
    return vox_scat


class detector_voxnet:
    """
    use model voxnet predict
    """

    def __init__(self, weights, nb_classes = 39):
        """
        Initializes the voxnet_model from model_keras with the given weights and classes, ready for detection

        :param weights: keras weights file for voxnet, hdf5 type
        :param nb_classes: number of classes that the model was trained with
        """

        self.mdl = model_keras.model_vt(nb_classes=nb_classes, dataset_name="modelnet")

        self.mdl.load_weights(weights)

    def predict(self, X_pred, is_pc=False):
        """
        This will predict the probability for every given Object and pool the results and return the label and
        achived probability
        :param X_pred: Input Object nd.array of min 3 dimensions either voxnet size or other densityr cloud size
        :param is_pc: if not voxnet size (1*1*32*32*32) this has to be set to true
        :return: return the label name of the detected object, currently only work for objects from modelnet40 set
                 return probality of which the detector puts in the detected object
        """
        if is_pc == True:
            X_pred - voxilize(X_pred)

        proba_all = self.mdl.predict(X_pred)

        label = str(np.argmax(proba_all) + 2)

        label = class_id_to_name_modelnet40[label]

        proba = np.amax(proba_all)

        return label, proba