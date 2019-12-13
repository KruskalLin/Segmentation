#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  Copyright (C) 2013
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#  Authors: Tobias Kuehnl <tkuehnl@cor-lab.uni-bielefeld.de>
#           Jannik Fritsch <jannik.fritsch@honda-ri.de>
#

import sys, os
from glob import glob
from helper import evalExp, pxEval_maximizeFMeasure, getGroundTruth
import numpy as np
import cv2  # OpenCV


class dataStructure:
    '''
    All the defines go in here!
    '''

    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_propertyList = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp']


#########################################################################


# function that does the evaluation
#########################################################################
def main(result_dir, train_dir, debug=False):
    '''
    main method of evaluateRoad
    :param result_dir: directory with the result propability maps, e.g., /home/elvis/kitti_road/my_results
    :param gt_dir: training directory (has to contain gt_image_2)  e.g., /home/elvis/kitti_road/training
    :param debug: debug flag (OPTIONAL)
    '''

    print("Starting evaluation ...")
    thresh = np.array(range(0, 256)) / 255.0
    trainData_subdir_gt = 'origin/'
    gt_dir = os.path.join(train_dir, trainData_subdir_gt)

    assert os.path.isdir(result_dir), 'Cannot find result_dir: %s ' % result_dir

    # In the submission_dir we expect the probmaps! 
    submission_dir = result_dir
    assert os.path.isdir(submission_dir), 'Cannot find %s, ' % submission_dir

    # init result
    prob_eval_scores = []  # the eval results in a dict
    gt_fileList = glob(os.path.join(gt_dir, '*'))
    assert len(gt_fileList) > 0, 'Error reading ground truth'
    # Init data for categgory
    category_ok = True  # Flag for each cat
    totalFP = np.zeros(thresh.shape)
    totalFN = np.zeros(thresh.shape)
    totalPosNum = 0
    totalNegNum = 0

    for fn_curGt in gt_fileList:
        fn_curGt = fn_curGt.replace('\\', '/')
        file_key = fn_curGt.split('/')[-1].split('.')[0]
        print("Processing file: %s " % file_key)

        # Read GT
        cur_gt, validArea = getGroundTruth(fn_curGt)
        # Read probmap and normalize
        fn_curProb = os.path.join(submission_dir, file_key + dataStructure.prob_end)
        if not os.path.isfile(fn_curProb):
            print("--> Will now abort evaluation for this particular category.")
            category_ok = False
            break

        cur_prob = cv2.imread(fn_curProb, 0)
        cur_prob = np.clip((cur_prob.astype('f4')) / (np.iinfo(cur_prob.dtype).max), 0., 1.)
        FN, FP, posNum, negNum = evalExp(cur_gt, cur_prob, thresh, validMap=None, validArea=validArea)
        assert FN.max() <= posNum, 'BUG @ poitive samples'
        assert FP.max() <= negNum, 'BUG @ negative samples'

        # collect results for whole category
        totalFP += FP
        totalFN += FN
        totalPosNum += posNum
        totalNegNum += negNum

    if category_ok:
        print("Computing evaluation scores...")
        # Compute eval scores!
        print(pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh=thresh))
        prob_eval_scores.append(pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh=thresh))

        factor = 100
        for property in dataStructure.eval_propertyList:
            print('%s: %4.2f ' % (property, prob_eval_scores[-1][property] * factor,))


#########################################################################
# evaluation script
#########################################################################
if __name__ == "__main__":
    main('./predict/', './')
