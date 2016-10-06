###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#		   http://ilastik.org/license.html
###############################################################################
import unittest
import numpy as np
import vigra

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume
from lazyflow.utility import Timer
from lazyflow.operators.opReorderAxes import OpReorderAxes
from lazyflow.request import Request, RequestPool
from functools import partial

from ilastik.applets.objectExtraction.opObjectExtraction import OpAdaptTimeListRoi, OpRegionFeatures, OpObjectExtraction

from ilastik.applets.objectExtraction.opObjectExtractionOptimized import OpAdaptTimeListRoi as OpAdaptTimeListRoiOpt
from ilastik.applets.objectExtraction.opObjectExtraction import OpRegionFeatures as OpRegionFeaturesOpt
from ilastik.applets.objectExtraction.opObjectExtraction import OpObjectExtraction as OpObjectExtractionOpt

from ilastik.plugins import pluginManager
from ilastik.applets.dataSelection.opDataSelection import OpDataSelection, DatasetInfo

from lazyflow.operators.ioOperators import OpStreamingHdf5Reader
import h5py

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import logging
logger = logging.getLogger(__name__)

NAME = "Standard Object Features"

FEATURES = {
    NAME : {
        "Count" : {},
        "RegionCenter" : {},
        "Coord<Principal<Kurtosis>>" : {},
        "Coord<Minimum>" : {},
        "Coord<Maximum>" : {},
    }
}

FILE_RAW = 'movie10.h5'
FILE_BINARY = 'movie10_Simple Segmentation.h5'

def binaryImage():
    img = np.zeros((2, 50, 50, 50, 1), dtype=np.float32)
    img[0,  0:10,  0:10,  0:10, 0] = 1
    img[0, 20:30, 20:30, 20:30, 0] = 1
    img[0, 40:45, 40:45, 40:45, 0] = 1

    img[1, 20:30, 20:30, 20:30, 0] = 1
    img[1, 5:10, 5:10, 0, 0] = 1
    img[1, 12:15, 12:15, 0, 0] = 1
    img = img.view(vigra.VigraArray)
    img.axistags = vigra.defaultAxistags('txyzc')

    return img

def rawImage():
    img = np.zeros((2, 50, 50, 50, 1), dtype=np.float32)
    img[0,  0:10,  0:10,  0:10, 0] = 200
    img[0, 20:30, 20:30, 20:30, 0] = 100

    # this object is further out than the margin and tests
    # regionCenter feature
    img[0, 40:45, 40:45, 40:45, 0] = 75

    img[1, 20:30, 20:30, 20:30, 0] = 50

    # this and next object are in each other's excl features
    img[1, 5:10, 5:10, 0, 0] = 25
    img[1, 12:15, 12:15, 0, 0] = 13
    img = img.view(vigra.VigraArray)
    img.axistags = vigra.defaultAxistags('txyzc')

    return img



class TestOpComparison(object):
    def setUp(self):
        binary_img = binaryImage()
        raw_img = rawImage()
        
        g = Graph()     
        
        self.h5FileRaw = h5py.File(FILE_RAW, 'r')        

        self.opReaderRaw = OpStreamingHdf5Reader(graph=g)
        self.opReaderRaw.Hdf5File.setValue(self.h5FileRaw)
        self.opReaderRaw.InternalPath.setValue('data')
         
        self.op5Raw = OpReorderAxes(graph=g)
        self.op5Raw.AxisOrder.setValue("txyzc")
        self.op5Raw.Input.connect(self.opReaderRaw.OutputImage)

        self.h5FileBinary = h5py.File(FILE_BINARY, 'r')

        self.opReaderBinary = OpStreamingHdf5Reader(graph=g)
        self.opReaderBinary.Hdf5File.setValue(self.h5FileBinary)
        self.opReaderBinary.InternalPath.setValue('exported_data')
         
        self.op5Binary = OpReorderAxes(graph=g)
        self.op5Binary.AxisOrder.setValue("txyzc")
        self.op5Binary.Input.connect(self.opReaderBinary.OutputImage)
        
        self.opLabel = OpLabelVolume(graph=g)
        #self.opLabel.Input.connect(self.opReaderBinary.OutputImage)
        self.opLabel.Input.connect(self.op5Binary.Output)
        
        # Object extraction operators
        self.op = OpRegionFeatures(graph=g)
        self.op.LabelVolume.connect(self.opLabel.Output)
        #self.op.RawVolume.connect(self.opReaderRaw.OutputImage)
        self.op.RawVolume.connect(self.op5Raw.Output)
        self.op.Features.setValue(FEATURES)

        self.opAdapt = OpAdaptTimeListRoi(graph=self.op.graph)
        self.opAdapt.Input.connect(self.op.Output)
        
        # Optimized object extraction operator
        self.opOpt = OpRegionFeaturesOpt(graph=g)
        self.opOpt.LabelVolume.connect(self.opLabel.Output)
        self.opOpt.RawVolume.connect(self.op5Raw.Output)
        self.opOpt.Features.setValue(FEATURES)

        self.opAdaptOpt = OpAdaptTimeListRoiOpt(graph=self.opOpt.graph)
        self.opAdaptOpt.Input.connect(self.opOpt.Output)

    def test_features(self):
        self.op.Output.fixed = False

        with Timer() as timerOp:
            featsOp = self.opAdapt.Output([]).wait()

        print "Op feature computation took: {} seconds".format(timerOp.seconds())
        
        with Timer() as timerOpOpt:
            featsOpOpt = self.opAdaptOpt.Output([]).wait()

        print "Op feature computation took: {} seconds".format(timerOpOpt.seconds())
    
        with Timer() as timerBasic:
            featsBasic = dict.fromkeys( range(self.op5Raw.Output.meta.shape[0]), None)
        
            pool = RequestPool()    
            for t in range(self.op5Raw.Output.meta.shape[0]):
                pool.add( Request( partial(self._computeFeatures, t, featsBasic) ) )
            pool.wait()
            
        print "Basic feature computation took: {} seconds".format(timerBasic.seconds())        
        

    def _computeFeatures(self, t, result):
        # Process entire spatial volume
        roi = [slice(None) for i in range(len(self.op5Raw.Output.meta.shape))]
        roi[0] = slice(t, t+1)
        roi = tuple(roi)

#         roiRaw = [slice(None) for i in range(len(self.opReaderRaw.OutputImage.meta.shape))]
#         roiRaw[0] = slice(t, t+1)
#         roiRaw = tuple(roiRaw)
#         
#         roiBinary = [slice(None) for i in range(len(self.opReaderBinary.OutputImage.meta.shape))]
#         roiBinary[0] = slice(t, t+1)
#         roiBinary = tuple(roiBinary)

        # Request in parallel
        image = self.op5Raw.Output(roi).wait()
        labels = self.opLabel.Output(roi).wait()
        #image = self.op5Raw.Output(roiRaw).wait()
        #labels = self.opLabel.Output(roiBinary).wait()
        
        features = ['Count', 'Coord<Minimum>', 'RegionCenter', 'Coord<Principal<Kurtosis>>', 'Coord<Maximum>']
        
        result[t] = vigra.analysis.extractRegionFeatures(image.squeeze().astype(np.float32), labels.squeeze().astype(np.uint32), features, ignoreLabel=0)
        
        
#     def test_table_export(self):
#         self.opAdapt = OpAdaptTimeListRoi(graph=self.op.graph)
#         self.opAdapt.Input.connect(self.op.Output)
# 
#         feats = self.opAdapt.Output([0, 1]).wait()
#         print "feature length:", len(feats)
#         OpObjectExtraction.createExportTable(feats)




if __name__ == '__main__':
    import sys
    import nose

    # Don't steal stdout. Show it on the console as usual.
    sys.argv.append("--nocapture")

    # Don't set the logging level to DEBUG. Leave it alone.
    sys.argv.append("--nologcapture")

    nose.run(defaultTest=__file__)
