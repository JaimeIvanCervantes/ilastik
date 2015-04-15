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

from lazyflow.graph import Graph
from ilastik.workflow import Workflow
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.tracking.structured.structuredTrackingApplet import StructuredTrackingApplet
from ilastik.applets.objectExtraction.objectExtractionApplet import ObjectExtractionApplet
from ilastik.applets.cropThresholdTwoLevels.cropThresholdTwoLevelsApplet import CropThresholdTwoLevelsApplet
from ilastik.applets.cropping.cropSelectionApplet import CropSelectionApplet

from lazyflow.operators.opReorderAxes import OpReorderAxes
from ilastik.applets.tracking.base.trackingBaseDataExportApplet import TrackingBaseDataExportApplet

class StructuredTrackingWorkflow( Workflow ):
    workflowName = "Structured Learning Tracking Workflow"
    workflowDisplayName = "Structured Learning Tracking Workflow [Inputs: Raw Data, Pixel Prediction Map]"
    workflowDescription = "Structured learning tracking of objects, based on Prediction Maps or (binary) Segmentation Images"

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName

    def __init__( self, shell, headless, workflow_cmdline_args, project_creation_args, *args, **kwargs ):
        graph = kwargs['graph'] if 'graph' in kwargs else Graph()
        if 'graph' in kwargs: del kwargs['graph']
        super(StructuredTrackingWorkflow, self).__init__(shell, headless, workflow_cmdline_args, project_creation_args, graph=graph, *args, **kwargs)

        data_instructions = 'Use the "Raw Data" tab to load your intensity image(s).\n\n'\
                            'Use the "Prediction Maps" tab to load your pixel-wise probability image(s).'
        ## Create applets
        self.dataSelectionApplet = DataSelectionApplet(self,"Input Data","Input Data",batchDataGui=False,force5d=True,instructionText=data_instructions,max_lanes=1)
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        opDataSelection.DatasetRoles.setValue( ['Raw Data', 'Prediction Maps'] )

        self.cropSelectionApplet = CropSelectionApplet(self,"Crop Selection","CropSelection")

        self.cropThresholdTwoLevelsApplet = CropThresholdTwoLevelsApplet( self,"Threshold and Size Filter for a Crop","CropThresholdTwoLevels" )

        self.objectExtractionApplet = ObjectExtractionApplet(name="Object Feature Computation",workflow=self, interactive=False)

        self.trackingApplet = StructuredTrackingApplet( workflow=self )
        opTracking = self.trackingApplet.topLevelOperator

        self.dataExportApplet = TrackingBaseDataExportApplet(self, "Tracking Result Export")
        opDataExport = self.dataExportApplet.topLevelOperator
        opDataExport.SelectionNames.setValue( ['Structured Learning Tracking', 'Object Identities'] )
        opDataExport.WorkingDirectory.connect( opDataSelection.WorkingDirectory )

        self._applets = []
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.cropSelectionApplet)
        self._applets.append(self.cropThresholdTwoLevelsApplet)
        self._applets.append(self.objectExtractionApplet)
        self._applets.append(self.trackingApplet)
        self._applets.append(self.dataExportApplet)

    def connectLane(self, laneIndex):
        opData = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opObjExtraction = self.objectExtractionApplet.topLevelOperator.getLane(laneIndex)
        opTracking = self.trackingApplet.topLevelOperator.getLane(laneIndex)
        opTwoLevelThreshold = self.cropThresholdTwoLevelsApplet.topLevelOperator.getLane(laneIndex)
        opDataExport = self.dataExportApplet.topLevelOperator.getLane(laneIndex)

        opCropSelection = self.cropSelectionApplet.topLevelOperator.getLane(laneIndex)

        ## Connect operators ##
        op5Raw = OpReorderAxes(parent=self)
        op5Raw.AxisOrder.setValue("txyzc")
        op5Raw.Input.connect(opData.ImageGroup[0])

        opCropSelection.InputImage.connect( opData.ImageGroup[0] )
        opCropSelection.PredictionImage.connect( opData.ImageGroup[1] )

        opTwoLevelThreshold.InputImage.connect( opData.ImageGroup[1] )
        opTwoLevelThreshold.RawInput.connect( opData.ImageGroup[0] ) # Used for display only
        opTwoLevelThreshold.Crops.connect( opCropSelection.Crops)


        # Use OpReorderAxis for both input datasets such that they are guaranteed to
        # have the same axis order after thresholding
        op5Binary = OpReorderAxes( parent=self )
        op5Binary.AxisOrder.setValue("txyzc")
        op5Binary.Input.connect( opTwoLevelThreshold.CachedOutput )

        opObjExtraction.RawImage.connect( op5Raw.Output )
        opObjExtraction.BinaryImage.connect( op5Binary.Output )

        opTracking.RawImage.connect( op5Raw.Output )
        opTracking.BinaryImage.connect( op5Binary.Output )
        opTracking.LabelImage.connect( opObjExtraction.LabelImage )
        opTracking.ObjectFeatures.connect( opObjExtraction.RegionFeatures )

        opDataExport.Inputs.resize(2)
        opDataExport.Inputs[0].connect( opTracking.TrackImage )
        opDataExport.Inputs[1].connect( opTracking.LabelImage )
        opDataExport.RawData.connect( op5Raw.Output )
        opDataExport.RawDatasetInfo.connect( opData.DatasetGroup[0] )

    def _inputReady(self, nRoles):
        slot = self.dataSelectionApplet.topLevelOperator.ImageGroup
        if len(slot) > 0:
            input_ready = True
            for sub in slot:
                input_ready = input_ready and \
                    all([sub[i].ready() for i in range(nRoles)])
        else:
            input_ready = False

        return input_ready

    def handleAppletStateUpdateRequested(self):
        """
        Overridden from Workflow base class
        Called when an applet has fired the :py:attr:`Applet.statusUpdateSignal`
        """
        # If no data, nothing else is ready.
        input_ready = self._inputReady(2) and not self.dataSelectionApplet.busy

        opCropSelection = self.cropSelectionApplet.topLevelOperator
        croppingOutput = opCropSelection.Crops
        cropping_ready = input_ready and \
                            len(croppingOutput) > 0

        opThresholding = self.cropThresholdTwoLevelsApplet.topLevelOperator
        thresholdingOutput = opThresholding.CachedOutput
        thresholding_ready = cropping_ready and \
                       len(thresholdingOutput) > 0

        opObjectExtraction = self.objectExtractionApplet.topLevelOperator
        objectExtractionOutput = opObjectExtraction.ComputedFeatureNames
        features_ready = thresholding_ready and \
                         len(objectExtractionOutput) > 0

        opTracking = self.trackingApplet.topLevelOperator
        tracking_ready = features_ready and \
                           len(opTracking.Labels) > 0 and \
                           opTracking.Labels.ready() and \
                           opTracking.TrackImage.ready()

        busy = False
        busy |= self.dataSelectionApplet.busy
        busy |= self.cropSelectionApplet.busy
        busy |= self.dataExportApplet.busy
        busy |= self.trackingApplet.busy
        self._shell.enableProjectChanges( not busy )

        self._shell.setAppletEnabled(self.dataSelectionApplet, not busy)
        self._shell.setAppletEnabled(self.cropSelectionApplet, input_ready and not busy)
        self._shell.setAppletEnabled(self.cropThresholdTwoLevelsApplet, input_ready and not busy)
        self._shell.setAppletEnabled(self.objectExtractionApplet, thresholding_ready and not busy)
        self._shell.setAppletEnabled(self.trackingApplet, features_ready and not busy)
        self._shell.setAppletEnabled(self.dataExportApplet, tracking_ready and not busy and \
                                        self.dataExportApplet.topLevelOperator.Inputs[0][0].ready() )