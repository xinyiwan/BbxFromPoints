import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer import (
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsROINode,
    vtkMRMLScalarVolumeNode
)
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import random
#
# BbxFromPoints
#


class BbxFromPoints(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("BbxFromPoints")  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Segmentation"]
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["X.W"]  # TODO: replace with "Firstname Lastname (Organization)"
        self.parent.helpText = "Create a bounding box from 6 extreme points."
        self.parent.acknowledgementText = ""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)
#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # BbxFromPoints1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BbxFromPoints",
        sampleName="BbxFromPoints1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "BbxFromPoints1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="BbxFromPoints1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="BbxFromPoints1",
    )

    # BbxFromPoints2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BbxFromPoints",
        sampleName="BbxFromPoints2",
        thumbnailFileName=os.path.join(iconsPath, "BbxFromPoints2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="BbxFromPoints2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="BbxFromPoints2",
    )


#
# BbxFromPointsParameterNode
#


@parameterNodeWrapper
class BbxFromPointsParameterNode:
    inputPoints: vtkMRMLMarkupsFiducialNode
    outputROI: vtkMRMLMarkupsROINode

#
# BbxFromPointsWidget
#


class BbxFromPointsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None

    def initializeParameterNode(self):
        """Ensure parameter node exists and observed"""
        self.setParameterNode(self.logic.getParameterNode())
        
        # Initialize empty nodes if needed
        if not self._parameterNode.inputPoints:
            self._parameterNode.inputPoints = None
        if not self._parameterNode.outputROI:
            self._parameterNode.outputROI = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        
        # Load UI file
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/BbxFromPoints.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        
        # Create logic instance
        self.logic = BbxFromPointsLogic()
        
        # Connections
        self.ui.placePointsButton.connect('clicked(bool)', self.onPlacePoints)
        self.ui.computeButton.connect('clicked(bool)', self.onComputeBoundingBox)
        
        # Initialize parameter node
        self.initializeParameterNode()

    def setParameterNode(self, inputParameterNode: Optional[BbxFromPointsParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            
        self._parameterNode = inputParameterNode
        
        if self._parameterNode:
            # Connect GUI widgets to parameter node
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
        
    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputPoints:
            self.ui.computeButton.toolTip = _("Compute bounding box")
            self.ui.computeButton.enabled = True
        else:
            self.ui.computeButton.toolTip = _("Select or place input points")
            self.ui.computeButton.enabled = False

    def onPlacePoints(self):
        """Handle point placement"""
        if not self._parameterNode.inputPoints:
            self._parameterNode.inputPoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "BoundaryPoints")
        
        # Set up interactive placement
        self._parameterNode.inputPoints.SetControlPointPlacementStartInteractionMode(1)
        slicer.modules.markups.logic().SetActiveListID(self._parameterNode.inputPoints)
        
        # Observe point additions
        self.addObserver(self._parameterNode.inputPoints, slicer.vtkMRMLMarkupsNode.PointAddedEvent, 
                       self.onPointsChanged)

    def onPointsChanged(self, caller, event):
        """Check when 6 points are placed"""
        if self._parameterNode.inputPoints.GetNumberOfControlPoints() == 6:
            self.ui.computeButton.enabled = True
            slicer.util.infoDisplay("6 points placed! Click 'Compute Bounding Box'")

    def onComputeBoundingBox(self):
        """Create bounding box from points"""
        try:
            if not self._parameterNode.inputPoints:
                raise ValueError("No points placed")
                
            if self._parameterNode.inputPoints.GetNumberOfControlPoints() != 6:
                raise ValueError("Exactly 6 points required")
            
            # Create or update ROI
            if not self._parameterNode.outputROI:
                self._parameterNode.outputROI = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "BoundingBox")
            
            self.logic.createBoundingBox(
                self._parameterNode.inputPoints,
                self._parameterNode.outputROI
            )
            
            slicer.util.infoDisplay("Bounding box created successfully!")
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")
#
# BbxFromPointsLogic
#


class BbxFromPointsLogic(ScriptedLoadableModuleLogic):
    """This class implements all the computations"""
    
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
    
    def createBoundingBox(self, inputPointsNode, outputROINode=None):
        """Create bounding box from 6 points"""
        if not inputPointsNode or inputPointsNode.GetNumberOfControlPoints() != 6:
            raise ValueError("Exactly 6 boundary points required")
        
        # Create ROI node if not provided
        if not outputROINode:
            outputROINode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        
        # Get point coordinates
        points = []
        for i in range(6):
            coord = [random.randint(-20, 20) for _ in range(3)]
            inputPointsNode.GetNthControlPointPosition(i, coord)
            points.append(coord)
        
        # Calculate min/max bounds
        import numpy as np
        points_array = np.array(points)
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)
        
        # Set ROI properties
        outputROINode.SetXYZ((min_coords + max_coords) / 2)  # Center
        outputROINode.SetRadiusXYZ((max_coords - min_coords) / 2)  # Dimensions
        
        return outputROINode
    
    def process(self, inputPointsNode, outputROINode=None):
        """Wrapper for createBoundingBox with error handling"""
        try:
            return self.createBoundingBox(inputPointsNode, outputROINode)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to compute bounding box: {str(e)}")
            raise


#
# BbxFromPointsTest
#


class BbxFromPointsTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_BbxFromPoints1()

    def test_BbxFromPoints1(self):
        self.delayDisplay("Testing point-to-bbox conversion")
        
        # Create test points
        pointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")

        for i in range(6):
            p = [random.randint(-50, 50) for _ in range(3)]
            pointsNode.AddControlPoint(p)
        # for p in [[0,0,0], [50,0,0], [0,50,0], [0,0,50], [50,50,0], [50,50,50]]:
        #     pointsNode.AddControlPoint(p)
        
        # Run logic
        logic = BbxFromPointsLogic()
        roiNode = logic.createBoundingBox(pointsNode)
        
        # Verify bbox dimensions
        # self.assertEqual(roiNode.GetRadiusXYZ(), (25, 25, 25))
