import logging
import os
from typing import Annotated, Optional
import vtk
import qt
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from slicer import (
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLSegmentationNode,
    vtkMRMLScalarVolumeNode
)
import csv
import numpy as np

#
# BbxFromPoints2
#

class BbxFromPoints2(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("BbxFromPoints2") 
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = [] 
        self.parent.contributors = ["X Wan (EMC)"]  
        self.parent.helpText = _("""
            This is an example of scripted loadable module bundled in an extension.
            See more information in <a href="https://github.com/organization/projectname#BbxFromPoints2">module documentation</a>.
            """)
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
            This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
            and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
            """)

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

    # BbxFromPoints21
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BbxFromPoints2",
        sampleName="BbxFromPoints21",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "BbxFromPoints21.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="BbxFromPoints21.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="BbxFromPoints21",
    )

    # BbxFromPoints22
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BbxFromPoints2",
        sampleName="BbxFromPoints22",
        thumbnailFileName=os.path.join(iconsPath, "BbxFromPoints22.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="BbxFromPoints22.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="BbxFromPoints22",
    )


#
# BbxFromPoints2ParameterNode
#

@parameterNodeWrapper
class BbxFromPoints2ParameterNode:
    """
    Custom parameters for bounding box workflow
    """
    scanDirectory: str  # Directory containing image_X/seg_X files
    currentScanIndex: int = 0
    pointsNode: vtkMRMLMarkupsFiducialNode  # Selected points
    bboxNode: vtkMRMLSegmentationNode  # Generated bounding box
    scoreBbox: str = "unrated"  # poor/sufficient/good
    scoreExistingSeg: str = "unrated"
    existingSegNode: vtkMRMLSegmentationNode  # Loaded segmentation

#
# BbxFromPoints2Widget
#


class BbxFromPoints2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = BbxFromPoints2Logic()
        self._parameterNode = None
        self._pointsNodeObserverTag = None
        self._parameterNodeGuiTag = None
        
        
        

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/BbxFromPoints2.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connections / buttons
        self.setupConnections()
        
        # Initialize parameter node after UI setup
        self.initializeParameterNode()

        # Add observer to points node
        if self._parameterNode.pointsNode:
            self._pointsNodeObserverTag = self._parameterNode.pointsNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent, 
                self.onPointsNodeChanged
            )

        # Update UI
        self.updateUI()

    
    def setupConnections(self):
        """Connect UI signals to slots."""
        self.ui.selectDirectoryButton.clicked.connect(self.onSelectDirectory)
        self.ui.previousButton.clicked.connect(self.onPreviousScan)
        self.ui.nextButton.clicked.connect(self.onNextScan)
        self.ui.saveButton.clicked.connect(self.onSave)
        self.ui.generateButton.clicked.connect(self.onGenerateBbox)
        # self.pointsPlaceWidget.currentNodeChanged.connect(self.onPointsNodeChanged)

    def initializeParameterNode(self):
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self._parameterNode = self.logic.getParameterNode()
        
        # Initialize nodes with scene association
        if not self._parameterNode.pointsNode:
            # Create new points node if none exists
            self._parameterNode.pointsNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", 
                "BbxPoints"
            )
            self._parameterNode.pointsNode.CreateDefaultDisplayNodes()
        
        if not self._parameterNode.bboxNode:
        # Create empty segmentation node for bbox
            self._parameterNode.bboxNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode",
                "BoundingBox"
        )
            
    def setParameterNode(self, inputParameterNode: Optional[BbxFromPoints2ParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        # TODO check the number of points
        if self._parameterNode and self._parameterNode.pointsNode and self._parameterNode.pointsNode.GetNumberOfControlPoints() >= 6:
            self.ui.generateButton.toolTip = _("Generate Bbox")
            self.ui.generateButton.enabled = True
        else:
            self.ui.generateButton.toolTip = _("Select at least 6 points for the boundary")
            self.ui.generateButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "generate" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.createBoundingBox(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)


    def onSelectDirectory(self):
        directory = qt.QFileDialog.getExistingDirectory()
        if directory:
            self.logic.loadScans(directory)
            print(directory)
            self._parameterNode.scanDirectory = directory
            self._parameterNode.currentScanIndex = 0
            self.loadCurrentScan()

    def loadCurrentScan(self):
        if not self._parameterNode.scanDirectory:
            return
            
        # Load image and segmentation
        imagePath, segPath = self.logic.getScanPaths(
            self._parameterNode.scanDirectory, 
            self._parameterNode.currentScanIndex
        )
        
        self._parameterNode.existingSegNode = slicer.util.loadSegmentation(segPath)
        slicer.util.loadVolume(imagePath)
        
        # Reset points and bbox
        if self._parameterNode.pointsNode:
            self._parameterNode.pointsNode.RemoveAllControlPoints()
        if self._parameterNode.bboxNode:
            slicer.mrmlScene.RemoveNode(self._parameterNode.bboxNode)

        # Reset UI elements
        self.ui.statusLabel.text = "Status: Ready"
        self.ui.statusLabel.styleSheet = ""
        self.updateUI()

    def onNextScan(self):
        if self.validateCurrentState():
            self._parameterNode.currentScanIndex += 1
            self.loadCurrentScan()
    
    def onPreviousScan(self):
        if self._parameterNode.currentScanIndex > 0:
            self._parameterNode.currentScanIndex -= 1
            self.loadCurrentScan()
    
    def onSave(self):
        if self.validateCurrentState():
            self.logic.saveResults(
                self._parameterNode.bboxNode,
                self._parameterNode.scanDirectory,
                self._parameterNode.currentScanIndex,
                self.ui.bboxScoreComboBox.currentText,
                self.ui.segScoreComboBox.currentText
            )
            slicer.util.infoDisplay("Results saved successfully")

    def validateCurrentState(self):
        valid = True
        if not self._parameterNode.pointsNode:
            valid = False
        elif self._parameterNode.pointsNode.GetNumberOfControlPoints() < 6:
            valid = False
        return valid
    
    def updateUI(self):

        # Points counter
        count = self._parameterNode.pointsNode.GetNumberOfControlPoints() if self._parameterNode.pointsNode else 0
        self.ui.pointsCountLabel.text = f"Points placed: {count}/6"
        self.ui.pointsCountLabel.styleSheet = "color: green" if count >=6 else "color: red"

        # Button states
        self.ui.generateButton.enabled = count >= 6
        self.ui.saveButton.enabled = self._parameterNode.bboxNode is not None

        # Scan info
        if self._parameterNode.scanDirectory:
            total = len(self.logic.scanPairs)
            self.ui.scanLabel.text = f"Scan {self._parameterNode.currentScanIndex+1}/{total}"
        else:
            self.ui.scanLabel.text = "No scans loaded"
    
    def onPointsNodeChanged(self, caller, event):
        """Handle points node changes"""
        self.updateUI()

    def onGenerateBbox(self):
        """Handle bounding box generation"""
        try:
            self._parameterNode.bboxNode = self.logic.createBoundingBox(self._parameterNode.pointsNode)
            self.updateUI()
            self.ui.statusLabel.text = "Status: Bounding box generated"
            self.ui.statusLabel.styleSheet = "color: green"
        except Exception as e:
            self.ui.statusLabel.text = f"Error: {str(e)}"
            self.ui.statusLabel.styleSheet = "color: red"
    
    def cleanup(self):
        """Cleanup observations"""
        self.removeObservers()
    
    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
    
    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
    
    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)
    
    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()
    

#
# BbxFromPoints2Logic
#

class BbxFromPoints2Logic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.viewersLinked = False
        self._parameterNode = None

    def getParameterNode(self):
        if not self._parameterNode:
            # Create new parameter node with wrapper
            print('There is no node initialised, now initialize!')
            baseNode = super().getParameterNode()
            self._parameterNode = BbxFromPoints2ParameterNode(baseNode)
        return self._parameterNode
        
    def loadScans(self, directory):
        self.scanDir = directory
        self.scanPairs = sorted([
            (f, f.replace("image_", "seg_")) 
            for f in os.listdir(directory) 
            if f.startswith("image_")
        ])
        print(self.scanPairs)

    def getScanPaths(self, directory, index):
        imageFile, segFile = self.scanPairs[index]
        return (
            os.path.join(directory, imageFile),
            os.path.join(directory, segFile)
        )
    
    def saveResults(self, bboxNode, directory, index, bboxScore, segScore):
        # Save segmentation
        outputPath = os.path.join(directory, f"seg_{index}_bbox.nii.gz")
        slicer.util.saveNode(bboxNode, outputPath)
        
        # Save scores
        csvPath = os.path.join(directory, "scores.csv")
        with open(csvPath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([index, bboxScore, segScore])
    
    def linkViewers(self):
        if not self.viewersLinked:
            for view in ["Red", "Green", "Yellow"]:
                widget = slicer.app.layoutManager().sliceWidget(view)
                widget.sliceController().setLinkedControl(True)
            self.viewersLinked = True
    
    def createBoundingBox(self, pointsNode):
        """Generate segmentation from markup points"""
        if not pointsNode or pointsNode.GetNumberOfControlPoints() < 6:
            raise ValueError("At least 6 points required")
            
        # Get points array
        points = []
        for i in range(pointsNode.GetNumberOfControlPoints()):
            point = [0.0, 0.0, 0.0]
            pointsNode.GetNthControlPointPosition(i, point)
            points.append(point)
        
        # Calculate bounding box dimensions
        
        points_array = np.array(points)
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)

        # Create segmentation node and cube representation
        bboxNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', "BoundingBox")
        bboxNode.CreateDefaultDisplayNodes()
        
        cube = vtk.vtkCubeSource()
        cube.SetCenter(
            (min_coords[0] + max_coords[0])/2,
            (min_coords[1] + max_coords[1])/2,
            (min_coords[2] + max_coords[2])/2
        )
        cube.SetXLength(max_coords[0] - min_coords[0])
        cube.SetYLength(max_coords[1] - min_coords[1])
        cube.SetZLength(max_coords[2] - min_coords[2])
        
        # Convert to closed surface
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(cube.GetOutputPort())
        triangleFilter.Update()

        # Add to segmentation
        segmentation = bboxNode.GetSegmentation()
        segment = slicer.vtkSegment()
        segment.SetName("BoundingBox")
        segment.AddRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName(),
            triangleFilter.GetOutput()
        )
        segmentation.AddSegment(segment)
        
        return bboxNode


#
# BbxFromPoints2Test
#


class BbxFromPoints2Test(ScriptedLoadableModuleTest):
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
        self.test_BbxFromPoints21()

    def test_BbxFromPoints21(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("BbxFromPoints21")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = BbxFromPoints2Logic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
