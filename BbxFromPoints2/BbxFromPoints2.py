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
    vtkMRMLScalarVolumeNode,
    vtkMRMLMarkupsROINode
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
    scanDirectory: str = ''  # Directory containing sessions
    sessionDirectory: str = '' # Directory of sessions
    sessionList: list = [] # list of sessions
    currentSessionIndex: int = 0
    pointsNode: vtkMRMLMarkupsFiducialNode  # Selected points
    bboxNode: vtkMRMLMarkupsROINode  # Generated bounding box
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
        self.ui.clearButton.clicked.connect(self.onClearBbox)
        # self.ui.clearPtsButton.clicked.connect(self.onClearPoints)

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
                "pts"
            )
            self._parameterNode.pointsNode.CreateDefaultDisplayNodes()
        
        # if not self._parameterNode.bboxNode:
        # # Create empty segmentation node for bbox
        #     self._parameterNode.bboxNode = slicer.mrmlScene.AddNewNodeByClass(
        #         "vtkMRMLMarkupsROINode",
        #         "BoundingBox"
        # )        
            
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
        
        if self._parameterNode and self._parameterNode.pointsNode and self._parameterNode.pointsNode.GetNumberOfControlPoints() >= 6:
            self.ui.generateButton.toolTip = _("Generate Bbox")
            self.ui.generateButton.enabled = True
        else:
            self.ui.generateButton.toolTip = _("Select at least 6 points for the boundary")
            self.ui.generateButton.enabled = False
        
        if self._parameterNode.bboxNode and self._parameterNode.bboxNode != None:
            self.ui.clearButton.enabled = True
        else:
            self.ui.clearButton.enabled = False
         

    def onSelectDirectory(self):
        directory = qt.QFileDialog.getExistingDirectory()
        if directory:

            # get available sessions
            self._parameterNode.scanDirectory = directory

            # load sessions
            sessions = sorted([f for f in os.listdir(directory) if not f.startswith('.')])
            self._parameterNode.sessionList = sessions
            self._parameterNode.currentSessionIndex = 0

            self.switchSessionPreparation()

            # load current session Dir
            self.logic.loadScans(directory, sessions, self._parameterNode.currentSessionIndex)
            print(f'Current session: {sessions[self._parameterNode.currentSessionIndex]} in {directory}')

            self.loadCurrentSession()

    def loadCurrentSession(self):
        if not self._parameterNode.scanDirectory:
            return
            
        # Load image and segmentation
        imagePath, segPath = self.logic.getScanPaths(
            self._parameterNode.scanDirectory,
            self._parameterNode.sessionList,
            self._parameterNode.currentSessionIndex
        )
        
        for p in segPath:
            self._parameterNode.existingSegNode = slicer.util.loadSegmentation(p)
            # do not display the segmentation 
            displayNode = self._parameterNode.existingSegNode.GetDisplayNode()
            displayNode.SetVisibility(False)


        for p in imagePath:
            slicer.util.loadVolume(p)
        
        # Reset points and bbox
        # TODO if already annotated, keep the points and box
        if self._parameterNode.pointsNode:
            self._parameterNode.pointsNode.RemoveAllControlPoints()
        if self._parameterNode.bboxNode:
            slicer.mrmlScene.RemoveNode(self._parameterNode.bboxNode)

        if self.logic.checkTaskStatus(self._parameterNode.scanDirectory, self._parameterNode.sessionList, self._parameterNode.currentSessionIndex):
            slicer.util.infoDisplay("This task is already completed!")

        # Reset UI elements
        self.ui.statusLabel.text = "Status: Ready"
        self.ui.statusLabel.styleSheet = ""
        self.updateUI()
    
    def switchSessionPreparation(self):
        """Clean up all loaded segmentation data before switching sessions"""
        # Remove all loaded segmentation nodes
        seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for seg_node in seg_nodes:
            slicer.mrmlScene.RemoveNode(seg_node)

        # Clear the existingSegNode reference
        self._parameterNode.existingSegNode = None
        
        # Remove loaded volume nodes (optional - include if you want to clear volumes too)
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        for volume_node in volume_nodes:
            slicer.mrmlScene.RemoveNode(volume_node)
        
        # Clear points (but keep the points node)
        if self._parameterNode.pointsNode:
            self._parameterNode.pointsNode.RemoveAllControlPoints()
        
        # Reset bounding box
        if self._parameterNode.bboxNode:
            slicer.mrmlScene.RemoveNode(self._parameterNode.bboxNode)
            self._parameterNode.bboxNode = None
        
        # Update UI
        print(self._parameterNode.scanDirectory, self._parameterNode.sessionList, self._parameterNode.currentSessionIndex)
        self.ui.statusLabel.text = "Status: Loading new session..."
        self.ui.statusLabel.styleSheet = "color: blue"

        # load scans
        self.logic.loadScans(self._parameterNode.scanDirectory, self._parameterNode.sessionList, self._parameterNode.currentSessionIndex)

        # Force GUI update
        self.updateUI()

        
    def onNextScan(self):
        if self._parameterNode.currentSessionIndex < len(self._parameterNode.sessionList) - 1:
            self._parameterNode.currentSessionIndex += 1
            self.switchSessionPreparation()
            self.loadCurrentSession()
        else:
            slicer.util.infoDisplay("Now is the last session.")
            return
    
    def onPreviousScan(self):
        if self._parameterNode.currentSessionIndex > 0:
            self._parameterNode.currentSessionIndex -= 1
            self.switchSessionPreparation()
            self.loadCurrentSession()
        else:
            slicer.util.infoDisplay("Now is the first session.")
            return

    
    def onSave(self):

        if self.validateCurrentState():
            self.logic.saveResults(
                self._parameterNode.bboxNode,
                self._parameterNode.pointsNode,
                self._parameterNode.scanDirectory,
                self._parameterNode.sessionList,
                self._parameterNode.currentSessionIndex,
                self.ui.segScoreComboBox.currentText
            )
            slicer.util.infoDisplay("Results are saved successfully!")
            self.updateUI()

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
        # self.ui.clearPtsButton.enabled = count >= 6

        # Check if bboxNode exists
        self.ui.saveButton.enabled = self._parameterNode.bboxNode and self._parameterNode.bboxNode is not None
        self.ui.clearButton.enabled = self._parameterNode.bboxNode and self._parameterNode.bboxNode is not None
        
        # Scan info
        if self._parameterNode.scanDirectory:
            total = len(self._parameterNode.sessionList)
            self.ui.scanLabel.text = f"Scan {self._parameterNode.currentSessionIndex+1}/{total}"
        else:
            self.ui.scanLabel.text = "No scans loaded"
        
        # Task status
        if self.logic.checkTaskStatus(self._parameterNode.scanDirectory, self._parameterNode.sessionList, self._parameterNode.currentSessionIndex):
            self.ui.statusLabel.text = "This task is completed!"
            self.ui.statusLabel.styleSheet = "color: green"

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

    def onClearBbox(self):
        """Clear the existing bounding box"""
        try:
            if self._parameterNode.bboxNode == None:
                self.ui.statusLabel.text = "Status: There is no Bounding box"
                self.ui.statusLabel.styleSheet = "color: orange"
                self._parameterNode.bboxNode = None
                return
            slicer.mrmlScene.RemoveNode(self._parameterNode.bboxNode)
            self._parameterNode.bboxNode = None
            self.updateUI()
            self.ui.statusLabel.text = "Status: Bounding box removed"
            self.ui.statusLabel.styleSheet = "color: blue"
        except Exception as e:
            self.ui.statusLabel.text = f"Error: {str(e)}"
            self.ui.statusLabel.styleSheet = "color: red"

    def onClearPoints(self):
        """Clear the existing selected points"""
        try:
            if self._parameterNode.pointsNode == None:
                self.ui.statusLabel.text = "Status: There is no selected point."
                self.ui.statusLabel.styleSheet = "color: orange"
                self._parameterNode.pointsNode = None
                return
            self._parameterNode.pointsNode.RemoveAllControlPoints()
            self.updateUI()
            self.ui.statusLabel.text = "Status: Selected points are removed"
            self.ui.statusLabel.styleSheet = "color: blue"
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
        self.viewersLinked = True
        self._parameterNode = None

    def getParameterNode(self):
        if not self._parameterNode:
            # Create new parameter node with wrapper
            print('There is no node initialised, now initialize!')
            baseNode = super().getParameterNode()
            self._parameterNode = BbxFromPoints2ParameterNode(baseNode)
        return self._parameterNode
        
    def loadScans(self, directory, sessionList, index):
        self.scanDir = directory
        session_dir = directory + '/' + sessionList[index]
        self.imagePaths = sorted([
            f            
            for f in os.listdir(session_dir) 
            if f.endswith("nii.gz")
        ])
        print(self.imagePaths)

    def checkTaskStatus(self, directory, sessionList, index):
        import os, json
        status_file = os.path.join(directory, sessionList[index], 'status.json')
        # Check if file exists
        if not os.path.exists(status_file):
            return False

        try:
            # Read and parse the JSON file
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            # Check if status is 'completed'
            return status_data.get('status', '').lower() == 'completed'
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading status file: {str(e)}")
            return False

    def getScanPaths(self, directory, sessionList, index):
        
        session_dir = directory + '/' + sessionList[index]
        imageFile = [session_dir + '/' + f
                    for f in self.imagePaths
                    if 'seg' not in f]
        
        segFile = [session_dir + '/' + f
                    for f in self.imagePaths
                    if 'seg' in f]
        return imageFile, segFile
    
    def saveResults(self, bboxNode, pointsNode, directory, sessions, index, segScore):
        import csv
        import json
        import os
        from datetime import datetime

        ########################################################
        #    Result1: Save the boundingbox into segmentation          
        ########################################################

        # Get bounding box in WORLD (RAS) coordinates first
        center = [0.0, 0.0, 0.0]
        bboxNode.GetXYZ(center)
        radius = [0.0, 0.0, 0.0]
        bboxNode.GetRadiusXYZ(radius)  # Now passing the output parameter
        
        # Calculate all 8 corners of the box in RAS space
        k = 1
        ras_corners = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    ras_corners.append([
                        center[0] + x * radius[0] * k,
                        center[1] + y * radius[1] * k, 
                        center[2] + z * radius[2] * k
                    ])
        
        ras_face_centers = []
        for axis in [0, 1, 2]:
            for direction in [-1, 1]:
                point = [center[0], center[1], center[2]]
                point[axis] += direction * radius[axis]
                ras_face_centers.append(point)

        
        def rasToIjk(pointRAS, refvolumeNode):
            rasToIjkMatrix = vtk.vtkMatrix4x4()
            refvolumeNode.GetRASToIJKMatrix(rasToIjkMatrix)
            pointIJK = [0, 0, 0, 1]  # Homogeneous coordinates
            rasToIjkMatrix.MultiplyPoint(tuple(pointRAS) + (1,), pointIJK)
            return [int(pointIJK[0]), int(pointIJK[1]), int(pointIJK[2])]
    
        def ras_to_lps(ras_coords):
            lps_coords = ras_coords.copy()
            lps_coords[0] *= -1  # R -> L
            lps_coords[1] *= -1  # A -> P
            return lps_coords

        def lps_to_ras(lps_coords):
            ras_coords = lps_coords.copy()
            ras_coords[0] *= -1  # L -> R
            ras_coords[1] *= -1  # P -> A
            return ras_coords
        
        def saveBboxForImage(ref_volume, volume_name):

            # Get reference volume geometry
            refImage = ref_volume
            # print(ref_volume)
            refDim = refImage.GetImageData().GetDimensions()
            # print(refDim)
            refSpacing = np.array(refImage.GetSpacing())  # [sx, sy, sz]
            # print(refSpacing)
            refOrigin = np.array(refImage.GetOrigin())    # RAS origin
            # print(refOrigin)

            # Convert points to IJK space
            ijk_points = []

            # use the points from bbox (corners)
            for i in range(len(ras_corners)):
                ijk_points.append(rasToIjk(ras_corners[i], ref_volume))

            # # choose not to use the original points but from bbx
            # for i in range(pointsNode.GetNumberOfControlPoints()):
            #     ras_point = [0, 0, 0]
            #     pointsNode.GetNthControlPointPosition(i, ras_point)
            #     # Convert RAS → IJK (accounting for spacing and origin)
            #     ijk_points.append(rasToIjk(ras_point, ref_volume))
            
            # Calculate grid-aligned bounds
            ijk_array = np.array(ijk_points)
            ijk_min = np.floor(np.min(ijk_array, axis=0)).astype(int)
            ijk_max = np.ceil(np.max(ijk_array, axis=0)).astype(int)      

            # ijk_min = np.min(ijk_array, axis=0).astype(int)
            # ijk_max = np.max(ijk_array, axis=0).astype(int) 

            ijk_min = np.clip(ijk_min, 0, np.array(refDim)-1)
            ijk_max = np.clip(ijk_max, 0, np.array(refDim)-1) 

            # Create mask (Z,Y,X order)
            mask_array = np.zeros((refDim[2], refDim[1], refDim[0]), dtype=np.uint8)
            mask_array[
                ijk_min[2]:ijk_max[2]+1,
                ijk_min[1]:ijk_max[1]+1,
                ijk_min[0]:ijk_max[0]+1
            ] = 1

            # Save mask
            mask_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            mask_volume.SetName(f'{volume_name}_bbx_temp')

            # Apply reference geometry (spacing + origin + directions)
            ref_ijk_to_ras = vtk.vtkMatrix4x4()
            ref_volume.GetIJKToRASMatrix(ref_ijk_to_ras)
            mask_volume.SetIJKToRASMatrix(ref_ijk_to_ras)  # Do NOT set spacing/origin separately!

            # Update volume from array (now in X,Y,Z order)
            slicer.util.updateVolumeFromArray(mask_volume, mask_array)
            
            # Check if segmentation with same name already exists and remove it
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            ref_volume_item = shNode.GetItemByDataNode(ref_volume)

            # Get all child items of the reference volume
            childIds = vtk.vtkIdList()
            shNode.GetItemChildren(ref_volume_item, childIds)

            # Find and remove existing segmentation with the same name
            for itemIdIndex in range(childIds.GetNumberOfIds()):
                shItemId = childIds.GetId(itemIdIndex)
                dataNode = shNode.GetItemDataNode(shItemId)
                if dataNode and dataNode.IsA("vtkMRMLSegmentationNode"):
                    slicer.mrmlScene.RemoveNode(dataNode)
            
            # Create new segmentation
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentation_node.SetName(f"{volume_name}_bbox")
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                mask_volume, 
                segmentation_node)
            
            # Set up hierarchy (put segmentation under reference volume)
            seg_item = shNode.GetItemByDataNode(segmentation_node)
            shNode.SetItemParent(seg_item, ref_volume_item)

            # Configure display
            seg_display = segmentation_node.GetDisplayNode()
            if not seg_display:
                segmentation_node.CreateDefaultDisplayNodes()
                seg_display = segmentation_node.GetDisplayNode()

            # Set display properties
            seg_display.SetVisibility(False)  # Make sure it's visible
            seg_display.SetOpacity(0.5)      # Fully opaque
            seg_display.SetAllSegmentsVisibility2DFill(False)  # Turn off fill
            seg_display.SetOpacity2DOutline(True)
            seg_display.SetAllSegmentsVisibility2DOutline(True)

            # Customize outline appearance
            seg_display.SetSelectedColor(1,0,0)
            seg_display.SetVisibility2D(True)
            seg_display.SetVisibility3D(False)  # No 3D display

            # Save and cleanup
            output_path = os.path.join(directory, sessions[index], f"{volume_name}_seg_bbox.nii.gz")
            slicer.util.saveNode(mask_volume, output_path)
            slicer.mrmlScene.RemoveNode(mask_volume)
            slicer.mrmlScene.RemoveNode(segmentation_node)

            # Load the saved segmentation and return it
            loaded_seg = slicer.util.loadSegmentation(output_path)

            # Set up hierarchy for loaded segmentation
            loaded_seg_item = shNode.GetItemByDataNode(loaded_seg)
            shNode.SetItemParent(loaded_seg_item, ref_volume_item)

            # Configure display for loaded segmentation
            loaded_display = loaded_seg.GetDisplayNode()
            if not loaded_display:
                loaded_seg.CreateDefaultDisplayNodes()
                loaded_display = loaded_seg.GetDisplayNode()
            
            loaded_display.SetVisibility(False)  # Keep hidden by default
            loaded_display.SetOpacity(0.5)
            loaded_display.SetSelectedColor(1, 0, 0)

        # use every image as reference volume and save bbox accordingly
        allVolumeNodes = slicer.util.getNodes('vtkMRMLScalarVolumeNode*')
        for volume_k in allVolumeNodes.keys():
            v_name = volume_k
            # if v_name == '901-T1_TSE_TRA':
            saveBboxForImage(allVolumeNodes[v_name], v_name)

        ########################################################
        #    Result2: Save score for bone annotations        
        ########################################################

        # Save scores
            csvPath = os.path.join(directory, sessions[index], 'scores.csv') 
            with open(csvPath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "SegScore"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    segScore
                ])

        ########################################################
        #    Result3: save task status, coordiates of points            
        ########################################################

        # save markup fiducials into the json file
        outputFilePath = os.path.join(directory, sessions[index], 'status.json')
        data = {'status': 'completed'}
        for fidIndex in range(pointsNode.GetNumberOfControlPoints()):
            coords=[0,0,0]
            pointsNode.GetNthControlPointPosition(fidIndex,coords)
            data[f'pts_{fidIndex}'] = {}
            data[f'pts_{fidIndex}']['coords'] = coords

        with open(outputFilePath, 'w') as outfile:
            json.dump(data, outfile)

        print('Selected points and bboxes are saved!')

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

        if not self._parameterNode.bboxNode:
        # Create empty segmentation node for bbox
            self._parameterNode.bboxNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsROINode",
                "BoundingBox"
        )

        # Get points array
        points = []
        for i in range(pointsNode.GetNumberOfControlPoints()):
            point = [0.0, 0.0, 0.0]
            pointsNode.GetNthControlPointPosition(i, point)
            points.append(point)

        # TODO TRY -> ] points into RAS 

        # Calculate bounding box dimensions (RAS)
        points_array = np.array(points)
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)

        bboxNode = self._parameterNode.bboxNode
        bboxNode.SetXYZ(
            (min_coords[0] + max_coords[0])/2,
            (min_coords[1] + max_coords[1])/2,
            (min_coords[2] + max_coords[2])/2
        )

        # default expansion 
        k = 1.1
        # expand radius 5% for every dimention
        bboxNode.SetRadiusXYZ(
            (max_coords[0] - min_coords[0])/2*k,
            (max_coords[1] - min_coords[1])/2*k,
            (max_coords[2] - min_coords[2])/2*k
        )

        # Get or create display node of bbox
        displayNode = bboxNode.GetDisplayNode()
        if not displayNode:
            bboxNode.CreateDefaultDisplayNodes()
            displayNode = bboxNode.GetDisplayNode()

        # Set display properties
        displayNode.SetVisibility(True)  # Make sure it's visible
        displayNode.SetOpacity(1.0)      # Fully opaque
        displayNode.SetFillVisibility(False)  # Turn off fill
        displayNode.SetOutlineVisibility(True)  # Turn on outline

        # Customize outline appearance
        displayNode.SetOutlineOpacity(1.0)    # Fully opaque outline
        displayNode.SetSelectedColor(0,1,0)
        
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
