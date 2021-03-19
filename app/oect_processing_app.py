import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import pyqtgraph as pg
from ..oect import *
from ..oect_utils.oect_plot import *
from ..oect_utils.oect_load import uC_scale
import os

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class getExistingDirectories(QFileDialog):
    '''
    Work around, since QFileDialog has no built in way to select multiple directories.
    https://stackoverflow.com/questions/18707261/how-to-select-multiple-directories-with-kfiledialog

    if the user closes the window, the path selection is set to the last directory
    '''

    def __init__(self, *args):
        super(getExistingDirectories, self).__init__(*args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.Directory)
        self.setOption(self.ShowDirsOnly, True)
        self.findChildren(QListView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.findChildren(QTreeView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.exec()


class MainWindow(QMainWindow):
    MAX_PLOTS = 5

    def __init__(self, *args, **kwargs):
        '''
        Set up main window. Two largest components are the menuLayout for the sidebar to the left,
        and graphsLayout for the graphs to the right.
        '''
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('OECT Processing')
        self.mainLayout = QHBoxLayout()  # overarching window layout

        # setup scrollarea that will contain listwidgets for each parent folder
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidget = QWidget()
        self.scrollArea.setWidget(self.scrollAreaWidget)
        self.scrollAreaContents = QVBoxLayout()
        self.scrollAreaWidget.setLayout(self.scrollAreaContents)

        # layout for the left sidebar
        self.menuLayout = QGridLayout()
        self.loadPushbutton = QPushButton('Load parent folder')
        self.averageCheckBox = QCheckBox('Average devices with the same WdL')
        self.analyzePushButton = QPushButton('Analyze')
        self.menuLayout.addWidget(self.loadPushbutton, 0, 0)
        self.menuLayout.addWidget(self.scrollArea, 1, 0, 1, 2)
        self.menuLayout.addWidget(self.averageCheckBox, 2, 0)
        self.menuLayout.addWidget(self.analyzePushButton, 3, 0)

        self.menuLayout.setColumnStretch(0, 1)
        self.menuLayout.setColumnStretch(1, 1)

        self.textBrowser = QTextBrowser()  # text browser displaying info
        self.addPlaceHolder(self.menuLayout, [4, 0])
        self.menuLayout.addWidget(self.textBrowser, 5, 0, 1, 2)
        self.menuLayout.setRowStretch(0, 1)
        self.menuLayout.setRowStretch(1, 10)
        self.menuLayout.setRowStretch(2, 1)
        self.menuLayout.setRowStretch(3, 1)
        self.menuLayout.setRowStretch(4, 1)
        self.menuLayout.setRowStretch(5, 5)

        self.linearPlotContainer = QFrame()
        self.linearPlotContainer.setLayout(QGridLayout())
        self.logPlotContainer = QFrame()
        self.logPlotContainer.setLayout(QGridLayout())
        self.graphsLayout = QVBoxLayout()

        self.graphsLayout.addWidget(self.linearPlotContainer)
        self.linearPlotLayout = pg.GraphicsLayoutWidget()
        self.linearPlotContainer.layout().addWidget(self.linearPlotLayout)
        self.linearPlot = self.linearPlotLayout.addPlot()
        self.linearPlot.setLabel('bottom', 'Wd/L * (Vg-Vt) (cm*V)')
        self.linearPlot.setLabel('left', 'gm (mS)')
        self.linearPlot.addLegend()

        self.graphsLayout.addWidget(self.logPlotContainer)
        self.logPlotLayout = pg.GraphicsLayoutWidget()
        self.logPlotContainer.layout().addWidget(self.logPlotLayout)
        self.logPlot = self.logPlotLayout.addPlot()
        self.logPlot.setLabel('bottom', 'Wd/L * (Vg-Vt) (cm*V)')
        self.logPlot.setLabel('left', 'gm (mS)')
        self.logPlot.setLogMode(True, True)

        self.mainLayout.addLayout(self.menuLayout)
        self.mainLayout.addLayout(self.graphsLayout)
        self.mainLayout.setStretchFactor(self.menuLayout, 2)
        self.mainLayout.setStretchFactor(self.graphsLayout, 4)

        widget = QWidget()
        widget.setLayout(self.mainLayout)
        self.setCentralWidget(widget)

        # setup ui signals
        self.loadPushbutton.clicked.connect(self.open_file)
        self.analyzePushButton.clicked.connect(self.analyze)

    def open_file(self):
        '''
        Prompt user to select parent folders and set up layout according to selection.
        If user closes file dialog, last directory is returned
        '''
        fileDialog = getExistingDirectories()
        selectedPaths = fileDialog.selectedFiles()

        # setup layout, after adding widgets
        for dirPath in selectedPaths:
            groupBox = QGroupBox()  # each parent folder has a groupbox
            groupBox.setMinimumHeight(500)
            groupBoxLayout = QGridLayout()
            groupBox.setLayout(groupBoxLayout)
            self.scrollAreaContents.addWidget(groupBox)

            listWidget = QListWidget()  # list widget showing subfolders
            listWidget.setWordWrap(True)
            listWidget.setSelectionMode(QAbstractItemView.MultiSelection)

            pathLabel = QLabel(dirPath)
            pathLabel.setStyleSheet('font-weight: bold')
            pathLabel.setWordWrap(True)
            groupBoxLayout.addWidget(pathLabel, 0, 0)

            removeFolderButton = QPushButton('X')  # button to delete parent folder
            removeFolderButton.setMaximumWidth(50)
            removeFolderButton.setStyleSheet('QPushButton {color: red;}')
            groupBoxLayout.addWidget(removeFolderButton, 0, 1)
            removeFolderButton.clicked.connect(self.deleteGroupBox)

            groupBoxLayout.addWidget(listWidget, 1, 0, 1, 2)

            subfolders = [f.path for f in os.scandir(dirPath) if f.is_dir()]
            for folder in subfolders:
                itemWidget = QWidget()
                itemLayout = QVBoxLayout()
                itemWidget.setLayout(itemLayout)

                itemLayout.addWidget(QLabel(os.path.basename(folder)))
                listWidget.addItem('')
                lastItem = listWidget.item(listWidget.count() - 1)

                params = None
                config = None

                # get parameters from config file
                for file in os.listdir(folder):
                    fullPath = os.path.join(folder, file)
                    if (fullPath.endswith('.cfg')):
                        config = fullPath
                    # params, opts = config_file(config)
                if config == None:
                    config = make_config(folder)
                params, opts = config_file(config)

                # if params loaded, then set up spinboxes with W and L
                if params:
                    width = params['W']
                    length = params['L']

                    dimensionWidget = QWidget()
                    dimensionLayout = QHBoxLayout()
                    dimensionWidget.setLayout(dimensionLayout)
                    dimensionLayout.addWidget(QLabel('Width (um)'))
                    widthSpinbox = QSpinBox()
                    widthSpinbox.setMaximum(9999)
                    widthSpinbox.setValue(width)
                    dimensionLayout.addWidget(widthSpinbox)
                    dimensionLayout.addWidget(QLabel('Length (um)'))
                    lengthSpinbox = QSpinBox()
                    lengthSpinbox.setMaximum(9999)
                    lengthSpinbox.setValue(length)
                    dimensionLayout.addWidget(lengthSpinbox)
                    itemLayout.addWidget(dimensionWidget)

                listWidget.setItemWidget(lastItem, itemWidget)
                lastItem.setSizeHint(itemWidget.sizeHint())
                lastItem.setSelected(True)

    def analyze(self):
        '''
        Plot uC graphs.
        '''
        self.linearPlot.clearPlots()
        self.logPlot.clearPlots()

        groupBoxes = self.scrollArea.findChildren(QGroupBox)

        # dictionary in format of {parentfolder1: {subfolder1: w1, l1}, {subfolder2: w2, l2}, parentfolder2...}
        dimensionDict = {}

        # list where each sublist contains selected subfolders
        allPaths = []

        for groupBox in groupBoxes:  # for loop grabbing info from sidebar
            parentFolderName = groupBox.findChildren(QLabel)[0].text()
            listWidget = groupBox.findChildren(QListWidget)[0]
            selectedItems = listWidget.selectedItems()
            parentFolderEntry = {}
            subfolders = []

            for item in selectedItems:
                itemWidget = listWidget.itemWidget(item)
                subfolderName = itemWidget.findChildren(QLabel)[0].text()
                subfolders.append(os.path.join(parentFolderName, subfolderName))
                dimensionSpinboxes = itemWidget.findChildren(QSpinBox)
                width = dimensionSpinboxes[0].value()
                length = dimensionSpinboxes[1].value()
                parentFolderEntry[subfolderName] = [width, length]

            dimensionDict[parentFolderName] = parentFolderEntry
            allPaths.append(subfolders)

        # plot each parent folder
        for i in range(self.MAX_PLOTS):
            uC_scale(allPaths[i], average_devices=self.averageCheckBox.isChecked(), dimDict=dimensionDict,
                     thickness=100e-9,
                     pg_graphs=[self.linearPlot, self.logPlot], dot_color=pg.intColor(i, alpha=128),
                     text_browser=self.textBrowser)
            if i == len(allPaths) - 1: break

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Analysis complete.")
        msg.setWindowTitle("")
        msg.exec()

    def addPlaceHolder(self, layout, coords=[]):
        '''
        Add a placeholder button.
        layout: layout to which to add button
        coords: [row, column] of layout, if needed
        '''
        placeholderButton = QPushButton()
        sp = placeholderButton.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        placeholderButton.setSizePolicy(sp)
        placeholderButton.setVisible(False)
        if coords:
            layout.addWidget(placeholderButton, coords[0], coords[1])
        else:
            layout.addWidget(placeholderButton)

    def deleteGroupBox(self):
        '''
        Function to handle when delete button is pressed
        '''
        mousePos = self.scrollArea.mapFromGlobal(QtGui.QCursor.pos())
        buttonClicked = self.scrollArea.childAt(mousePos)
        groupBox = buttonClicked.parent()
        self.scrollAreaContents.removeWidget(groupBox)
        groupBox.deleteLater()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
