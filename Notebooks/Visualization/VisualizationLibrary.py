"""The main class that contains all the tabs in the visualization"""
import ipywidgets as widgets
from IPython.display import display

import Tabs


class Visualization:

    def __init__(self):
        self.tab = widgets.Tab()
        self.tab.children = [
            Tabs.PolicyComparisonTab(),
            Tabs.LearningCurvesTab(),
            Tabs.EpisodeReviewTab(),
            Tabs.PiePlotTab(),
            Tabs.HeatMapTab(),
            Tabs.HeatMapComparisonTab(),
            Tabs.DimensionalityReductionPlotTab(),
            Tabs.ExperimentsDirectoryTab()
        ]
        titles = ['Policy performance',
                  'Learing curves',
                  'Episode review',
                  'Success Pie',
                  'Success Heatmap',
                  'Heatmap Compare',
                  'Failures scatter plot',
                  'Edit',
                  ]
        # self.tab.children = [Tabs.HeatMapComparisonTab()]
        # titles = ['test']
        for i in range(len(self.tab.children)):
            self.tab.set_title(i, titles[i])

    def display(self):
        display(self.tab)
