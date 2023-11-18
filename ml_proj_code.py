# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:12:59 2023

@author: asmit
"""

from ml_proj_code_cl import classification

import warnings
warnings.filterwarnings("ignore")


clf=classification("C:/Users/asmit/Downloads/",clf_opt='rf',
                        no_of_selected_features=4)

clf.classification()

