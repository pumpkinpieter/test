#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:26:01 2020

@author: pv
"""

import numpy as np


def goodfunc(x):
    return x

class Happy:
    
    mood = "Happy"
    
    def __init__(self, look):
        self.look = look

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

class Square(Rectangle):

    def __init__(self, length):
        super().__init__(length, length)
