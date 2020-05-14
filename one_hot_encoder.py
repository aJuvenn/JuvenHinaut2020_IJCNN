#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:52:27 2019

@author: ajuven
"""


import numpy as np


class OneHotEncoder():
    
    def __init__(self, possible_elements):
        self.encoding_dict = {element : index for index, element in enumerate(possible_elements)}
        self.decoding_dict = dict(enumerate(possible_elements))
        self.encoded_vector_size = len(possible_elements)
        
        
    def encode(self, element):
        
        output = np.zeros(self.encoded_vector_size)
        
        element_index = self.encoding_dict.get(element, None)
        
        if element_index is not None:
            output[element_index] = 0.9999
            
        return output
    
    
    def decode(self, index):
        return self.decoding_dict[index]
    
    
    def print_vect(self, vect, names = None):
        
        j = 0
        
        for i in range(len(vect)):
            
            if i % self.encoded_vector_size == 0:
                print '--------------', (names[j] if names != None and j < len(names) else '')
                j += 1
                 
            print vect[i], self.decoding_dict[i % self.encoded_vector_size]
            
        