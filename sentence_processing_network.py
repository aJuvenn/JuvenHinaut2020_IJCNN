#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:48:54 2019

@author: ajuven
"""


import numpy as np
from FORCE_learning import EchoStateNetworkRLS


def split_sentence(sentence):
    """
        Transforms a sentence into a list of words
    """
    return [x for x in sentence.split(' ') if x != '']



class SentenceProcessingNetwork(EchoStateNetworkRLS):
    """
        Online reservoir that takes sentences as input
    """
    
    def __init__(self, output_size, 
                 reservoir_size = 300, 
                 spectral_radius = 1.25,
                 leaking_rate = 0.3,
                 average_nb_connexions = None, 
                 use_feedback = False,
                 feedback_scaling = 1.,
                 input_scaling = 1.,
                 output_activation_function = None,
                 output_activation_inverse_function = None,
                 reservoir_noise_scaling  = 0.,
                 input_sparsity = 1.,
                 
                 ridge_coef = 1e-6,
                 
                 
                 split_sentence_func = split_sentence
                ):
        """
        Inputs (see mother class for reservoir parameters)
            split_sentence_func : function used to a sentence into a list of words  
        """
                 
                
        EchoStateNetworkRLS.__init__(self,
                                     input_size = 0,
                                     output_size = output_size,
                                     reservoir_size = reservoir_size, 
                                     spectral_radius = spectral_radius,
                                     leaking_rate = leaking_rate,
                                     average_nb_connexions = average_nb_connexions, 
                                     use_feedback = use_feedback,
                                     use_raw_input = False,
                                     feedback_scaling = feedback_scaling,
                                     input_scaling = input_scaling,
                                     output_activation_function = output_activation_function,
                                     output_activation_inverse_function = output_activation_inverse_function,
                                     reservoir_noise_scaling  = reservoir_noise_scaling,
                                     input_sparsity = input_sparsity,
                 
                                     ridge_coef = ridge_coef
                                     )
        
        self.word_to_id_dict = {}
        self.split_sentence = split_sentence_func
        
        
    def add_word_id_if_unkown(self, word):
        """
            If a words was never seen, it is added to known words. Reservoir input
            size is increased and an input id is given to the word.
        """
        
        keys = self.word_to_id_dict.keys()
        
        if word in keys:
            return  self.word_to_id_dict[word]
        
        new_id = self.input_size
        self.word_to_id_dict[word] = new_id
        self.increase_input_size()
        
        return new_id


    def one_hot_encoding_input_for_word(self, word):
        """
            Transforms a word into a vector of 0s with only one 1, corresponding
            to what the network receives as input.
        """
        
        word_id = self.add_word_id_if_unkown(word)
        network_input = np.zeros(self.input_size)
        network_input[word_id] = 1.
        
        return network_input


    def one_hot_encoding_inputs_for_sentence(self, sentence):
        """
            Transforms a sentence into a list of encoded vectors
        """
        
        words = self.split_sentence(sentence)
        
        for w in words:
            self.add_word_id_if_unkown(w)
        
        return [self.one_hot_encoding_input_for_word(w) for w in words]
        

    def next_output_for_word(self, word):
        """
            Provides the word to the network and returns its output
        """
        
        network_input = self.one_hot_encoding_input_for_word(word)
        return self.next_output(network_input)
        
    
    def run_on_sentence(self, sentence):
        """
            Feeds the network with each words of the sentence and returns 
            its activation output through time
        """
        
        outputs = []
        words = self.split_sentence(sentence)
        
        for w in words:
            outputs.append(self.next_output_for_word(w))

        return np.asarray(outputs)

        
    def plot_state_record_for_sentences(self, *sentences):
        """
            Plots a 3D graph of the state activation through time while reading the sentences
        """
        
        for sentence in sentences:
            self.run_on_sentence(sentence)
            self.reset_memory()
            
        input_lists = [self.one_hot_encoding_inputs_for_sentence(s) for s in sentences]
        all_words = [self.split_sentence(s) for s in sentences]
        
        self.plot_state_record(input_lists, sentences, all_words)
        
    
    
    