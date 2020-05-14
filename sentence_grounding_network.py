#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:35:27 2019

@author: ajuven
"""

from sentence_processing_network import SentenceProcessingNetwork, split_sentence
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np

    

def softmax(x, beta = 1.):
    y = np.exp(beta * (x - np.max(x)))
    return y / np.sum(y)    
    



    
class SentenceGroundingNetwork(SentenceProcessingNetwork):
    
    
    def __init__(self,
                 max_nb_seen_objects,
                 concept_lists,
                 
                 activation_threshold,
                 
                 use_softmax,
                 softmax_beta,
                 softmax_threshold,
                 
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
            
            max_nb_seen_objects: Number of seen objects 
            concept_lists : list of list of concepts. Each list of concept corresponds to a caracterictic
                            (ex : color, category, ...)
                 
            activation_threshold : minimum output activation to consider a concept as activated
                 
            use_softmax: whether to use or not a second verification step with softmax
            softmax_beta: softmax inverse temperature
            softmax_threshold: minimum softmax(output activation) to consider a concept as activated
            
            split_sentence_func: function used to transform a sentence into a list of words
        """
        
        
        self.max_nb_seen_objects = max_nb_seen_objects
        self.concept_lists = concept_lists
        self.concept_delimitations = [0]
        self.concepts = []    
        
        for l in concept_lists:
            self.concept_delimitations.append(self.concept_delimitations[-1] + len(l))
            self.concepts.extend(l)
        
        self.nb_concepts = len(self.concepts)
        self.nb_object_properties = len(concept_lists)
    
        self.activation_threshold = activation_threshold
        self.use_softmax = use_softmax
        self.softmax_beta = softmax_beta
        self.softmax_threshold = softmax_threshold
        
        self.concept_to_output_id_dict = dict()
        
        for i, c in enumerate(self.concepts):
            self.concept_to_output_id_dict[c] = i
            
        
        
        SentenceProcessingNetwork.__init__(self, 
                                           output_size = max_nb_seen_objects * self.nb_concepts, 
                                           reservoir_size = reservoir_size, 
                                           spectral_radius = spectral_radius,
                                           leaking_rate = leaking_rate,
                                           average_nb_connexions = average_nb_connexions, 
                                           use_feedback = use_feedback,
                                           feedback_scaling = feedback_scaling,
                                           input_scaling = input_scaling,
                                           output_activation_function = output_activation_function,
                                           output_activation_inverse_function = output_activation_inverse_function,
                                           reservoir_noise_scaling  = reservoir_noise_scaling,
                                           input_sparsity = input_sparsity,
                                           ridge_coef = ridge_coef,
                                           split_sentence_func = split_sentence_func)     
        
        
        
    
    def maximum_output_concept_id(self, object_id, caracteristic_id):
        """
            Returns the index of the maxmimum activation for an object and caracteristic (ex : color of the first object)
            If the maximum activation is not beyond the threshold, or too much concepts have a hight activation,
            None is returned.
        """
        
        delimitations = [object_id * self.nb_concepts + x for x in self.concept_delimitations]
        caracteristic_outputs = self.output_values[delimitations[caracteristic_id] : delimitations[caracteristic_id+1]]
           
        max_caracteristic_index = np.argmax(caracteristic_outputs)
        caracteristic_max_output = caracteristic_outputs[max_caracteristic_index]
                
        if caracteristic_max_output < self.activation_threshold:
            # Then maximum activation is not enough
            return None
                    
        if self.use_softmax and softmax(caracteristic_outputs, self.softmax_beta)[max_caracteristic_index] < self.softmax_threshold:
            # Then too many concepts are activated at the same time
            return None
                    
        return max_caracteristic_index
        
    
    
    def ground_sentence(self, sentence):
        """
            Reads the sentence and returns the concepts contained in the sentence
            outupt : a list of list of concepts. (every concepts for each objects)
        """
    
        self.run_on_sentence(sentence)
        self.reset_memory()
        
        output = []
        
        for object_id in range(self.max_nb_seen_objects):
    
             
            recognized_object_properties = []
            
            for caracteristic_id, concepts  in enumerate(self.concept_lists):
                
                max_caracteristic_id = self.maximum_output_concept_id(object_id, caracteristic_id)
                
                if max_caracteristic_id is None:
                    chosen_concept = None
                else:
                    chosen_concept = concepts[max_caracteristic_id]
                
                recognized_object_properties.append(chosen_concept)
                
            output.append(recognized_object_properties)
            
        return output
    
    
    
    def caracteristics_to_output_teacher(self, object_lists):
        """
            Transforms a list of caracteristics into a teacher numpy vector
        """
        
        targeted_output = np.zeros(self.output_size)
        
        for i, obj in enumerate(object_lists):
            
            offset = i * self.nb_concepts
            
            for j, concept in enumerate(obj):
                
                if concept is None:
                    continue
                
                concept_id = offset + self.concept_to_output_id_dict[concept]
                targeted_output[concept_id] = 1.
                
        return targeted_output
                
                
    def cross_situational_learning(self, sentence, object_lists, continuous_learning = False):
        """
            Learns by cross situational learning the meaning of the sentence pronounced
            while seing the object list.
        """
        
        targeted_output = self.caracteristics_to_output_teacher(object_lists)
                
        if not continuous_learning:
            self.run_on_sentence(sentence)
            self.learn_from_current_state_RLS(targeted_output)
        else:
            words = self.split_sentence(sentence)
            for w in words:
                self.next_output_for_word(w)    
                self.learn_from_current_state_RLS(targeted_output)
                
        self.reset_memory()

    
    
    def plot_concept_activation(self, sentence, output_function = None, savefig=False, sub_ttl_fig=''):
        """
            Plots activation through time of the different concepts while hearing
            the sentence. If output_function is not None, it is applied to the reservoir
            output vector before plotting.
        """
        
        outputs = self.run_on_sentence(sentence)
        activation_threshold = self.activation_threshold
        
        if output_function is not None:
            
            activation_threshold = output_function(activation_threshold)
            
            for i in range(outputs.shape[0]):
                outputs[i, :] = output_function(outputs[i, :])
        
        self.reset_memory()
        
        words = self.split_sentence(sentence)
        
        fig, axes = plt.subplots(self.nb_object_properties, self.max_nb_seen_objects, figsize=(25,20))
            
        for i in range(self.max_nb_seen_objects):
        
            offset = i * self.nb_concepts
            
            axes[0, i].set_title("Object " + str(i+1), fontsize = 22)
            
            for j in range(self.nb_object_properties):
                
                ax = axes[j, i]
                ax.plot(outputs[:, offset + self.concept_delimitations[j] : offset + self.concept_delimitations[j+1], 0], linewidth = 4)
                ax.legend(self.concept_lists[j], loc = 2, fontsize = 22)
                
                ax.set_yticks([0., 0.5, 1.])
                ax.set_yticklabels([0., 0.5, 1.], fontsize = 20)
                
                ax.set_ylim([-0.2, 1.2])
                
                ax.set_xticks(np.arange(len(words)))
                ax.set_xticklabels(words, fontsize = 24)
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize = 22)
                
                ax.plot(len(words) * [activation_threshold], '--', color = 'grey', linewidth = 3)
    
        fig.suptitle(sentence, fontsize = 26)
        plt.subplots_adjust(hspace = 0.3)
        
        if savefig:
            plt.savefig('sentence_'+sub_ttl_fig+".pdf", bbox_inches='tight')
            plt.close()
        else:
            fig.show()
            #plt.show()
        
        
        
    def plot_output_values_matrix(self, sentence, savefig=False, sub_ttl_fig=''):
        """
            Plots a matrix showing which concepts are activated after hearing the sentence 
        """
        
        self.run_on_sentence(sentence)
        self.reset_memory()
            
        values_matrix = self.output_values.T.reshape(self.max_nb_seen_objects, -1)
        cropped_values_matrix = np.clip(values_matrix, 0., 1.)
            
        fig, ax = plt.subplots(figsize=(15,10))
        ax.imshow(cropped_values_matrix, cmap = plt.get_cmap('Greys'), vmin=0., vmax=1.)
        
        ax.set_xticks(np.arange(self.nb_concepts))   
        ax.set_xticklabels(self.concepts)
        
        ax.set_yticks(range(self.max_nb_seen_objects))
        ax.set_yticklabels(['object ' + str(i+1) for i in range(self.max_nb_seen_objects)], fontsize = 16)
            
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize = 16)
            
        # Loop over data dimensions and create text annotations.
        for i in range(self.max_nb_seen_objects):
            
            for j in range(self.nb_object_properties):
                
                id_to_plot_in_red = self.maximum_output_concept_id(i, j)
                
                for k in range(len(self.concept_lists[j])):
                    
                    matrix_column_id = self.concept_delimitations[j] + k

                    value = values_matrix[i, matrix_column_id]
                    value_to_plot = int(100. * value)/100.
                    font = FontProperties()
                    
                    if k == id_to_plot_in_red:
                        color = 'r'    
                        font.set_weight('bold')
                    else:
                        color = 'black'
                    
                    ax.text(matrix_column_id, i, value_to_plot, ha="center", va="center", 
                            color= color, fontsize=12, fontproperties = font)
            
        ax.set_title(sentence, fontsize = 22)
        fig.tight_layout()
        if savefig:
            plt.savefig('matrix_'+sub_ttl_fig+".pdf", bbox_inches='tight')
            plt.close()
        else:
            fig.show()
        
        
        
        
        
        
        
    



