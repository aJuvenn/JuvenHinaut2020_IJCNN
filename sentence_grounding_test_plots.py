#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:52:17 2020

@author: ajuvenn
"""



import numpy as np
from sentence_grounding_network import SentenceGroundingNetwork
import tqdm
import matplotlib.pyplot as plt

import sentence_grounding_test_parameters as param
import sentence_grounding_test as sgt



def valid_and_exact_representation_errors_after_training(reservoir_size, nb_trainings, nb_tests, nb_eror_averaging, TEST_OFFLINE = False):
    """
        Trains nb_eror_averaging different networks of reservoir_size size on nb_trainings sentences and evaluates
        two kind of errors (valid representaion or exact representation) by selecting nb_tests sentences from the
        testing set
    """
    
    network_params = {
                'activation_threshold' : 0.6,
                'softmax_beta' : 2.2,
                'softmax_threshold' : 0.1,
                'spectral_radius' : 1.1,
                'leaking_rate' : 0.05,
                'average_nb_connexions' : int(0.85 * reservoir_size), 
                'ridge_coef' : 10.**(-3.5)
    }
    
    
    errors_valid_repr = np.empty(nb_eror_averaging)
    errors_exact_repr = np.empty(nb_eror_averaging)
   
    for i in range(nb_eror_averaging):
            
        
        nw = SentenceGroundingNetwork(max_nb_seen_objects = 2,
                                  concept_lists = param.CONCEPT_LISTS,
                                  use_softmax = True,
                                  reservoir_size = reservoir_size,
                                  
                                  use_feedback = False,
                                  feedback_scaling = 1.,
                                  
                                  
                                  input_scaling = 1.,
                                  output_activation_function = None,
                                  output_activation_inverse_function = None,
                                  split_sentence_func = sgt.split_sentence_func_with_dot,
                        
                                  **network_params
                                 )
        
    
        if TEST_OFFLINE:
            train_sentences = sgt.train_sentence_grounding_network_offline(nw, nb_trainings, network_params['ridge_coef'], verbose = False, learn_individual_words = False)
        else:
            train_sentences = sgt.train_sentence_grounding_network(nw, nb_trainings, verbose = False, learn_individual_words = False)
                                                          
        err_valid = sgt.sentence_grounding_network_error(nw, nb_tests, sgt.is_a_valid_representation, train_sentences)
        err_exact = sgt.sentence_grounding_network_error(nw, nb_tests, sgt.is_an_exact_representation, train_sentences)
        
        errors_valid_repr[i] = err_valid
        errors_exact_repr[i] = err_exact
        
    return np.mean(errors_valid_repr), np.mean(errors_exact_repr)



def errors_in_function_of_nb_trainings(reservoir_size, nb_tests, nb_error_averaging, start_nb_training, stop_nb_training, nb_training_step, TEST_OFFLINE = False):
    
    tested_nb_trainings = np.arange(start_nb_training, stop_nb_training+1, nb_training_step, dtype = int)
    errors_valid = np.empty_like(tested_nb_trainings, dtype = float)
    errors_exact = np.empty_like(tested_nb_trainings, dtype = float)
    length = len(tested_nb_trainings)
    
    for i in tqdm.trange(length):
        nb_trainings = tested_nb_trainings[i]
        err_valid, err_exact = valid_and_exact_representation_errors_after_training(reservoir_size, nb_trainings, nb_tests, nb_error_averaging, TEST_OFFLINE = TEST_OFFLINE)
        print nb_trainings, err_valid, err_exact
        errors_valid[i] = err_valid
        errors_exact[i] = err_exact
        
    return tested_nb_trainings, errors_valid, errors_exact
    

def errors_in_function_of_reservoir_size(nb_trainings, nb_tests, nb_error_averaging, start_reservoir_size, stop_reservoir_size, step_reservoir_size, TEST_OFFLINE = False):
    
    tested_reservoir_size = np.arange(start_reservoir_size, stop_reservoir_size+1, step_reservoir_size, dtype = int)
    errors_valid = np.empty_like(tested_reservoir_size, dtype = float)
    errors_exact = np.empty_like(tested_reservoir_size, dtype = float)
    length = len(tested_reservoir_size)
    
    for i in tqdm.trange(length):
        reservoir_size = tested_reservoir_size[i]
        err_valid, err_exact = valid_and_exact_representation_errors_after_training(reservoir_size, nb_trainings, nb_tests, nb_error_averaging, TEST_OFFLINE = TEST_OFFLINE)
        print reservoir_size, err_valid, err_exact
        errors_valid[i] = err_valid
        errors_exact[i] = err_exact
        
    return tested_reservoir_size, errors_valid, errors_exact
    


def errors_in_function_of_nb_object(nb_trainings, nb_tests, nb_error_averaging, reservoir_size, stop_nb_object, TEST_OFFLINE = False):
    
    param.create_dataset()
    default_nb_objects = len(param.OBJECT_NAMES)
    
    tested_nb_objects = np.array(range(default_nb_objects, stop_nb_object+1), dtype = int)
    errors_valid = np.empty_like(tested_nb_objects, dtype = float)
    errors_exact = np.empty_like(tested_nb_objects, dtype = float)
    
    for nb_objects in tqdm.trange(default_nb_objects, stop_nb_object+1):
        
        err_valid, err_exact = valid_and_exact_representation_errors_after_training(reservoir_size, nb_trainings, nb_tests, nb_error_averaging, TEST_OFFLINE = TEST_OFFLINE)
        print nb_objects, err_valid, err_exact
        errors_valid[nb_objects - default_nb_objects] = err_valid
        errors_exact[nb_objects - default_nb_objects] = err_exact
        
        if nb_objects != stop_nb_object:
            param.add_object('obj_' + str(nb_objects))
            
    return tested_nb_objects, errors_valid, errors_exact
        
        



def err(nb_object, nb_trainings):
    return 100. * np.power((1. - 1./(85. * nb_object)), 2. * nb_trainings)

    

        
def plot_errors_in_function_of_reservoir_size(tested_reservoir_size, errors_valid, errors_exact, 
                                              offline_tested_reservoir_size, offline_errors_valid, offline_errors_exact, 

                                              x_plot_ticks=50., y_plot_ticks = 5., log_display = False):

    if log_display:
        plt.plot(np.log(tested_reservoir_size), np.log(errors_valid), color = 'orange')
        plt.plot(np.log(tested_reservoir_size), np.log(errors_exact), color = 'green')
    else:
        
        plt.plot(tested_reservoir_size, errors_valid, color = 'orange', linewidth = 3)
        plt.plot(offline_tested_reservoir_size, offline_errors_valid, color = 'brown', linestyle = '--', linewidth = 1.5)
       
        plt.plot(tested_reservoir_size, errors_exact, color = 'yellowgreen', linewidth = 3)
        plt.plot(offline_tested_reservoir_size, offline_errors_exact, color = 'darkolivegreen', linestyle = '--', linewidth = 1.5)
        
        plt.xlim(0., max(tested_reservoir_size))
        plt.ylim(0., 100.)
        plt.xticks(np.arange(0., max(tested_reservoir_size)+0.1, x_plot_ticks))
        plt.yticks(np.arange(0., 100.1, y_plot_ticks))

        
    plt.grid(linestyle='--')
    plt.legend(['online learning, valid metric',
                'offline learning, valid metric', 
                'online learning, exact metric',
                'offline learning, exact metric'])
    plt.xlabel('Reservoir size')
    plt.ylabel('Error percentage')
    plt.savefig('error_curves_ressize.pdf')
    plt.show()



def plot_errors_in_function_of_nb_trainings(tested_nb_trainings, errors_valid, errors_exact, 
                                            offline_tested_nb_trainings, offline_errors_valid, offline_errors_exact,
                                            x_plot_ticks=50., y_plot_ticks = 5., log_display = False):
        
    th_loose_func = lambda n : err(4, n)
    
    if log_display:
        plt.plot(np.log(tested_nb_trainings), np.log(errors_valid), color = 'orange')
        plt.plot(np.log(tested_nb_trainings), np.log(errors_exact), color = 'green')
    else:
        
        plt.plot(tested_nb_trainings, errors_valid, color = 'orange', linewidth = 3)
        plt.plot(offline_tested_nb_trainings, offline_errors_valid, color = 'brown', linestyle = '--', linewidth = 1.5)
       
        plt.plot(tested_nb_trainings, errors_exact, color = 'yellowgreen', linewidth = 3)
        plt.plot(offline_tested_nb_trainings, offline_errors_exact, color = 'darkolivegreen', linestyle = '--', linewidth = 1.5)

        
        plt.plot(tested_nb_trainings, th_loose_func(tested_nb_trainings), color = 'black', linestyle = (0, (1, 3)), linewidth = 4)
        
        
        
        plt.xlim(0., max(tested_nb_trainings))
        plt.ylim(0., 100.)
        plt.xticks(np.arange(0., max(tested_nb_trainings)+0.1, x_plot_ticks))
        plt.yticks(np.arange(0., 100.1, y_plot_ticks))
        
    plt.grid(linestyle='--')
    plt.legend(['online learning, valid metric',
                'offline learning, valid metric', 
                'online learning, exact metric',
                'offline learning, exact metric',
                'abstract model error'])
    
    plt.xlabel('Number of trainings')
    plt.ylabel('Error percentage')
    plt.savefig('error_curves_nb_trainings.pdf')
    plt.show()
    
    
def plot_errors_in_function_of_nb_objects(tested_nb_objects, errors_valid, errors_exact, 
                                          offline_tested_nb_objects, offline_errors_valid, offline_errors_exact, 
                                          x_plot_ticks = 5., y_plot_ticks = 5., log_display = False):
     
    tested_nb_objects = np.asarray(tested_nb_objects)
    
    if log_display:
        plt.plot(np.log(tested_nb_objects), np.log(errors_valid), color = 'orange')
        plt.plot(np.log(tested_nb_objects), np.log(errors_exact), color = 'green')
    else:
        
        
        plt.plot(tested_nb_objects, errors_valid, color = 'orange', linewidth = 3)
        plt.plot(offline_tested_nb_objects, offline_errors_valid, color = 'brown', linestyle = '--', linewidth = 1.5)
       
        
        plt.plot(tested_nb_objects, errors_exact, color = 'yellowgreen', linewidth = 3)
        plt.plot(offline_tested_nb_objects, offline_errors_exact, color = 'darkolivegreen', linestyle = '--', linewidth = 1.5)

        
        plt.plot(tested_nb_objects, err(tested_nb_objects, 1000), color = 'black', linestyle = (0, (1, 3)), linewidth = 4)
        
        plt.xlim(min(tested_nb_objects), max(tested_nb_objects))
        plt.ylim(0., 100.)
        plt.xticks(np.arange(0., max(tested_nb_objects)+0.1, x_plot_ticks))
        plt.yticks(np.arange(0., 100.1, y_plot_ticks))
        
    
    plt.grid(linestyle='--')
    plt.legend(['online learning, valid metric',
                'offline learning, valid metric', 
                'online learning, exact metric',
                'offline learning, exact metric',
                'abstract model error'])
    
    plt.xlabel('Number of objects')
    plt.ylabel('Error percentage')
    plt.savefig('error_curves_nb_obj.pdf')
    plt.show()




def plot_saved_curve(tested_values_path, error_valid_path, error_exact_path, 
                     offline_values_path, offline_error_valid_path, offline_error_exact_path, 
                     in_function_of, x_plot_ticks = 100.,  y_plot_ticks = 5.):
    
    tested_values = np.loadtxt(tested_values_path)
    error_valid = np.loadtxt(error_valid_path)
    error_exact = np.loadtxt(error_exact_path)
      
    offline_tested_values = np.loadtxt(offline_values_path)
    offline_error_valid = np.loadtxt(offline_error_valid_path)
    offline_error_exact = np.loadtxt(offline_error_exact_path)
    
    if in_function_of == 'nb_training':
         plot_errors_in_function_of_nb_trainings(tested_values, error_valid, error_exact, 
                                                 offline_tested_values, offline_error_valid, offline_error_exact, 
                                                 x_plot_ticks = x_plot_ticks, y_plot_ticks = y_plot_ticks)
         
    elif in_function_of == 'reservoir_size':
        plot_errors_in_function_of_reservoir_size(tested_values, error_valid, error_exact,
                                                  offline_tested_values, offline_error_valid, offline_error_exact, 
                                                  x_plot_ticks = x_plot_ticks, y_plot_ticks = y_plot_ticks)
        
    elif in_function_of == 'nb_obj':
        plot_errors_in_function_of_nb_objects(tested_values, error_valid, error_exact, 
                                              offline_tested_values, offline_error_valid, offline_error_exact, 
                                              x_plot_ticks = x_plot_ticks, y_plot_ticks = y_plot_ticks)
        
    else:
        print "Invalid IN_FUNCTION_OF"    




def train_and_plot(in_function_of, x_plot_ticks = 100.,  y_plot_ticks = 5., log_display = False, TEST_OFFLINE = False):

    NB_TESTS = 1000
    NB_ERROR_AVERAGING = 5
            
    if in_function_of == 'nb_training':
            
        RESERVOIR_SIZE = 1000
        START_NB_TRAININGS = 1
        STOP_NB_TRAININGS = 1001
        NB_TRAININGS_STEP = 100
            
        plot_errors_in_function_of_nb_trainings(*errors_in_function_of_nb_trainings(RESERVOIR_SIZE, NB_TESTS, NB_ERROR_AVERAGING, 
                                                                                    START_NB_TRAININGS, STOP_NB_TRAININGS, NB_TRAININGS_STEP,
                                                                                    TEST_OFFLINE = TEST_OFFLINE),
                                                x_plot_ticks = x_plot_ticks, y_plot_ticks = y_plot_ticks, log_display = log_display)
            
    elif in_function_of == 'reservoir_size':
            
        NB_TRAININGS = 1000
        START_RESERVOIR_SIZE = 10
        STOP_RESERVOIR_SIZE = 1000
        STEP_RESERVOIR_SIZE = 100       
        plot_errors_in_function_of_reservoir_size(*errors_in_function_of_reservoir_size(NB_TRAININGS, NB_TESTS, NB_ERROR_AVERAGING, 
                                                                                        START_RESERVOIR_SIZE, STOP_RESERVOIR_SIZE, STEP_RESERVOIR_SIZE,
                                                                                        TEST_OFFLINE = TEST_OFFLINE),
                                                  x_plot_ticks = x_plot_ticks, y_plot_ticks = y_plot_ticks, log_display = log_display)
            
            
    elif in_function_of == 'nb_obj':
        
        NB_TRAININGS = 1000
        RESERVOIR_SIZE = 1000
        STOP_NB_OBJ = 50
        plot_errors_in_function_of_nb_objects(*errors_in_function_of_nb_object(NB_TRAININGS, NB_TESTS, NB_ERROR_AVERAGING, RESERVOIR_SIZE, STOP_NB_OBJ,
                                                                               TEST_OFFLINE = TEST_OFFLINE),
                                              x_plot_ticks = x_plot_ticks, y_plot_ticks = y_plot_ticks, log_display = log_display)
        
    else:
        print "Invalid IN_FUNCTION_OF"    

  
    

if __name__ == '__main__':
    
    TRAIN_AND_PLOT = False
    PLOT_SAVED_CURVES = True 
    TEST_OFFLINE = False
    
    
    if TRAIN_AND_PLOT:
        train_and_plot('nb_training', x_plot_ticks = 100.,  y_plot_ticks = 5., log_display = False, TEST_OFFLINE = TEST_OFFLINE)
        train_and_plot('reservoir_size', x_plot_ticks = 100.,  y_plot_ticks = 5., log_display = False, TEST_OFFLINE = TEST_OFFLINE)
        train_and_plot('nb_obj', x_plot_ticks = 100.,  y_plot_ticks = 5., log_display = False, TEST_OFFLINE = TEST_OFFLINE)
        
    
    if PLOT_SAVED_CURVES:
        plot_saved_curve('./Results/Nb_trainings/nb_trainings.txt', 
                         './Results/Nb_trainings/err_valid.txt', 
                         './Results/Nb_trainings/err_exact.txt', 
                         './Results/Nb_trainings/nb_trainings_offline.txt', 
                         './Results/Nb_trainings/err_valid_offline.txt', 
                         './Results/Nb_trainings/err_exact_offline.txt', 
                         'nb_training')
        
        plot_saved_curve('./Results/Nb_objects/tested_nb_objects.txt', 
                         './Results/Nb_objects/err_valid.txt', 
                         './Results/Nb_objects/err_exact.txt', 
                         './Results/Nb_objects/tested_nb_objects_offline.txt', 
                         './Results/Nb_objects/err_valid_offline.txt', 
                         './Results/Nb_objects/err_exact_offline.txt',
                         'nb_obj', x_plot_ticks=5., y_plot_ticks = 5.)
  
    
        plot_saved_curve('./Results/Reservoir_size/tested_reservoir_size.txt', 
                         './Results/Reservoir_size/err_valid.txt', 
                         './Results/Reservoir_size/err_exact.txt', 
                         './Results/Reservoir_size/tested_reservoir_size_offline.txt', 
                         './Results/Reservoir_size/err_valid_offline.txt', 
                         './Results/Reservoir_size/err_exact_offline.txt', 
                         'reservoir_size')
        
   
            
    
    
    
    