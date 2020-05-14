#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import hyperopt as hyp
import sentence_grounding_test as sgt
import tqdm



NB_EVAL = 200
NB_TRAININGS = 500
NB_TESTS = 1000
NB_ERROR_AVERAGING = 1

RESERVOIR_SIZE = 200
OUTPUT_ACTIVATION_FUNCTION = None
OUTPUT_ACTIVATION_INVERSE_FUNCTION = None

EVALUATION_METHOD = sgt.is_an_exact_representation

CONTINUOUS_LEARNING = False
INDIVIDUAL_WORDS = False

TEST_OFFLINE = False
USE_FEEDBACK = False




def cost_function(args):
    
    (
     leaking_rate, 
     spectral_radius, 
     density, 
     ridge_coef_log, 
     #input_scaling, 
     feedback_scaling, 
     activation_threshold,
     softmax_beta,
     softmax_threshold
     ) = args
    
    
    errors = []
    
    for i in tqdm.trange(NB_ERROR_AVERAGING):
        
        nw = sgt.SentenceGroundingNetwork(max_nb_seen_objects = 2,
                                  concept_lists = sgt.param.CONCEPT_LISTS,
                                  activation_threshold = activation_threshold,
                                  use_softmax = True,
                                  softmax_beta = softmax_beta,
                                  softmax_threshold = softmax_threshold,
                                  reservoir_size = RESERVOIR_SIZE,
                                  spectral_radius = spectral_radius,
                                  leaking_rate = leaking_rate,
                                  average_nb_connexions = int(density * RESERVOIR_SIZE), 
                                  use_feedback = USE_FEEDBACK,
                                  feedback_scaling = feedback_scaling,
                                  input_scaling = 1.,
                                  
                                  output_activation_function = OUTPUT_ACTIVATION_FUNCTION,
                                  output_activation_inverse_function = OUTPUT_ACTIVATION_INVERSE_FUNCTION,
                                  
                                  ridge_coef = 10.**(ridge_coef_log),
                                      
                                  split_sentence_func = sgt.split_sentence_func_with_dot
                                 )
    
    
        if TEST_OFFLINE:
            train_sentences = sgt.train_sentence_grounding_network_offline(nw, NB_TRAININGS,
                                                                   offline_ridge = 10 ** (ridge_coef_log),
                                                                   verbose = False,
                                                                   learn_individual_words = INDIVIDUAL_WORDS,
                                                                   continuous_learning=CONTINUOUS_LEARNING)
        else:
            train_sentences = sgt.train_sentence_grounding_network(nw, NB_TRAININGS, verbose = False,
                                                          learn_individual_words = INDIVIDUAL_WORDS,
                                                          continuous_learning=CONTINUOUS_LEARNING)
    
        errors.append(sgt.sentence_grounding_network_error(nw, NB_TESTS, evaluation_method = EVALUATION_METHOD, train_sentences = train_sentences))
    
    return np.mean(errors)
    
    
    
parameters = (
    'leaking_rate',
    'spectral_radius',
    'density',
    'ridge_coef_log',
   # 'input_scaling',
    'feedback_scaling',
    
    'activation_threshold',
   'softmax_beta',
   'softmax_threshold'
)



# Create the domain space
space = [hyp.hp.uniform('leaking_rate', 0., 1.),
         hyp.hp.uniform('spectral_radius', 0.3, 3.),
         hyp.hp.uniform('density', 0.1, 0.9),
         hyp.hp.uniform('ridge_coef_log', -4., -0.1),
         
        # hyp.hp.uniform('input_scaling', 0.1, 4.),
        hyp.hp.uniform('feedback_scaling', 0., 2.),
         
         hyp.hp.uniform('activation_threshold', 0., 1.),
         hyp.hp.uniform('softmax_beta', 0.5, 5.),
        hyp.hp.uniform('softmax_threshold', 0., 1.)
        ]



tpe_algo = hyp.tpe.suggest

tpe_trials = hyp.Trials()


tpe_best = hyp.fmin(fn = cost_function, space = space, 
                    algo = tpe_algo, trials = tpe_trials, 
                    max_evals = NB_EVAL)

print(tpe_best)



def show_influence_on_objective(parameter_label, trials):
    
    fig, ax = plt.subplots(1)
    
    xs = [t['misc']['vals'][parameter_label] for t in trials.trials]
    ys = [t['result']['loss'] for t in trials.trials]
    
    ax.scatter(xs, ys, s = 20, linewidth = 0.01, alpha = 0.75, color = (0.8,0.,0.))
    ax.set_title('error vs ' + parameter_label, fontsize = 18)
    ax.set_xlabel(parameter_label, fontsize = 16)
    ax.set_ylabel('error', fontsize = 16)

    plt.savefig('influence of ' + parameter_label)
    #plt.show()


for p in parameters:
    show_influence_on_objective(p, tpe_trials)
