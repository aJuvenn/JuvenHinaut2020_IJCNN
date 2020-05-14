#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:26:15 2019

@author: ajuven
"""

import numpy as np
from recognized_object import RecognizedObject
from sentence_grounding_network import SentenceGroundingNetwork
import random
import sklearn.linear_model as sklm

import sentence_grounding_test_parameters as param



def random_recognized_object(category_choices = param.CATEGORIES, 
                             position_choices = param.POSITIONS, 
                             color_choices = param.COLORS):
    """
        Returns a random object. One half of the time,
        an empty object is returned.
    """
    
    if np.random.rand() < 0.5:
        random_category = np.random.choice(category_choices)
        random_position = np.random.choice(position_choices)
        random_color = np.random.choice(color_choices)
        return RecognizedObject(random_category, random_position, random_color)
    
    return RecognizedObject(None, None, None)
    


def possible_recognized_object_for_predicate(predicate, fill_unkown_fields = True):
    """
       From a predicate using words from sentence, returns an object from vision module
       corresponding to the situation described by the predicate (in french), using grounded concept
       (in english)
    """
        
    if predicate is None:
        
        if fill_unkown_fields:
            return random_recognized_object()
        
        return RecognizedObject(None, None, None)
    
    
    if fill_unkown_fields:
        default_category = np.random.choice(param.CATEGORIES)
        default_position = np.random.choice(param.POSITIONS)
        default_color = np.random.choice(param.COLORS)
    else:
        default_category = None
        default_position = None
        default_color = None    
        
        
    seen_category = param.OBJ_NAME_TO_CONCEPT.get(predicate.object, default_category)
    seen_position = param.POSITION_NAME_TO_CONCEPT.get(predicate.action, default_position)
    seen_color = param.COLOR_NAME_TO_CONCEPT.get(predicate.color, default_color)
    
    return RecognizedObject(seen_category, seen_position, seen_color)

    

        
        



def recognized_object_to_vect(recognized_object):
    """
        Returns a numpy array of zero and ones describing which features are 
        present in recognized_object
        
        Inputs:
            recognized_object: RecognizedObject instance
    """
    
    output = param.VISION_ENCODER.encode(recognized_object.category)
    output += param.VISION_ENCODER.encode(recognized_object.position)
    output += param.VISION_ENCODER.encode(recognized_object.color)
    
    return output
  

def vision_to_vect(recognized_objects):
    
    vectors = map(recognized_object_to_vect, recognized_objects)
    
    return np.hstack(vectors)
    
    


def is_a_valid_imagined_object(predicate, imagined_object):
    """
        Returns whether the predicate description could apply to the imagine_object.
        
        Inputs:
            predicate: WordPredicate instance
            imagined_object: RecognizedObject instance
        
    """
    
    target = possible_recognized_object_for_predicate(predicate, fill_unkown_fields = False)
    
    for field in ['category', 'position', 'color']: 
    
        wanted_field = getattr(target, field)
        
        if wanted_field is not None and getattr(imagined_object, field) != wanted_field:
            return False
    
    return True
        


def is_a_valid_representation(predicates, imagined_objects):
    """
        Returns whether each predicate  description could apply to its 
        corresponding imagined_object.
        
        
        Inputs:
            predicates: a list of WordPredicate instances
            imagined_objects: a list of RecognizedObject instance
    """
    
    return all(map(is_a_valid_imagined_object, predicates, imagined_objects))
    
    
    
    

def is_an_exact_imagined_object(predicate, imagined_object):
    """
        Returns whether the imagined object matches exactly what the predicate 
        describes.
        
        Inputs:
            predicate: WordPredicate instance
            imagined_object: RecognizedObject instance
        
    """
    target = possible_recognized_object_for_predicate(predicate, fill_unkown_fields = False)
    
    for field in ['category', 'position', 'color']: 

        if getattr(imagined_object, field) != getattr(target, field):
            return False

    return True



def is_an_exact_representation(predicates, imagined_objects):
    """
        Returns whether the imagined object matches exactly what the predicate 
        describes.
        
        Inputs:
            predicates: a list of WordPredicate instances
            imagined_objects: a list of RecognizedObject instance
    """

    return all(map(is_an_exact_imagined_object, predicates, imagined_objects))

        
    



def random_sentence_and_predicates():
    """
        Returns a ranodm sentence and predicates it contains.
        50% of the sentences concern one object only, 50% two.
    """
    
    nb_sentences = len(param.SENTENCE_TO_PREDICATE.items())
    rand_sentence_id_1 = np.random.choice(nb_sentences)
    sentence_1, predicate_1 = param.SENTENCE_TO_PREDICATE.items()[rand_sentence_id_1]
        
    if np.random.rand() < 0.5:
        rand_sentence_id_2 = np.random.choice(nb_sentences)
        sentence_2, predicate_2 = param.SENTENCE_TO_PREDICATE.items()[rand_sentence_id_2]
        sentence = sentence_1 + ' and ' + sentence_2
    else:
        sentence = sentence_1
        predicate_2 = None
       
    predicates = [predicate_1, predicate_2]
    
    return sentence, predicates
    
    

    
    
    
def random_sentences_and_predicates_out_of_set(nb_sentences, sentence_set):
    """
        Selects nb_sentences sentences which are not in sentence_set and
        their corresponding predicates 
    """
    
    output_sentences = []
    output_predicates = []
    
    while len(output_sentences) < nb_sentences:
            
        sentence, predicates = random_sentence_and_predicates()
                
        if sentence in output_sentences or sentence in sentence_set:
            continue
        
        output_sentences.append(sentence)
        output_predicates.append(predicates)
        
    return output_sentences, output_predicates


def random_train_sentences_and_predicates(nb_sentences, already_trained_sentences = []):
    return random_sentences_and_predicates_out_of_set(nb_sentences, already_trained_sentences)  
    

def random_test_sentences(nb_sentences, train_sentences):
    return random_sentences_and_predicates_out_of_set(nb_sentences, train_sentences)
            

    
    
    

def sentence_grounding_network_error(nw, nb_tests, 
                                     evaluation_method = is_a_valid_representation, 
                                     train_sentences = []):
    """
        Computes and returns error percentage made on nb_tests randomly picked sentences
        not apart of the training sentences. Evaluation method, which determines if
        an imagined vision matches or not a predicate list, can be chosen 
        (either is_a_valid_representation or is_an_exact_representation)
    """
    
    test_sentences, test_predicates = random_test_sentences(nb_tests, train_sentences)
       
    nb_bad = 0.
    
    for i, sentence in enumerate(test_sentences):

        predicates = test_predicates[i]
        imagined_vision = [RecognizedObject(*caracterics) for caracterics in nw.ground_sentence(sentence)]
        
        if not evaluation_method(predicates, imagined_vision):
            nb_bad += 1.
            
    return (nb_bad / nb_tests) * 100.
    


    

def train_sentence_grounding_network(nw, nb_trainings, verbose = False, learn_individual_words = False, continuous_learning = False):
    """
        Trains the network nw by randomly picking nb_training sentences
        If learn_individual_words is True, the network learns also the meaning of individual words
        contained in the sentence
    """
    
    if learn_individual_words and continuous_learning:
        print("ERROR: learn_individual_words and continuous_learning can't be booth true")
        return
    
    train_sentences, train_predicates = random_train_sentences_and_predicates(nb_trainings)
    
    for i in range(nb_trainings):
        
        sentence = train_sentences[i]
        predicates = train_predicates[i]
        
        # A vision corresponding to the sentence description is created
        provided_vision = map(possible_recognized_object_for_predicate, predicates)
        caracteristics_list_to_learn = [[x.category, x.position, x.color] for x in provided_vision]
        
        if verbose:
            print sentence
            imagined_caracteristics_list = nw.ground_sentence(sentence)
            imagined_vision = [RecognizedObject(*caracterics) for caracterics in imagined_caracteristics_list]
            print "Imagined:", imagined_vision
            print "Is a valid representation:", is_a_valid_representation(predicates, imagined_vision)
            print "Is an exact representation:", is_an_exact_representation(predicates, imagined_vision)
            print "Provided vision for learning:", provided_vision
            print '---------------------------------'
        
        nw.cross_situational_learning(sentence, caracteristics_list_to_learn, continuous_learning = continuous_learning)
        
        if learn_individual_words:
            # Then every word meaning is also learned separately
            for w in nw.split_sentence(sentence):
                nw.cross_situational_learning(w, caracteristics_list_to_learn)
        

    return train_sentences
    
    

def train_sentence_grounding_network_offline(nw, nb_trainings, offline_ridge, verbose = False, learn_individual_words = False, continuous_learning = False):
    """
        Trains with offline method the network nw by randomly picking nb_training sentences
        If learn_individual_words is True, the network learns also the meaning of individual words
        contained in the sentence
    """
    
    if learn_individual_words:
        print("ERROR: learn_individual_words can't be true")
        return
    
    
    nw.linear_model = sklm.Ridge(offline_ridge)
    
    train_sentences, train_predicates = random_train_sentences_and_predicates(nb_trainings)
    
    for sentence in train_sentences:
        for w in nw.split_sentence(sentence):
            nw.add_word_id_if_unkown(w)
    
    
    input_teachers = []
    output_teachers = []
    
    for i in range(nb_trainings):
        
        sentence = train_sentences[i]
        predicates = train_predicates[i]
        
        # A vision corresponding to the sentence description is created
        provided_vision = map(possible_recognized_object_for_predicate, predicates)
        caracteristics_list_to_learn = [[x.category, x.position, x.color] for x in provided_vision]
        
        if verbose:
            print sentence
            imagined_caracteristics_list = nw.ground_sentence(sentence)
            imagined_vision = [RecognizedObject(*caracterics) for caracterics in imagined_caracteristics_list]
            print "Imagined:", imagined_vision
            print "Is a valid representation:", is_a_valid_representation(predicates, imagined_vision)
            print "Is an exact representation:", is_an_exact_representation(predicates, imagined_vision)
            print "Provided vision for learning:", provided_vision
            print '---------------------------------'
        
        
        input_teachers.append(nw.one_hot_encoding_inputs_for_sentence(sentence))
        output_teachers.append((len(input_teachers[-1])-2) * [np.zeros_like(nw.output_values)]  + 2 * [nw.caracteristics_to_output_teacher(caracteristics_list_to_learn)])
        
    if continuous_learning:
        nb_washing_list = 0
    else:
        nb_washing_list = [len(inpt)-1 for inpt in input_teachers]
        
    nw.learn_series(input_teachers, output_teachers, nb_washing_list, reset_memory = True)
     
    return train_sentences
    

def split_sentence_func_with_dot(sentence):
    """
        Function the network uses to transform a sentence into a list of words.
        In this version, BEGIN and END makers are added to the sentence
    """
    return ['BEGIN'] + [w for w in sentence.split(' ') if w != ''] + ['END']
    
    

def output_activation_plot_func(output):
    """
        Sigmoid function used to make output curve plots more readable
    """
    return 1./ (1. + np.exp(- 5. * (output - 0.5)))
    



if __name__ == '__main__':
    
    NB_TRAININGS = 1000
    NB_TESTS = 1000
    RESERVOIR_SIZE = 1000
    NB_SHOWING = 20
    SHOW_RESERVOIR_PCA = False
    SAVE_FIG = True
    
    CONTINUOUS_LEARNING = False
    INDIVIDUAL_WORDS = False
    TEST_OFFLINE = False
    
    OFFLINE_RIDGE  = 10**(-3.5)
    
    USE_FEEDBACK = False
    FEEDBACK_SCALING = 0.1
    ACTIVATION_FUNC = None
    ACTIVATION_INVERT_FUNC = None
    
    
    
    n = len(param.SENTENCE_TO_PREDICATE.keys())
    print "Number of one object sentences:", n
    print "Total number of two objects sentences:", n * n
    print "Total number of sentences:", n * (n+1)
    
    print "Ojects:", param.OBJECT_NAMES
    print "Positions:", param.POSITION_NAMES
    print "Colors:", param.COLOR_NAMES
    

    if CONTINUOUS_LEARNING:
        network_params = {
                'activation_threshold' : 0.6,
                'softmax_beta' : 2.,
                'softmax_threshold' : 0.25,
                'spectral_radius' : 1.3,
                'leaking_rate' : 0.04,
                'average_nb_connexions' : int(0.81 * RESERVOIR_SIZE), 
                'ridge_coef' : 10.**(-3.7)
        }
    else:
        network_params = {
                'activation_threshold' : 0.6,
                'softmax_beta' : 2.2,
                'softmax_threshold' : 0.1,
                'spectral_radius' : 1.1,
                'leaking_rate' : 0.05,
                'average_nb_connexions' : int(0.85 * RESERVOIR_SIZE), 
                'ridge_coef' : 10.**(-3.5)
        }
    
    nw = SentenceGroundingNetwork(max_nb_seen_objects = 2,
                                  concept_lists = param.CONCEPT_LISTS,
                                  use_softmax = True,
                                  reservoir_size = RESERVOIR_SIZE,
                                  
                                  use_feedback = USE_FEEDBACK,
                                  feedback_scaling = FEEDBACK_SCALING,
                                  
                                  
                                  input_scaling = 1.,
                                  output_activation_function = ACTIVATION_FUNC,
                                  output_activation_inverse_function = ACTIVATION_INVERT_FUNC,
                                  split_sentence_func = split_sentence_func_with_dot,
                        
                                  **network_params
                                 )
        
    print "Training..."
    
    
    if TEST_OFFLINE:
        train_sentences = train_sentence_grounding_network_offline(nw, NB_TRAININGS,
                                                                   offline_ridge = OFFLINE_RIDGE,
                                                                   verbose = False,
                                                                   learn_individual_words = INDIVIDUAL_WORDS,
                                                                   continuous_learning=CONTINUOUS_LEARNING)
    else:
        train_sentences = train_sentence_grounding_network(nw, NB_TRAININGS, verbose = False,
                                                          learn_individual_words = INDIVIDUAL_WORDS,
                                                          continuous_learning=CONTINUOUS_LEARNING)
                                                      

    print "Testing on testing set..."
    
    err_testing_valid = sentence_grounding_network_error(nw, NB_TESTS, 
                                     evaluation_method = is_a_valid_representation,
                                     train_sentences = train_sentences)
    
    err_testing_exact = sentence_grounding_network_error(nw, NB_TESTS, 
                                     evaluation_method = is_an_exact_representation,
                                     train_sentences = train_sentences)  
    print "Testing set:"
    print "\tNot valid : {} %".format(err_testing_valid)
    print "\tNot exact : {} %".format(err_testing_exact)

    print "Showing trained sentences"

    shown_train_sentences = []
    
    for i in range(NB_SHOWING):
        sentence = random.choice(train_sentences)
        shown_train_sentences.append(sentence)
        #nw.plot_concept_activation(sentence, output_function = output_activation_plot_func)
        nw.plot_concept_activation(sentence, savefig=SAVE_FIG, sub_ttl_fig='train_sigm_'+str(i), output_function = output_activation_plot_func)
        #nw.plot_concept_activation(sentence, savefig=True, sub_ttl_fig='train_nosigm_'+str(i))
        
        nw.plot_output_values_matrix(sentence, savefig=SAVE_FIG, sub_ttl_fig='train_'+str(i))
    
    if SHOW_RESERVOIR_PCA:
        nw.plot_state_record_for_sentences(*shown_train_sentences) 
    
    
    print "Showing test sentences"
    
    shown_test_sentences, _ = random_test_sentences(NB_SHOWING, train_sentences)
    
    for i, sentence in enumerate(shown_test_sentences):
        #nw.plot_concept_activation(sentence, output_function = output_activation_plot_func)
        nw.plot_concept_activation(sentence, savefig=SAVE_FIG, sub_ttl_fig='test_sigm_'+str(i) , output_function = output_activation_plot_func)
        #nw.plot_concept_activation(sentence, savefig=True, sub_ttl_fig='test_nosigm_'+str(i))
        nw.plot_output_values_matrix(sentence, savefig=SAVE_FIG, sub_ttl_fig='test_'+str(i))
    
    
    if SHOW_RESERVOIR_PCA:
        nw.plot_state_record_for_sentences(*shown_test_sentences) 
        nw.plot_state_record_for_sentences(*(shown_train_sentences + shown_test_sentences)) 




        
        