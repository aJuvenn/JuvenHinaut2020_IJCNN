#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import one_hot_encoder
from grammar_manipulation import role_for_words, cat, union, maybe, sentence_to_roles
from sentence_to_predicate import WordPredicate, NO_ROLE, ACTION, OBJECT, COLOR
import string



def create_dataset(high_difficulty = True):
    
    object_names = ['glass', 'orange', 'cup', 'bowl']
    color_names = ['blue', 'orange', 'green', 'red']
    position_names = ['left', 'right', 'middle']
     
    clean_dataset()
    
    for obj in object_names:
        add_object(obj, build_after = False)
        
    for col in color_names:
        add_color(col, build_after = False)
        
    for pos in position_names:
        add_position(pos, build_after = False)
    
    add_position('center', '<middle_pos>', build_after = False)
    
    build_all()



def create_grammar():

    OBJ = role_for_words(OBJECT, OBJECT_NAMES)
    COL = role_for_words(COLOR, COLOR_NAMES)
    
    POSITIONS = role_for_words(ACTION, POSITION_NAMES)
    
    
    IS_ACTION = role_for_words(ACTION, ['is'])
    IS_NOROLE = role_for_words(NO_ROLE, ['is']) 
    
    TO_THE = union(
            sentence_to_roles('on the', [NO_ROLE, NO_ROLE])
    )
    
    THIS_IS = union(
            sentence_to_roles('this is', [ACTION, NO_ROLE]),
            sentence_to_roles('that is', [ACTION, NO_ROLE])      
    )
    
    THERE_IS = sentence_to_roles('there is', [NO_ROLE, NO_ROLE])
    
    DET = role_for_words(NO_ROLE, ['a', 'the'])
    
    GN = cat(DET, maybe(COL), OBJ)
    
    TO_THE_POSITION = cat(TO_THE, POSITIONS)
    
    return union(
        cat(THIS_IS, GN),
        cat(DET, OBJ, IS_ACTION, COL),
        cat(DET, OBJ, TO_THE_POSITION, IS_NOROLE, COL),
        cat(GN, IS_NOROLE, TO_THE_POSITION),
        cat(THERE_IS, GN, TO_THE_POSITION),
        cat(TO_THE_POSITION, union(IS_NOROLE, THERE_IS), GN)
    )
    



# Words
OBJECT_NAMES = []
COLOR_NAMES = []
POSITION_NAMES = []

# Concepts
CATEGORIES = []
POSITIONS = []
COLORS = []
  
# Word to concept mapping
OBJ_NAME_TO_CONCEPT = dict()
COLOR_NAME_TO_CONCEPT = dict()
POSITION_NAME_TO_CONCEPT = dict()


def clean_dataset():
    
    global OBJECT_NAMES, COLOR_NAMES, POSITION_NAMES
    global CATEGORIES, POSITIONS, COLORS
    global OBJ_NAME_TO_CONCEPT, COLOR_NAME_TO_CONCEPT, POSITION_NAME_TO_CONCEPT
    
    OBJECT_NAMES = []
    COLOR_NAMES = []
    POSITION_NAMES = []
    CATEGORIES = []
    POSITIONS = []
    COLORS = []
    OBJ_NAME_TO_CONCEPT = dict()
    COLOR_NAME_TO_CONCEPT = dict()
    POSITION_NAME_TO_CONCEPT = dict()


def build_all():
    
    global SENTENCE_TO_ROLES
    global SENTENCE_TO_PREDICATE
    global CONCEPT_LISTS
    global VISION_ENCODER

    CONCEPT_LISTS = [
            CATEGORIES,
            POSITIONS,
            COLORS
            ]
    
    VISION_ENCODER = one_hot_encoder.OneHotEncoder(CATEGORIES + POSITIONS + COLORS)

    SENTENCE_TO_ROLES = create_grammar()
    SENTENCE_TO_PREDICATE = {s : WordPredicate(s, r) for s, r in SENTENCE_TO_ROLES.items()}



def add_object(name, concept = None, build_after = True):
    
    if concept is None:
        concept = '<' + string.lower(name) + '_obj>'
    
    if name not in OBJECT_NAMES:
        OBJECT_NAMES.append(name)
        
    if concept not in CATEGORIES:
        CATEGORIES.append(concept)
        
    OBJ_NAME_TO_CONCEPT[name] = concept
    
    if build_after:
        build_all()
    
    
def add_position(name, concept = None, build_after=True):
    
    if concept is None:
        concept = '<' + string.lower(name) + '_pos>'
    
    if name not in POSITION_NAMES:
        POSITION_NAMES.append(name)
    
    if concept not in POSITIONS:
        POSITIONS.append(concept)
        
    POSITION_NAME_TO_CONCEPT[name] = concept
    
    if build_after:
        build_all()
       
        
        
def add_color(name, concept = None, build_after=True):
    
    if concept is None:
        concept = '<' + string.lower(name) + '_col>'
    
    
    if name not in COLOR_NAMES:
        COLOR_NAMES.append(name)
    
    if concept not in COLORS:
        COLORS.append(concept)
        
    COLOR_NAME_TO_CONCEPT[name] = concept
    
    if build_after:
        build_all()
    

create_dataset()    
    
