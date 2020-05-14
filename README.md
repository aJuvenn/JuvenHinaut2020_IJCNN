# JuvenHinaut2020_IJCNN: sentence grounding network with cross situational learning

Everything is in Python 2.7 (sorry). See requierements.txt for dependencies.

## Echo State Networks
(Can be executed for a little example)

- **echo_state_network.py  :** Definition of EchoStateNetwork class, which can only learn offline. 
- **FORCE_learning.py :** Definition of EchoStateNetworkRLS, which provides RLS online learning.

## Cross Situational Learning

### Networks
- **sentence_processing_network.py :** Definition of class SentenceProcessingNetwork, which takes sentences as input and can learn with RLS.
 
- **sentence_grounding_network.py :** Definition of class SentenceGroundingNetwork, which takes sentences as input and returns a list of object having different caracteristics, and provides a method to do cross situational learning.

### Test
(Can be executed)

- **sentence_grounding_test.py  :** Test of the sentence gounding network. A training is done creating a false vision and randomly combining sentences from sentences_to_predicate. Can be modified changing reservoir parameters, and CATEGORIES, POSITIONS, COLORS in **sentence_grounding_test_parameters.py** file.

- **sentence_grounding_test_plots.py  :** Plots the saved performance curves, or computes them. 

### Utils for test

- **sentence_grounding_test_parameters.py**: uses **grammar_manipulation.py** to create the grammar used for the test

- **recognized_object.py :** Definition of object returns by the module, representing an object seen on the picture

- **sentence_to_predicate.py :** Definition of the class WordPredicate, wich is construct by reading an annotated sentence from sentence_to_role.py. Definition of a set of sentences annotaded with the predicate they contain.

- **one_hot_encoder.py :** To transform object from a given set into a (0,...,1,...,0) vector.

