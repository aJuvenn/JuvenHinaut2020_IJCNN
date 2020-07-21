
import sentence_grounding_test as sgt
import numpy as np
import tqdm
import matplotlib.pyplot as plt



PROCESS_X = 1
PROCESS_NO_X = 2
PROCESS_NO_X_AND_SORT = 3


class SimpleModel:
    
    SPLIT_WORD = 'and'
        
    FUNC_WORDS = ['a', 'the', 'this', 'that', 
                  'is' , 'on', 'the', 'there']
   
    @staticmethod
    def word_list_to_sentence(word_list):
        output = ""
        for i, w in enumerate(word_list):
            if i != 0:
                output += " "
            output += w
        return output
     
    @staticmethod
    def process_func_1(sentence):
        words = [w if w not in SimpleModel.FUNC_WORDS else 'X' for w in sentence.split(' ') if w != '']
        return SimpleModel.word_list_to_sentence(words)
    
    @staticmethod
    def process_func_2(sentence):
        words = [w for w in sentence.split(' ') if w != '' and w not in SimpleModel.FUNC_WORDS]
        return SimpleModel.word_list_to_sentence(words)
    
    @staticmethod
    def process_func_3(sentence):
        words = [w for w in sentence.split(' ') if w != '' and w not in SimpleModel.FUNC_WORDS]
        words.sort()
        return SimpleModel.word_list_to_sentence(words)
    

    def __init__(self, process_func_id = PROCESS_NO_X_AND_SORT):
        self.processed_sentences = set()
        self.process_func = eval("SimpleModel.process_func_" + str(process_func_id))
    

    def learn(self, sentence):
        
        sub_sentences = sentence.split(SimpleModel.SPLIT_WORD)
        
        for sub_sentence in sub_sentences:
            processed_sentence = self.process_func(sub_sentence)
            self.processed_sentences.add(processed_sentence)
        
        
    def is_learned(self, sentence):
        
        sub_sentences = sentence.split(SimpleModel.SPLIT_WORD)
        
        for sub_sentence in sub_sentences:
            processed_sentence = self.process_func(sub_sentence)
            if processed_sentence not in self.processed_sentences:
                return False
            
        return True
    
    
    
    
def train_simple_model(model, nb_trainings):

    train_sentences, unused = sgt.random_train_sentences_and_predicates(nb_trainings)
    
    for i in range(nb_trainings):
        sentence = train_sentences[i]
        model.learn(sentence)
        

    return train_sentences

    

def test_simple_model(model, nb_tests, train_sentences = []):

    test_sentences, unused = sgt.random_test_sentences(nb_tests, train_sentences)
       
    nb_bad = 0.
    
    for sentence in test_sentences:
        if not model.is_learned(sentence):
            nb_bad += 1.
            
    return (nb_bad / nb_tests) * 100.
    


def simple_model_error_after_training(process_func_id, nb_trainings, nb_tests, nb_eror_averaging):
  
    errors = np.empty(nb_eror_averaging)

    for i in range(nb_eror_averaging):
        model = SimpleModel(process_func_id)
        train_sentences = train_simple_model(model, nb_trainings)
        error = test_simple_model(model, nb_tests, train_sentences)
        errors[i] = error
        
    return np.mean(errors)



def errors_in_function_of_nb_trainings(process_func_id, nb_tests, nb_error_averaging, start_nb_training, stop_nb_training, nb_training_step):
    
    tested_nb_trainings = np.arange(start_nb_training, stop_nb_training+1, nb_training_step, dtype = int)
    errors = np.empty_like(tested_nb_trainings, dtype = float)
    length = len(tested_nb_trainings)
    
    for i in tqdm.trange(length):
        nb_trainings = tested_nb_trainings[i]
        err = simple_model_error_after_training(process_func_id, nb_trainings, nb_tests, nb_error_averaging)
        print nb_trainings, err
        errors[i] = err
        
    return tested_nb_trainings, errors
    


def errors_in_function_of_nb_object(process_func_id, nb_trainings, nb_tests, nb_error_averaging, stop_nb_object):
    
    sgt.param.create_dataset()
    default_nb_objects = len(sgt.param.OBJECT_NAMES)
    
    tested_nb_objects = np.array(range(default_nb_objects, stop_nb_object+1), dtype = int)
    errors = np.empty_like(tested_nb_objects, dtype = float)

    for nb_objects in tqdm.trange(default_nb_objects, stop_nb_object+1):
        
        err = simple_model_error_after_training(process_func_id, nb_trainings, nb_tests, nb_error_averaging)
        print nb_objects, err
        errors[nb_objects - default_nb_objects] = err
         
        if nb_objects != stop_nb_object:
            sgt.param.add_object('obj_' + str(nb_objects))
            
    return tested_nb_objects, errors
        


def plot_errors_in_function_of_nb_trainings(tested_nb_trainings, errors_x, errors_no_x, errors_no_x_and_sort, x_plot_ticks=100., y_plot_ticks = 5.):

    plt.plot(tested_nb_trainings, errors_x, color = 'orange', linewidth = 3)
    plt.plot(tested_nb_trainings, errors_no_x, color = 'brown', linewidth = 3)
    plt.plot(tested_nb_trainings, errors_no_x_and_sort, color = 'green', linewidth = 3)
        
    plt.xlim(0., max(tested_nb_trainings))
    plt.ylim(0., 100.)
    plt.xticks(np.arange(0., max(tested_nb_trainings)+0.1, x_plot_ticks))
    plt.yticks(np.arange(0., 100.1, y_plot_ticks))
        

    plt.grid(linestyle='--')
    plt.legend(['X', 'No X', 'No X and sort'])
    
    plt.xlabel('Number of trainings')
    plt.ylabel('Error percentage')
    plt.savefig('error_curves_simple_model_nb_trainings.pdf')
    plt.show()
    
    
    
def plot_errors_in_function_of_nb_objects(tested_nb_objects, errors_x, errors_no_x, errors_no_x_and_sort, x_plot_ticks = 5., y_plot_ticks = 5.):
     
    tested_nb_objects = np.asarray(tested_nb_objects)
    
    plt.plot(tested_nb_objects, errors_x, color = 'orange', linewidth = 3)
    plt.plot(tested_nb_objects, errors_no_x, color = 'brown', linewidth = 3)
    plt.plot(tested_nb_objects, errors_no_x_and_sort, color = 'green', linewidth = 3)
             
    plt.xlim(min(tested_nb_objects), max(tested_nb_objects))
    plt.ylim(0., 100.)
    plt.xticks(np.arange(0., max(tested_nb_objects)+0.1, x_plot_ticks))
    plt.yticks(np.arange(0., 100.1, y_plot_ticks))
        
    plt.grid(linestyle='--')
    plt.legend(['X', 'No X', 'No X and sort'])
    
    plt.xlabel('Number of objects')
    plt.ylabel('Error percentage')
    plt.savefig('error_curves_simple_model_nb_obj.pdf')
    plt.show()




if __name__ == '__main__':
    
    NB_TRAININGS = 1000
    NB_TRAINING_STEPS = 50
    
    
    NB_TESTS = 1000
    NB_ERROR_AVERAGING = 10
    MAX_NB_OBJ = 50
    
    
    print "Error in function of the number of trainings..."

    errors = []
    
    for process_func_id in (PROCESS_X, PROCESS_NO_X, PROCESS_NO_X_AND_SORT):
        tested_nb_trainings, error = errors_in_function_of_nb_trainings(process_func_id, NB_TESTS, NB_ERROR_AVERAGING, 0, NB_TRAININGS, NB_TRAINING_STEPS)
        errors.append(error)
    
    plot_errors_in_function_of_nb_trainings(tested_nb_trainings, *errors)


    print "Error in function of the number of objects..."  
    
    errors = []
    
    for process_func_id in (PROCESS_X, PROCESS_NO_X, PROCESS_NO_X_AND_SORT):
        tested_nb_objects, error = errors_in_function_of_nb_object(process_func_id, NB_TRAININGS, NB_TESTS, NB_ERROR_AVERAGING, MAX_NB_OBJ)
        errors.append(error)
        
    plot_errors_in_function_of_nb_objects(tested_nb_objects, *errors)
    
    
    
    


