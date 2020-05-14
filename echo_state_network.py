# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:20:09 2019

@author: ajuven
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as npmat
import pickle
import sklearn.linear_model as sklm
import sklearn.decomposition as skdcmp
import mpl_toolkits.mplot3d 


class EchoStateNetwork:
    
    """
        Machine learning network from reservoir computing algorithm family 
    """


    
    
    """
                            CREATION OF THE NETWORK
    """

 
    def __init__(self, input_size, output_size, 
                 reservoir_size = 300, 
                 spectral_radius = 1.25,
                 leaking_rate = 0.3,
                 linear_model = None,
                 average_nb_connexions = None, 
                 use_feedback = False,
                 use_raw_input = True,
                 feedback_scaling = 1.,
                 input_scaling = 1.,
                 output_activation_function = None,
                 output_activation_inverse_function = None,
                 reservoir_noise_scaling  = 0.,
                 input_sparsity = 0.):
        
        """
            Creates and returns a new random network.
            
            Parameters:
            
                input_size (int > 0): dimension of the input
                output_size (int > 0): dimension of the output
            
            Optional parameters:            
            
                reservoir_size (int > 1): number of neurones in the reservoir 
                spectral_radius (float > 0.): scaling of the internal reservoir matrix weights
                leaking_rate (float between 0. and 1.): scales neurone outputs evolution speed
                
                linear_model: must one of the sklearn.linear_model classifier, used to make the network learn.
                                If not set, pseudo inverse method is used during learning phase
                
                average_nb_connexions (int > 0, or None): Average number of input a neuron from the reservoir receives from
                                other reservoir neurons. If set to None, each neuron is connected to others.
                
                use_feedback (bool): If set to True, reservoir neurons receive also as input the whole network output
                
                use_raw_input (bool): Whether raw input is used to compute output or just resevoir neurons                
                
                feedback_scaling (float > 0.): scaling of the feedback matrix weights
                
                input_scaling (float > 0.): scaling of the input matrix weights
                
                
                output_activation_function (numpy function): function f such as "network_output = f(output_weight * network_state)"
                                            default is identity
                                            
                output_activation_inverse_function (numpy function): if output_activation_function is a non linear function f,
                        providing its inverse f^-1 allows to solve "f^-1(network_output) = output_weight * network_state"
                        during learning phase. (Example : use np.arctanh if you used np.tanh for the output_activation_function field)  
                        
                        
                reservoir_noise_scaling (float): TODO  
                
                input_sparsity (float between 0. and 1.) : TODO      NOT IMPLEMENTED                 
                        
        """
        
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        self.linear_model = linear_model        
        self.use_feedback = use_feedback   
        self.output_activation_function = output_activation_function
        self.output_activation_inverse_function = output_activation_inverse_function
        
        self.reservoir_weights = EchoStateNetwork._rand_matrix(reservoir_size, 
                                                               reservoir_size, 
                                                               spectral_radius, 
                                                               average_nb_connexions,
                                                               gaussian = True)
        
        self.input_scaling = input_scaling                                    
        self.input_weights = input_scaling * EchoStateNetwork._rand_matrix(reservoir_size, input_size)
        self.use_raw_input = use_raw_input 
        
        if use_raw_input:
            self.state_size = 1 + reservoir_size + input_size
        else:
            self.state_size = 1 + reservoir_size    
        
        self.state = npmat.zeros((self.state_size, 1))          
        self.output_weights = npmat.zeros((output_size, self.state_size))
        
        if use_feedback:
            self.feedback_scaling = feedback_scaling
            self.feedback_weights = feedback_scaling * EchoStateNetwork._rand_matrix(reservoir_size, output_size)
        
        self.reservoir_values = npmat.zeros((reservoir_size, 1))
        self.output_values = npmat.zeros((output_size, 1))
    
        self.reservoir_noise_scaling = reservoir_noise_scaling
    
        
    @classmethod
    def _rand_matrix(cls, nb_rows, nb_columns, spectral_radius = None, average_nb_connexions = None, gaussian = False):
        
        if gaussian:
            M = np.asmatrix(np.random.normal(size=(nb_rows, nb_columns)))
        else:
            M = np.asmatrix(np.random.uniform(-1., 1., size=(nb_rows, nb_columns)))
            
        if average_nb_connexions is not None:
            nb_zeros = max(0, int(nb_rows * (nb_columns - average_nb_connexions)))
            np.put(M, np.random.choice(range(nb_rows * nb_columns), nb_zeros, replace = False), 0.)
           
        if spectral_radius is None:
            return M
            
        actual_radius = max(abs(np.linalg.eigvals(M)))
        
        return (spectral_radius/actual_radius) * M



    def increase_input_size(self, nb_added_input = 1):
        """
            Increases the number of inputs the network needs
        """
        
        added_input_weights = self.input_scaling * EchoStateNetwork._rand_matrix(self.reservoir_size, nb_added_input)
        self.input_weights = npmat.hstack([self.input_weights, added_input_weights])
        self.input_size += nb_added_input
        
        if not self.use_raw_input:
            return
        
        self.state_size += nb_added_input
        self.state = npmat.vstack([self.state, npmat.zeros((nb_added_input, 1))])
        self.output_weights = npmat.hstack([self.output_weights, npmat.zeros((self.output_size, nb_added_input))])



    def increase_output_size(self, nb_added_output = 1):
        """
            Increases the number of inputs the network needs
        """
        
        self.output_size += nb_added_output
        self.output_weights = npmat.vstack([self.output_weights, npmat.zeros((nb_added_output, self.state_size))])
        


    def add_noise_to_output_weights(self, noise_scaling):
        """
            Adds noise to output weights (you don't say)
        """
        
        noise = noise_scaling * EchoStateNetwork._rand_matrix(self.output_size, self.state_size)
        self.output_weights += noise
        
        




    """
                                SETTERS, GETTERS
    """


    def set_reservoir_noise_scaling(self, noise):
        self.reservoir_noise_scaling = noise


    def get_reservoir_memory(self):
        return self.reservoir_values.copy()
    
    def set_reservoir_memory(self, memory):
        self.reservoir_values = np.asmatrix(memory).reshape(self.reservoir_size, 1)




        
    """
                               RUNNING THE NETORK ON DATA
    """
    
        
        
    def next_output(self, new_input = []):
        """
            Provides new_input to the network and computes it's next output
            
            Parameter:
                new_input float or numpy column, depending on network input dimension
                
            Output : 
                float (if self.output_size == 1), otherwise numpy matrix column
        """
        
    
        # in case new_input is a scalar value or an array, it is viewed as a matrix column
        new_input = np.asmatrix(new_input).reshape((self.input_size, 1))
        
        
        #shorter variable names to make formulas readable
        u = new_input
        x = self.reservoir_values
        y = self.output_values
        Wres = self.reservoir_weights
        Win = self.input_weights
        lr = self.leaking_rate
        
            
        if self.use_feedback:
            Wfb = self.feedback_weights
            new_x = Wres * x + Win * u + Wfb * y
        else:
            new_x = Wres * x + Win * u
                
        if self.reservoir_noise_scaling != 0.:
            new_x += self.reservoir_noise_scaling * np.random.uniform(-1., 1., size = (self.reservoir_size, 1))
            
        self.reservoir_values = (1. - lr) * x + lr * np.tanh(new_x) 
            
    
        if self.use_raw_input:
            self.state = np.vstack((1., self.reservoir_values, new_input))
        else:
            self.state = np.vstack((1., self.reservoir_values))
    
        self.output_values = self._output_weight_sum_and_activation(self.state)
                      
        if self.output_size == 1:
            # A scalar is returned instead of a (1, 1) dimension matrix
            return self.output_values[0, 0]

        return self.output_values.copy()
        

        
    def _output_weight_sum_and_activation(self, state):
        
        output = self.output_weights * state        
        
        if self.output_activation_function is None:
            return output
        
        return self.output_activation_function(output)
        
     
        
    def run(self, inputs):
        """
            Provides the network a list of inputs and returns computation output serie
            
            Parameter:            
                inputs : list of inputs to feed to the network
                
            Output: 
                numpy array of network outputs
        """
        
        outputs = []

        for inpt in inputs:
            outputs.append(self.next_output(inpt))
            
        return np.asarray(outputs)

        
        
    def generate_outputs(self, nb_outputs = 1, initial_value = None):
        """
            Makes run the network nb_outputs time : if network needs inputs, its last output is used.
            If initial_value is not None, it is used as first input to feed the network
            
            Output : 
                numpy array of network outputs
        """
        
        outputs = []
        
        if initial_value != None:
            next_input = initial_value
        else:
            next_input = self.output_values
        
        for i in range(nb_outputs):
            next_input = self.next_output(next_input if self.input_size != 0 else [])
            outputs.append(next_input)
        
        return np.asarray(outputs)
        
    
    

    def last_output(self, inputs):
        
        self.run(inputs[0:-1])
        return self.next_output(inputs[-1])
        

    
    def classify_inputs(self, inputs):

        return np.argmax(self.last_output(inputs))

    
    
        
    def reset_memory(self):
        """
            Resets network memory by setting internal values to zero
        """
        self.reservoir_values = npmat.zeros((self.reservoir_size, 1))
        
    
    
    
    def record_activity(self, inputs):
        """
            Returns states and outputs through time after beeing fed inputs          
        """
        
        nb_inputs = len(inputs)
        states = npmat.empty((self.state_size, nb_inputs))
        outputs = npmat.empty((self.output_size, nb_inputs))

        for i in range(nb_inputs):
           self.next_output(inputs[i])
           states[:, i] = self.state
           outputs[:, i] = self.output_values
           
        return states, outputs
        
        
    
    def plot_state_record(self, inputs_list, legends = None, texts = None):
        """
            Makes a 3D plot of reservoir state thought time (using PCA) after beeing fed
            inputs form inputs_list (a reset is made between each run)
        """
        
        states_list = []
        
        for inputs in inputs_list:
            states, outputs = self.record_activity(inputs)
            states_list.append(states)
            self.reset_memory()
            
        all_states = np.hstack(states_list)
        pca = skdcmp.PCA(n_components = 3)
        reduces_states = pca.fit_transform(all_states.T)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        offset = 0
        
        for i, inputs in enumerate(inputs_list):
            
            nb_inputs = len(inputs)
            xs = reduces_states[offset : offset + nb_inputs, 0]
            ys = reduces_states[offset : offset + nb_inputs, 1]
            zs = reduces_states[offset : offset + nb_inputs, 2]
            ax.plot(xs, ys, zs)
            offset += nb_inputs
            
            if texts is None:
                continue
            
            for j in range(nb_inputs):
                ax.text(xs[j], ys[j], zs[j], texts[i][j], fontsize = 20)
            
            
        if legends is not None:
            ax.legend(legends, fontsize = 24)
        
        plt.show()
        
    
    
    
    """               
                        MAKING THE NETWORK LEARN FROM DATA        
    """
    
    
    
    
    
    
    def learn_series(self, input_series, target_output_series, nb_washing_steps_int_or_list = 0, reset_memory = True):
        """
            Learns from a set of inputs.
            nb_washing_steps_int_or_list : if int, it is the same for each sequence of input, if it's a list,
                                            each serie of input has its own number of washing steps
            reset _memory : if True, network state is reset to zero after each input series.
        """
        
        nb_series = len(input_series)
        
        if len(target_output_series) < nb_series:
            raise Exception, "Number of target outputs does not match inputs"
        
        state_recording_matrices = []
        output_target_matrices = [] 
        
        for i in range(nb_series):
            
            inputs = map(np.asmatrix, input_series[i])
            output_formater = lambda x : np.asmatrix(x).reshape((self.output_size, 1))
            target_outputs = map(output_formater, target_output_series[i])
            training_size = len(inputs)
        
            if hasattr(nb_washing_steps_int_or_list, '__iter__'):
                # nb_washing_steps is a list of different values for each serie
                nb_washing_steps = nb_washing_steps_int_or_list[i]
            else:
                nb_washing_steps = nb_washing_steps_int_or_list
                
        
            if nb_washing_steps >= training_size or len(target_outputs) < training_size:
                raise Exception, "Invalid training/washing set sizes"
            
            state_rec, output_rec = self._record_state_matrices(inputs, target_outputs, nb_washing_steps)      
            state_recording_matrices.append(state_rec)
            output_target_matrices.append(output_rec)
        
            if reset_memory:
                self.reset_memory()
        
        state_recording_matrix = np.hstack(state_recording_matrices)
        output_target_matrix = np.hstack(output_target_matrices)
        
        self._learn_from_state_recordings(state_recording_matrix, output_target_matrix)
        
    


    def learn(self, inputs, target_outputs, nb_washing_steps = 0):
        """
            Makes the network learn from provided dataset. 
            The first nb_washing_step values from the dataset are not used to learn, but just to make
            the network run on data. If return_results is True, learning recording is returned : a matrix 
            storing successive input and internal neurons outputs in each columns, and a matrix storing 
            successive network output in each column.
        """
        
        return self.learn_series([inputs], [target_outputs], nb_washing_steps_int_or_list = nb_washing_steps, reset_memory = False)




    def _record_state_matrices(self, inputs, target_outputs, nb_washing_steps = 0):
        
        training_size = len(inputs)
        
        state_recording_matrix = npmat.empty((self.state_size, training_size - nb_washing_steps))
        
        for i in range(training_size):
      
            self.next_output(inputs[i])
            
            if self.use_feedback:
                #teacher forcing : output_values are changed for the next self.next_output() call
                self.output_values = target_outputs[i]
            
            if i < nb_washing_steps:
                continue
                                    
            state_recording_matrix[:, i - nb_washing_steps] = self.state
            
        output_target_matrix = np.hstack(target_outputs[nb_washing_steps : training_size])

        return state_recording_matrix, output_target_matrix
       
    
    
    def _learn_from_state_recordings(self, state_recording_matrix, output_target_matrix):
        
        if self.output_activation_inverse_function != None:
            # "f(Weight * State) = Target", so "Weight * State = f^-1(Target)" is solved
            output_target_matrix = self.output_activation_inverse_function(output_target_matrix)
        
        if self.linear_model is not None:
            # Linear model provided during instanciation is used
            X = state_recording_matrix
            Ytarget = output_target_matrix
            # Learning of the model (first row of X, which contains only ones, is removed) 
            self.linear_model.fit(X[1:, :].T, Ytarget.T)
        
            # linear_model provides Matrix A and Vector b such as A * X[1:, :] + b ~= Ytarget       
            A = np.asmatrix(self.linear_model.coef_)
            b = np.asmatrix(self.linear_model.intercept_).T
        
            # Then the matrix W = "[b | A]" statisfies "W * X ~= Ytarget"
            self.output_weights = np.hstack([b, A])

        else:    
            # Default : pseudo inverse method
            self.output_weights = output_target_matrix * np.linalg.pinv(state_recording_matrix)
        


    """               
                        SAVING AND LOADING THE NETWORK
    """


    def save_into_file(self, filename):
        
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file)
            

                   
    @classmethod 
    def load_from_file(cls, filename):
        
        with open(filename, 'rb') as input_file:
            return pickle.load(input_file)
           
          

if __name__ == '__main__':

    train_size = 2000
    test_size = 400
  
    test_sequence = np.loadtxt('MackeyGlass_t17.txt')    
    
    nw = EchoStateNetwork(0, 1, 
                          reservoir_size = 400, 
                          spectral_radius = 1.3, 
                          average_nb_connexions = 5, 
                          use_feedback = True, 
                          leaking_rate = 0.3,
                          linear_model = sklm.Ridge(1e-5))
                       
                       
    nw.learn(train_size * [[]], test_sequence[0:train_size], nb_washing_steps=int(0.05 * train_size))
        
    state_copy = nw.get_reservoir_memory()
    
    outputs = nw.generate_outputs(test_size)    
        
    fig, ax = plt.subplots(figsize=(30., 20.))    
    ax.plot(test_sequence[train_size: train_size + test_size])
    ax.plot(outputs)    
    ax.legend(["Targeted signal", "Generated signal"], fontsize = 24)
    plt.show()
    
    nw.set_reservoir_memory(state_copy)
    nw.plot_state_record([400 * [[]]], ["reservoir state"])
    
    
    
    
    
    
    
    
    
    
    
