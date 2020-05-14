# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:08:35 2019

@author: ajuven
"""

import numpy as np
from echo_state_network import EchoStateNetwork
import matplotlib.pyplot as plt



class EchoStateNetworkRLS(EchoStateNetwork):
    """
        Online reservoir using RLS algorithm to learn
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
                 input_sparsity = 1.,
                 
                 ridge_coef = 1e-6):
        
        EchoStateNetwork.__init__(self, input_size, output_size, 
                 reservoir_size = reservoir_size, 
                 spectral_radius = spectral_radius,
                 leaking_rate = leaking_rate,
                 linear_model = linear_model,
                 average_nb_connexions = average_nb_connexions, 
                 use_feedback = use_feedback,
                 use_raw_input = use_raw_input,
                 feedback_scaling = feedback_scaling,
                 input_scaling = input_scaling,
                 output_activation_function = output_activation_function,
                 output_activation_inverse_function = output_activation_inverse_function,
                 reservoir_noise_scaling = reservoir_noise_scaling,
                 input_sparsity = input_sparsity)
        
        self.ridge_coef = ridge_coef
        self.reset_correlation_matrix()
        
        
        
    def reset_correlation_matrix(self):
        self.state_corr_inv = np.asmatrix(np.eye(self.state_size)) / self.ridge_coef
    
        
        
    def learn_from_current_state_RLS(self, target_output, indexes = None):
        """
            Learns to return the target when beeing in its current state.
            If indexes is not None, only the provided output indexes are learned
        """
        
        target_output = np.asmatrix(target_output).reshape(-1, 1)
        last_output = self.output_values
        error = last_output - target_output
        
        self.state_corr_inv = EchoStateNetworkRLS._new_correlation_matrix_inverse(self.state, self.state_corr_inv)
        
        if indexes == None:
            self.output_weights -= error * (self.state_corr_inv * self.state).T
        else:
            self.output_weights[indexes] -= error[indexes] * (self.state_corr_inv * self.state).T
        
        
    def learn_new_input_RLS(self, new_input, target_output):
        """
            Feeds the input to the network and learns to return the target
        """
        
        self.next_output(new_input)
        self.learn_from_current_state_RLS(target_output)
        
    
    def learn_inputs_RLS(self, inputs, target_outputs, nb_washing_steps = 0):
        """
            Feeds the inputs through time to the netwrok and makes it learn
            to return th targets through time. The first nb_washing_steps step time
            are not learned, just feeded.
        """
        
        for i in range(nb_washing_steps):
            self.next_output(inputs[i])
        
        nb_trainings = len(inputs) - nb_washing_steps
    
        for i in range(nb_trainings):
            self.learn_new_input_RLS(inputs[i + nb_washing_steps], target_outputs[i + nb_washing_steps])


    
    @staticmethod
    def _new_correlation_matrix_inverse(new_data, old_corr_mat_inv):
        """
            If old_corr_mat_inv is an approximation for the correlation
            matrix inverse of a dataset (p1, ..., pn), then the function 
            returns an approximatrion for the correlation matrix inverse
            of dataset (p1, ..., pn, new_data)
            
            TODO : add forgetting parameter lbda
        """    
        
        P = old_corr_mat_inv
        x = new_data
        
        # TODO : numerical instabilities if tmp is not computed first (order of multiplications)  
        tmp = x.T * P
        P = P - (P * x * tmp)/(1. + tmp * x)
        
        return P
    
    
    

    
  
         
if __name__ == '__main__':

    train_size = 5000
    test_size = 1000
  
    test_sequence = np.tile(np.loadtxt('MackeyGlass_t17.txt'), 10)    
    
    nw = EchoStateNetworkRLS(1, 1, 
                          reservoir_size = 400, 
                          spectral_radius = 1.3, 
                          average_nb_connexions = 3, 
                          use_feedback = True, 
                          leaking_rate = 0.3,
                          use_raw_input = True,
                          ridge_coef = 1e-8)
        
    nw.learn_inputs_RLS(test_sequence[0:train_size], test_sequence[1:train_size+1], nb_washing_steps = int(0.05 * train_size))
    
    outputs = nw.generate_outputs(test_size)
    
    fig, ax = plt.subplots(figsize=(30., 20.))    
    ax.plot(test_sequence[train_size + 1 : train_size + test_size + 1])
    ax.plot(outputs)    
    ax.legend(["target signal", "generated signal"])
    plt.show()
    
    
