import numpy as np



class TaskCognitive:
    
    def __init__(self, params:dict, batch_size:int) -> None:
        self._params = params
        self._batch_size = batch_size
        self._ob_size = 0  
        self._act_size = 0
    
    def dataset(self): 
        """ Return dataset . """
        return None
    
    @property
    def feature_and_act_size(self):
        return self._ob_size, self._act_size
    
    @property
    def task_parameters(self): 
        """ Property to return the Tasks parameters . """ 
        return self._params

    @task_parameters.setter
    def task_parameters(self, params:dict):
        """ Setter for task parameters . """
        self._params = params
    
    @property
    def batch_size(self): 
        """ The number of batches returned by this query . """
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size: int):
        """ Setter for _setter_size . """
        
        self._batch_size = batch_size
    
    
    
    
class ContextDM(TaskCognitive):
    
    def __init__(self, params: dict, bath_size: int) -> None:
        super().__init__(params, bath_size)
        self._ob_size = 5 # number of inputs 
        self._act_size = 3 # number of outputs
    def dataset(self): 
        """ Create a dataset containing the training data . """
        
        # TODO: добавить выход правила (необходимо для сети, чтобы она могла определять решение задачи)
        sigma = self._params['sigma']
        t_fixation = self._params['fixation']
        t_target = self._params['target']
        t_delay = self._params['delay']
        t_trial = self._params['trial']
        f_time = self._params['time']
        batch_size = self._batch_size
        dt = self._params['dt']
        
            # two stimuly and two (m.b. one) context signal = 4 (3) inputs and fixation 
         
        
        fixation = int(t_fixation / dt)
        target = int(t_target / dt)
        delay = int(t_delay / dt)
        trial = int(t_trial / dt)
        full_interval = fixation + target + delay + trial
        full_interval_and_delay = full_interval + delay
        if (f_time < full_interval_and_delay):
            f_time = full_interval_and_delay
        if (f_time % full_interval_and_delay != 0):
            f_time -= f_time % full_interval_and_delay
        number_of_trials = int(f_time / full_interval_and_delay)
        context = np.zeros((2, batch_size))
        inputs = np.zeros((f_time, batch_size, self._ob_size))
        labels = np.zeros((f_time, batch_size)) # if crossentropy
        
        for i in range(number_of_trials):
            context[0, :] = np.random.choice([0, 1], size=batch_size)
            context[1, :] = 1 - context[0, :]
            move_average = np.random.uniform(0, 1, size=batch_size)
            color_average = np.random.uniform(0, 1, size=batch_size)
            move_average_label = move_average > 0.5
            move_average_label = move_average_label.astype(np.longlong)
            color_average_label = color_average > 0.5
            color_average_label = color_average_label.astype(np.longlong)
            fixation_array = np.ones((full_interval, batch_size, 1)) # m.b full_interva - time of between trials
            context_one = np.ones((full_interval, batch_size, 1)) 
            context_one[:, :, 0] *= context[0]
            context_two = np.ones((full_interval, batch_size, 1)) 
            context_two[:, :, 0] *= context[1]
            input_one = np.zeros((full_interval, batch_size, 1))   
            input_two = np.zeros((full_interval, batch_size, 1)) 
            indexes_context = np.where(context == 0)[0].astype(bool) # list 0 1 0 1 1 0 
            for j in range(batch_size):
                input_one[:, j] += np.random.normal(move_average[j], sigma, size=(full_interval, 1))
                input_two[:, j] += np.random.normal(color_average[j], sigma, size=(full_interval, 1))
                if indexes_context[j]:
                    labels[i * full_interval_and_delay + full_interval: (i + 1) * full_interval_and_delay, j] \
                        += move_average_label[j] + 1
                else:
                    labels[i * full_interval_and_delay + full_interval: (i + 1) * full_interval_and_delay, j] \
                        += color_average_label[j] + 1
            inputs[i * full_interval_and_delay: (i + 1) * (full_interval_and_delay ) - delay] \
                                                            = np.concatenate((fixation_array, context_one, 
                                                                              context_two, input_one, 
                                                                              input_two), axis=-1)
        return inputs, labels

class AntiSaccade(TaskCognitive):
    def __init__(self, params: dict, batch_size: int) -> None:
        super().__init__(params, batch_size)

    def dataset(self):
        # TODO: добавить выход правила (необходимо для сети, чтобы она могла определять решение задачи)
        
        dt = self._params['dt']
        f_time = self.__params['time']
        batch_size = self._batch_size
        t_delay = self._params['delay']
        t_trial = self._params['trial']
        ob_size = 3 # number of inputs (fixation + two inputs)
        act_size = 3 # number of outputs (fixation + two)
        fixation = int(t_fixation / dt)
        target = int(t_target / dt)
        delay = int(t_delay / dt)
        trial = int(t_trial / dt)
        full_interval = fixation + target + delay + trial
        full_interval_and_delay = full_interval + delay
        inputs = np.zeros((full_interval, batch_size, 1))
        
        
        # comment return inputs, labels, ob_size, act_size

    
    
#import numpy as np
#import matplotlib.pyplot as plt
#t_fixation = .3
#t_target = .35
#t_trial = .75
#t_delay = .3 #300 - 1500 (.3s - 1.5s)
#
#f_time = 10000
#dt = 1e-3
#params = dict([('sigma', 0.5), 
#              ('fixation', t_fixation),
#              ('target', t_target),
#              ('delay', t_delay),
#              ('trial', t_trial),
#              ('time', f_time),
#              ('dt', dt)])
#batch_size = 100
#CDM_task = ContextDM(params, batch_size)
#
#inputs, labels, ob_size, act_size = CDM_task.dataset()
#
#print('Good')