import numpy as np
from typing import Tuple

class TaskCognitive:
    ob_size = 0  
    act_size = 0
    def __init__(self, params:dict, batch_size:int) -> None:
        self._params = params
        self._batch_size = batch_size
        
    
    def dataset(self): 
        """ Return dataset . """
        return None
    
    @property
    def feature_and_act_size(self):
        return self.ob_size, self.act_size
    
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
    
    ob_size = 5 # number of inputs 
    act_size = 3 # number of outputs
    def __init__(self, params: dict, bath_size: int = 1) -> None:
        super().__init__(params, bath_size)
        
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
        inputs = np.zeros((f_time, batch_size, self.ob_size))
        target_outputs = np.zeros((f_time, batch_size, self.act_size))
        
        # labels = np.zeros((f_time, batch_size)) # if crossentropy
        
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
            output_one = np.zeros((full_interval_and_delay, batch_size, 1))   
            output_two = np.zeros((full_interval_and_delay, batch_size, 1)) 

            
            target_fixation = np.zeros((full_interval_and_delay, batch_size, 1))
            target_fixation[0:full_interval, ...] = fixation_array[...]
            
            indexes_context = np.where(context == 0)[0].astype(bool) # list 0 1 0 1 1 0 
            for j in range(batch_size):
                input_one[:, j] += np.random.normal(move_average[j], sigma, size=(full_interval, 1))
                input_two[:, j] += np.random.normal(color_average[j], sigma, size=(full_interval, 1))
                
                if indexes_context[j]:
                    output_one[:, j] += move_average_label[j]
                    output_two[:, j] += 1 - output_one[:, j]
                else: 
                    output_one[:, j] += move_average_label[j]
                    output_two[:, j] += 1 - output_one[:, j] 
                    
                
                #if indexes_context[j]:
                    #labels[i * full_interval_and_delay + full_interval: (i + 1) * full_interval_and_delay, j] \
                    #    += move_average_label[j] + 1
                #else:
                    #labels[i * full_interval_and_delay + full_interval: (i + 1) * full_interval_and_delay, j] \
                    #    += color_average_label[j] + 1
            inputs[i * full_interval_and_delay: (i + 1) * (full_interval_and_delay ) - delay] \
                                                            = np.concatenate((fixation_array, input_one, 
                                                                              input_two, context_one, 
                                                                              context_two), axis=-1)
                                                            
            target_outputs[i * full_interval_and_delay: (i + 1) * (full_interval_and_delay)] = np.concatenate((target_fixation, output_one, output_two), axis=-1)
        return inputs, target_outputs
    
class AntiSaccade(TaskCognitive):
    ob_size = 3 # number of inputs (fixation + two inputs)
    act_size = 3 # number of outputs (fixation + two)
    def __init__(self, params: dict, batch_size: int) -> None:
        super().__init__(params, batch_size)

    def dataset(self):
        # TODO: добавить выход правила (необходимо для сети, чтобы она могла определять решение задачи)
        
        dt = self._params['dt']
        #_time = self._params['time']
        batch_size = self._batch_size
        #t_delay = self._params['delay']
        #t_trial = self._params['trial']
        
        fixation = int(t_fixation / dt)
        target = int(t_target / dt)
        delay = int(t_delay / dt)
        trial = int(t_trial / dt)
        full_interval = fixation + target + delay + trial
        full_interval_and_delay = full_interval + delay
        inputs = np.zeros((f_time, batch_size, 1))
        outputs = np.zeros((f_time, full_interval, batch_size, 1))
        
        for j in range(batch_size):
            pass
        
        
        
        # comment return inputs, labels, ob_size, act_size

   
class CompareObjects(TaskCognitive):
    ob_size = 2 # number of inputs (fixation + one input)
    act_size = 2 # number of outputs (fixation + one output)
    def __init__(self, params: dict, bath_size:int) -> None:
        super().__init__(params, bath_size)
        
    def dataset(self):
        dt = self._params['dt']
        f_time = self.__params['time']
        batch_size = self._batch_size
        t_delay = self._params['delay']
        t_trial = self._params['trial']
        t_fixation = self._params['fixation']
        t_target = self._params['target']
        
        fixation = int(t_fixation / dt)
        target = int(t_target / dt)
        delay = int(t_delay / dt)
        trial = int(t_trial / dt)
        full_interval = fixation + target + delay + trial
        full_interval_and_delay = full_interval + delay
        inputs = np.zeros((f_time, batch_size, 1))
        outputs = np.zeros((f_time, full_interval, batch_size, 1))
        
        for j in range(batch_size):
            pass
        
        
class MultyTask(ContextDM, AntiSaccade, CompareObjects):
    TASKSDICT = dict([('ContextDM', ContextDM), 
                 ('AntiSaccade', AntiSaccade),
                 ('CompareObjects', CompareObjects)])
    def __init__(self, tasks: dict[str, dict], batch_size: int = 1) -> None:
        # tasks : dict(task_name -> parameters)
        for i, name in enumerate(tasks):
            if not name in self.TASKSDICT:
                raise KeyError(f'"{name}" not supported')
        self._tasks = tasks
        self._sorted_tasks()
        self._batch_size = batch_size
        self._task_list = []
        self._init_tasks() # init all tasks
        
    def _sorted_tasks(self):
        new_dict = dict()
        for key in sorted(self._tasks):
            new_dict[key] = self._tasks[key]
        self._tasks = new_dict
    def _init_tasks(self):
        self._task_list.clear()
        for key in self._tasks: 
            self._task_list.append(self.TASKSDICT[key](self._tasks[key], self._batch_size))
        
    def dataset(self, number_of_generations: int = 1) -> tuple[np.ndarray, np.ndarray]:
        number_of_tasks = len(self._tasks)
        choice_tasks = [i for i in range(number_of_tasks)]
        all_inputs, all_outputs = self._count_feature_and_act_size()            
        time = 1
        RuleMatrix = np.eye(number_of_tasks)
        inputs = np.zeros((0, self._batch_size, all_inputs))
        outputs = np.zeros((0, self._batch_size, all_outputs))
        sizes_all_tasks = self._feature_and_act_size_every_task()
        start_input_tasks = [1 + number_of_tasks]
        start_output_tasks = [1]
        for key in sizes_all_tasks:
            n_inputs, n_outputs = sizes_all_tasks[key]
            n_inputs -= 1 # -fix
            n_outputs -= 1 # -fix
            start_input_tasks.append(n_inputs + start_input_tasks[-1])
            start_output_tasks.append(n_outputs + start_output_tasks[-1])
        print(start_input_tasks)
        print(start_output_tasks)
            
        
        for i in range(number_of_generations):
            task_number = np.random.choice(choice_tasks)
            task_inputs, task_outputs = self._task_list[i].dataset()
            # 1. expansion of matrices
            inputs = np.concatenate((inputs, np.zeros((task_inputs.shape[0], self._batch_size, all_inputs))))
            outputs = np.concatenate((outputs, np.zeros((task_outputs.shape[0], self._batch_size, all_outputs))))
            # 2. put fixations
            inputs[-task_inputs.shape[0]:, :, 0] = task_inputs[-task_inputs.shape[0], :, 0]
            outputs[-task_inputs.shape[0]:, :, 0] = task_outputs[-task_outputs.shape[0], :, 0]
            # 3. put rule
            inputs[-task_inputs.shape[0]:, :, 1: 1 + number_of_tasks] += RuleMatrix[:, task_number]
            # 4. put stimuly and outputs
            
            
            
        
        return 1
    
    @property
    def feature_and_act_size(self) -> tuple[tuple[int, int], dict]:
        return (self._count_feature_and_act_size(), self._feature_and_act_size_every_task())
    
    def _count_feature_and_act_size(self) -> tuple[int, int]:
        all_inputs = 0
        all_outputs = 0
        for key in self._tasks:
            all_inputs += self.TASKSDICT[key].ob_size - 1 # minus fix
            all_outputs += self.TASKSDICT[key].act_size - 1 # minus_fix
            
        all_inputs += 1 + len(self._tasks) # fix + rule vector
        all_outputs += 1 # fix
        return (all_inputs, all_outputs)
    
    def _feature_and_act_size_every_task(self):
        sizes = dict()
        for key in self._tasks:
            sizes[key] = (self.TASKSDICT[key].ob_size, self.TASKSDICT[key].act_size)
        return sizes
    
    @property
    def tasks(self) -> dict:
        return self._tasks
    
    @tasks.setter
    def tasks(self, tasks) -> None:
        self.__init__(tasks)  
        
    def GetTask(self, key) -> dict:
        if not key in self._tasks:
            raise KeyError()
        return self._tasks[key]
    
    def SetTask(self, key: str, params:dict):
        if not key in self._tasks:
            raise KeyError()
        self._tasks[key] = params
        self._init_tasks()
    
    def __getitem__(self, index:int) -> tuple:
        if index < 0 and index > len(self._tasks) - 1:
            raise IndexError(f'index not include in [{0}, {len(self._tasks)}]')
        for i, key in enumerate(self._tasks):
            if index == i:
                return key, self._tasks[key]
            
    def __setitem__(self, index: int, new_task: tuple):
        if index < 0 and index > len(self._tasks) - 1:
            raise IndexError(f'index not include in [{0}, {len(self._tasks)}]')
        new_name, new_parameters = new_task
        if not new_name in self.TASKSDICT:
            raise KeyError(f'"{new_name}" not supported')
        for i, key in enumerate(self._tasks):
            if index == i:
                old_key = key
                break
        del self._tasks[old_key]
        self._tasks[new_name] = new_parameters
        self._init_tasks()
        
    def __len__(self):
        return len(self._tasks)
                

t_fixation = .3
t_target = .35
t_trial = .75
t_delay = .3 #300 - 1500 (.3s - 1.5s)

f_time = 30000
dt = 1e-3
dict_ContextDM = dict([('sigma', .1), 
              ('fixation', t_fixation),
              ('target', t_target),
              ('delay', t_delay),
              ('trial', t_trial),
              ('time', f_time),
              ('dt', dt)])
dict_Compare = dict([('dt', 0.001)])
new_task = 'CompareObjects'
tasks = dict([('ContextDM', dict_ContextDM),
              (new_task, dict_Compare),
              ('AntiSaccade', dict_Compare)])
multytas = MultyTask(tasks)
#print(multytas.tasks)

#multytas[0] = (new_task, dict_Compare)
for i in range(len(multytas)):
    print(multytas[i])

#multytas.dataset()

#multytas.dataset()

params = dict([('sigma', .1), 
              ('fixation', t_fixation),
              ('target', t_target),
              ('delay', t_delay),
              ('trial', t_trial),
              ('time', f_time),
              ('dt', dt)])

CDM = ContextDM(params)

print(multytas.feature_and_act_size)