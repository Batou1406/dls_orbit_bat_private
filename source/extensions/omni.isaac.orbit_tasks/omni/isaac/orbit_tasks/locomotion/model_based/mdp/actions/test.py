"""
Unit testing for model based actions
"""                
import torch
from model_base_controller import modelBaseController, samplingController
# from .model_base_controller import *

verbose = 0

class modelBaseControllerTest():

    controller : modelBaseController

    def __init__(self, controller: modelBaseController):
        self.controller = controller()

        device='cpu'
        num_env=2
        num_leg=3
        time_horizon=4
        dt_out=0.5
        decimation=5
        dt_in=0.1
        p_default=torch.tensor([[1,1,1],[0,0,0],[0.1, 0.2, 0.5]]).unsqueeze(0).expand(num_env, -1, -1)
        self.controller.late_init(device, num_env, num_leg, time_horizon, dt_out, decimation, dt_in, p_default)

    def test_optimize_latent_variable(self):
        raise NotImplementedError


    def test_compute_control_output(self):
        raise NotImplementedError


    def test_gait_generator(self):
        
        prin1('\n--- Test Gait Generator ---')

        prin1('\nTest 1')
        # f : leg frequency : shape(batch, parallel_rollout, num_leg)
        # shape (batch=2, leg=3)
        f = torch.tensor([[1, 2, 3],[0.5, 0.1, 1.0]])
        prin2('     f shape :', f.shape, ' -  wished sized (2,3)')

        # d : leg duty cyle : shape(batch, num_leg)
        d = torch.tensor([[0.5, 1.5, 0.7],[0.5, 0.1, 1.0]])
        prin2('     d shape :', d.shape, ' -  wished sized (2,3)')

        # p : leg phase : shape(batch, num_leg)
        phase = torch.tensor([[0, 0.0, 0],[0, 0.1, 1.2]])
        prin2(' phase shape :', phase.shape, ' -  wished sized (2,3)') 

        time_horizon = 4
        dt=0.2
        prin2('Time Horizon :', time_horizon, ' -  dt :', dt)

        # Compute the value
        c, new_phase = self.controller.gait_generator(f, d, phase, time_horizon, dt)

        prin2('Contact sequence shape :',c.shape,' -  wished sized (2,3,4)')
        prin2('New phase shape :',new_phase.shape,' -  wished sized (2,3)')

        # Assert Test
        if (c == torch.tensor([[[1,1,0,0],[1,1,1,1],[1,1,0,1]],[[1,1,1,1],[0,0,0,0],[1,1,1,1]]])).all():
            prin1('Correct Contact sequence has been computed')
        else:
            raise ValueError('Invalid contact sequence for Gait Generator - test 1')
        
        if ((new_phase - torch.tensor([[0.2, 0.4, 0.6],[0.1, 0.12, 0.4]])).sum()) < 1e-7:
            prin1('Correct phase increment has been computed')
        else:
            raise ValueError('Invalid phase increment for Gait Generator - test 1')


        prin1('\nTest 2')
        # f : leg frequency : shape(batch, parallel_rollout, num_leg)
        # shape (batch=2,rollout=3, leg=4)
        f = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],[[0.5, 0.1, 1.0, 2.35], [0.5, 0.1, 1.0, 2.35], [0.3, 0.3, 1.0, 2.35]]])
        prin2('     f shape :', f.shape, ' -  wished sized (2,3,4)')

        # d : leg duty cyle : shape(batch, parallel_rollout, num_leg)
        d = torch.tensor([[[0.5, 1.5, 0.7, 0.2], [0.5, 1.5, 0.7, 0.2], [0.5, 1.5, 0.7, 0.2]],[[0.5, 0.1, 1.0, 0.5], [0.5, 0.1, 1.0, 0.5], [0.0, 0.8, 1.0, 0.5]]])
        prin2('     d shape :', d.shape, ' -  wished sized (2,3,4)')

        # p : leg phase : shape(batch, parallel_rollout, num_leg)
        phase = torch.tensor([[[0, 0.0, 0, 0.3], [0, 0.0, 0, 0.3], [0, 0.0, 0, 0.3]],[[0, 0.1, 1.2, 0.5], [0, 0.1, 1.2, 0.5], [0.025, 0.6, 1.2, 0.5]]])
        prin2(' phase shape :', phase.shape, ' -  wished sized (2,3,4)') 

        time_horizon = 5
        dt=0.2
        prin2('Time Horizon :', time_horizon, ' -  dt :', dt)

        # Compute the value
        c, new_phase = self.controller.gait_generator(f, d, phase, time_horizon, dt)

        prin2('Contact sequence shape :',c.shape,' -  wished sized (2,3,4,5)')
        prin2('New phase shape :',new_phase.shape,' -  wished sized (2,3,4)')

        # Assert Test
        if (c == torch.tensor([[[[1,1,0,0,1],[1,1,1,1,1],[1,1,0,1,1],[1,0,0,0,0]], [[1,1,0,0,1],[1,1,1,1,1],[1,1,0,1,1],[1,0,0,0,0]], [[1,1,0,0,1],[1,1,1,1,1],[1,1,0,1,1],[1,0,0,0,0]]], [[[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[0,1,0,1,0]], [[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[0,1,0,1,0]], [[0,0,0,0,0],[1,1,1,0,0],[1,1,1,1,1],[0,1,0,1,0]]]])).all():
            prin1('Correct Contact sequence has been computed')
        else:
            raise ValueError('Invalid contact sequence for Gait Generator - test 2')
        
        if ((new_phase - torch.tensor([[[0.2, 0.4, 0.6, 0.1],[0.2, 0.4, 0.6, 0.1],[0.2, 0.4, 0.6, 0.1]],[[0.1, 0.12, 0.4, 0.97], [0.1, 0.12, 0.4, 0.97], [0.085, 0.66, 0.4, 0.97]]])) < 1e-7).all():
            prin1('Correct phase increment has been computed')
        else:
            raise ValueError('Invalid phase increment for Gait Generator - test 2')
        
        print('Successfully tested gait generator')


    def test_swing_trajectory_generator(self):
        raise NotImplementedError


# Level 2 printing : low importance
def prin2(*args):
    if verbose >= 2:
        print(*args)

# Level 1 printing : Medium Importance
def prin1(*args):
    if verbose >= 1:
        print(*args)

#----------------------------- Main -----------------------------
def main(controller_name, controller):
    print('---- Starting %s class unit testing' % controller_name)

    test_class = modelBaseControllerTest(controller)

    test_class.test_gait_generator()



if __name__ == '__main__':

    controller_name = 'samplingController'    # 'samplingController', ...

    if controller_name == 'samplingController':
        main(controller_name=controller_name, controller=samplingController)

