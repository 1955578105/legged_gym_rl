from legged_gym.envs.base.legged_robot import LeggedRobot
import torch

class go2_task(LeggedRobot):
    def aver(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indice, 2] > 1.
        # 本次触地和上次触地 都算
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        #  在触地时触发奖励： 滞空时间大于 0.5 正奖励  小于 0.5 负
        self.sum_contacts += first_contact 
    
        self.feet_air_time *= ~contact_filt
        return rew_airTime