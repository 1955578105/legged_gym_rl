from legged_gym.envs.base.legged_robot import LeggedRobot
import torch

class go2_walk(LeggedRobot):
   def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # 本次触地和上次触地 都算
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        #  在触地时触发奖励： 滞空时间大于 0.5 正奖励  小于 0.5 负
        rew_airTime = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        stuck_air_time = ((self.feet_air_time-0.8).clip(min=0.))*0.3 +self.last_stuck_air_time*0.7
        self.last_stuck_air_time = stuck_air_time
        airtime = torch.sum(stuck_air_time*~contact_filt,dim=1)
        self.feet_air_time *= ~contact_filt
        return rew_airTime-airtime