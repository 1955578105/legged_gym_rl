#  目标是 训练 go2 用 后两支脚行走
# 1. 先降低 速度 角速度  2 . 关闭推力
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch

class go2_task(LeggedRobot):
 
      # 角速度直接改为 世界下的表示
      def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.root_states[:, 12])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
   
     # 直接将 command 指令 变到世界坐标系下 
      def _reward_tracking_lin_vel(self):
        # 1. 从四元数中提取每个机器人的 Yaw 角
        # 在 Legged Gym 中，quat_rotate_inverse 等工具很常用，
        # 但对于 2D 旋转，直接计算更直观：
        quat = self.base_quat
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # 2. 计算 Yaw 的正余弦值
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # 3. 将指令从指令坐标系（Heading Frame）转换到世界坐标系（World Frame）
        # 指令通常是 [前进速度, 侧移速度]
        v_cmd_x = self.commands[:, 0]
        v_cmd_y = self.commands[:, 1]

        # 2D 旋转矩阵计算:
        # world_x = cmd_x * cos(yaw) - cmd_y * sin(yaw)
        # world_y = cmd_x * sin(yaw) + cmd_y * cos(yaw)
        cmd_world_x = v_cmd_x * cos_yaw - v_cmd_y * sin_yaw
        cmd_world_y = v_cmd_x * sin_yaw + v_cmd_y * cos_yaw

        # 合成世界坐标系下的指令张量
        commands_world = torch.stack([cmd_world_x, cmd_world_y], dim=1)

        # 4. 计算误差（世界坐标系下的指令 vs 世界坐标系下的实际速度）
        lin_vel_error = torch.sum(torch.square(commands_world - self.root_states[:, 7:9]), dim=1)

        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
      

      def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.root_states[:, 9])
    
      # 惩罚 x y 轴角速度 改为只惩罚x轴
      def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.square(self.base_ang_vel[:, 0])
      
      # 重力 投影   改为 惩罚 y 方向的投影（不左右偏）   奖励 x方向的投影（鼓励站起来）
      def _reward_orientation_y(self):
        # Penalize non flat base orientation
        return torch.square(self.projected_gravity[:, 1])
      
      def _reward_orientation_x(self):
        # 设定目标仰角，例如 0.785 弧度 (45度)
        target_pitch = 0.785 
        target_gravity_x = -torch.sin(torch.tensor(target_pitch, device=self.device))
        return torch.exp(-torch.square(self.projected_gravity[:, 0] - target_gravity_x)/0.1)
        

      # 前腿 如果触地 直接惩罚 
      def _reward_feet_air_time_front(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices[:2], 2] > 1.
        # 本次触地和上次触地 都算
        contact_filt = torch.logical_or(contact, self.last_contacts1) 
        self.last_contacts1 = contact
        # 返回 触地数量前两只腿触地数量和  
        rew_airTime = torch.sum(contact_filt,dim=1) 
        return rew_airTime
      
      # 后腿正常处理
      def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices[2:4], 2] > 1.
        # 本次触地和上次触地 都算
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        #  在触地时触发奖励： 滞空时间大于 0.5 正奖励  小于 0.5 负
        rew_airTime = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
      
      # 速度 大于 0.1时 两只脚都在地上的数量