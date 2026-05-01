#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment similar to CyberAttackEnv but with simplified observation:
    - Headway distance error to the lead vehicle (front bumper to lead vehicle's rear bumper)
    - Speed error (lead speed - own speed)
    - Acceleration error (lead acceleration - own acceleration)
All other functionalities (cyber attack, CBF, filtering, etc.) remain unchanged.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any

# Ensure project root is in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env import CyberAttackEnv


class CyberAttackEnvErrorObs(CyberAttackEnv):
    """CyberAttackEnv with a simplified observation space.

    The observation for each CAV is a 3‑dimensional vector:
        [headway_distance_error, speed_error, acceleration_error]
    where the error is defined as (lead vehicle value - own value).
    """

    def _build_observation(self) -> np.ndarray:
        """Construct observation array with only the three error terms.
        
        The errors are defined as deviations from desired states (e.g. gap - desired_gap).
        We use a weighted sum of the error relative to the immediate leader and 
        the error relative to the lead CAV (weighted by p-value).

        Returns:
            np.ndarray: Shape (num_cav, 3) where each row corresponds to a CAV.
        """
        # Ensure simulation and vehicles are available
        if not self.sim or not self.sim.vehicles:
            return np.zeros((len(self.cav_ids), 3))

        # Get p-values for all CAVs
        cav_p_values = self.get_cav_p_values()
        
        # IDM parameters for desired gap calculation
        s0 = 2.0
        T = 1.
        avg_length = 5.0

        obs = []
        for cav_id in self.cav_ids:
            # Locate the CAV vehicle
            cav_idx = next((i for i, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id), None)
            if cav_idx is None:
                obs.append([0.0, 0.0, 0.0])
                continue
            cav_vehicle = self.sim.vehicles[cav_idx]
            
            # Calculate desired gap for this CAV
            # desired_gap = s0 + v * T
            desired_gap = s0 + max(0.0, cav_vehicle.speed) * T

            # --- 1. Error with immediate preceding vehicle ---
            lead_vehicle = None
            if cav_idx > 0:
                lead_vehicle = self.sim.vehicles[cav_idx - 1]
            
            headway_err_front = 0.0
            speed_err_front = 0.0
            accel_err_front = 0.0

            if lead_vehicle is not None:
                # Actual gap
                gap_front = (lead_vehicle.x_front - lead_vehicle.length) - cav_vehicle.x_front
                # Error = Actual - Desired
                headway_err_front = gap_front - desired_gap
                
                # Speed error (relative speed)
                speed_err_front = lead_vehicle.speed - cav_vehicle.speed
                # Acceleration error
                accel_err_front = lead_vehicle.acceleration - cav_vehicle.acceleration
            else:
                # No leader: treat as infinite gap (large positive error) or zero error?
                # Usually for control, no leader means we track desired speed. 
                # But here we are defining "following error". 
                # Let's set to 0 to avoid ghost forces, or a large value if we want to encourage speed up?
                # Given the user's prompt "bias towards 0", 0 is safest for "no error".
                headway_err_front = 0.0
                speed_err_front = 0.0
                accel_err_front = 0.0

            # --- 2. Error with lead CAV ---
            lead_cav = None
            lead_cav_idx = -1
            for i in range(cav_idx - 1, -1, -1):
                if self.sim.vehicles[i].is_cav:
                    lead_cav = self.sim.vehicles[i]
                    lead_cav_idx = i
                    break
            
            headway_err_cav = 0.0
            speed_err_cav = 0.0
            accel_err_cav = 0.0
            p = 0.0

            if lead_cav is not None:
                # Get p value
                if self.force_lead_cav_p_one:
                    p = 1.0
                else:
                    p = cav_p_values.get(lead_cav.vehicle_id, 1.0)
                
                # Get lead CAV network state
                lead_cav_state = self.get_cav_state(lead_cav.vehicle_id, "cav", cav_id)
                l_speed = lead_cav_state.get('speed', 0.0)
                l_pos = lead_cav_state.get('position', 0.0)
                l_accel = lead_cav_state.get('acceleration', 0.0)
                if l_speed is None: l_speed = 0.0
                if l_pos is None: l_pos = 0.0
                if l_accel is None: l_accel = 0.0

                # Calculate desired gap to lead CAV
                # If there are N vehicles between them (indices diff = N+1)
                # The total desired gap includes N+1 gaps and N vehicle lengths
                num_vehicles_between = cav_idx - lead_cav_idx - 1
                # Total desired distance (bumper to bumper)
                # = (N+1) * desired_gap + N * avg_length
                # Wait, gap is bumper to bumper.
                # Distance from LeadCAV Rear to Ego Front = 
                #   Gap_1 + Length_1 + Gap_2 + Length_2 ... + Gap_{N+1}
                # So N intermediate vehicles means N lengths + (N+1) gaps.
                total_desired_gap = (num_vehicles_between + 1) * desired_gap + num_vehicles_between * avg_length
                
                # Actual gap to lead CAV
                # l_pos is front of lead CAV. Rear is l_pos - lead_cav.length
                gap_cav = (l_pos - lead_cav.length) - cav_vehicle.x_front
                
                headway_err_cav = gap_cav - total_desired_gap
                speed_err_cav = l_speed - cav_vehicle.speed
                accel_err_cav = l_accel - cav_vehicle.acceleration

            # --- 3. Combine ---
            # obs = error_front + p * error_cav
            final_headway_err = headway_err_front + 0 * headway_err_cav
            final_speed_err = speed_err_front + p * speed_err_cav
            final_accel_err = accel_err_front + p * accel_err_cav

            obs.append([final_headway_err, final_speed_err, final_accel_err])

        return np.array(obs, dtype=np.float32)

    def _reward_multi(self,obs,action):
        return None
