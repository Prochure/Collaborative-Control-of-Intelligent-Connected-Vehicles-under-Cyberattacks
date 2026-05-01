"""
演示CAV IDM参数的设置和使用
"""

import os
import sys

# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv
from sim_env.models.idm import IDMParameters


def demo_cav_with_manual_idm():
    """演示手动设置CAV的IDM参数"""
    print("=== 演示1: 手动设置CAV的IDM参数 ===")
    
    # 手动定义车辆，包括CAV的IDM参数
    manual_vehicles = [
        {
            "vehicle_id": "Leader",
            "is_cav": False,
            "x_front": 50.0,
            "speed": 10.0,
            "idm_params": {
                "desired_speed": 15.0,
                "minimum_spacing": 2.0,
                "desired_time_headway": 1.5
            }
        },
        {
            "vehicle_id": "CAV_Conservative",
            "is_cav": True,
            "x_front": 25.0,
            "speed": 8.0,
            "idm_params": {  # CAV的保守IDM参数
                "desired_speed": 12.0,
                "minimum_spacing": 3.0,
                "desired_time_headway": 2.0,
                "max_acceleration": 0.8,
                "comfortable_deceleration": 1.2
            }
        }
    ]
    
    env = SingleLaneFollowingEnv(num_vehicles=2, cav_indices=[1], dt=0.1)
    obs, info = env.reset(options={"manual_vehicles": manual_vehicles})
    
    print("车辆IDM参数:")
    for vehicle in env.sim.vehicles:
        print(f"  {vehicle.vehicle_id} ({'CAV' if vehicle.is_cav else 'HV'}): {vehicle.idm_params}")
    
    print(f"\n--- 场景1: CAV使用外部控制 ---")
    for step in range(5):
        # CAV使用外部控制
        action = [0.5]  # CAV加速
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步骤 {step}: CAV外部控制, 加速度=0.5 m/s²")
        if terminated:
            break
    
    print(f"\n--- 场景2: CAV使用IDM自动驾驶 ---")
    for step in range(5):
        # 不提供CAV动作，让其使用IDM参数
        action = {}  # 空动作，CAV将使用IDM
        obs, reward, terminated, truncated, info = env.step(action)
        cav_accel = env.sim.vehicles[1].acceleration  # CAV是第2辆车
        print(f"步骤 {step}: CAV IDM自动驾驶, 加速度={cav_accel:.3f} m/s²")
        if terminated:
            break


def demo_cav_heterogeneous_idm():
    """演示异质IDM环境中的CAV参数"""
    print("\n=== 演示2: 异质IDM环境中的CAV ===")
    
    env = SingleLaneFollowingEnv(
        num_vehicles=4,
        cav_indices=[1, 2],  # 第1、2辆车为CAV
        heterogeneous_idm=True,
        driver_types=["aggressive", "normal", "conservative", "cautious"],
        dt=0.1
    )
    
    obs, info = env.reset()
    
    print("所有车辆的IDM参数:")
    driver_types = ["aggressive", "normal", "conservative", "cautious"]
    for i, vehicle in enumerate(env.sim.vehicles):
        driver_type = driver_types[i] if i < len(driver_types) else "normal"
        cav_status = "CAV" if vehicle.is_cav else "HV"
        print(f"  {vehicle.vehicle_id} ({cav_status}, {driver_type}): 期望速度={vehicle.idm_params.v0:.1f}, 跟车时距={vehicle.idm_params.T:.1f}")
    
    print(f"\n--- 混合控制演示 ---")
    for step in range(8):
        if step < 4:
            # 前4步：只控制第一辆CAV，第二辆CAV使用IDM
            action = [0.8, None]  # 第一辆CAV加速，第二辆使用IDM
            cav1_control = "外部控制(0.8)"
            cav2_control = "IDM自动"
        else:
            # 后4步：两辆CAV都使用IDM
            action = {}
            cav1_control = "IDM自动"
            cav2_control = "IDM自动"
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取CAV加速度
        cav1_accel = env.sim.vehicles[1].acceleration
        cav2_accel = env.sim.vehicles[2].acceleration
        
        print(f"步骤 {step}: CAV1({cav1_control})={cav1_accel:.3f}, CAV2({cav2_control})={cav2_accel:.3f}")
        
        if terminated:
            break


def demo_cav_idm_comparison():
    """对比不同IDM参数的CAV行为"""
    print("\n=== 演示3: 不同IDM参数的CAV行为对比 ===")
    
    # 创建两个不同IDM参数的CAV
    manual_vehicles = [
        {"vehicle_id": "Leader", "is_cav": False, "x_front": 60.0, "speed": 10.0},
        {
            "vehicle_id": "CAV_Aggressive",
            "is_cav": True,
            "x_front": 35.0,
            "speed": 8.0,
            "idm_params": {
                "desired_speed": 18.0,  # 高期望速度
                "minimum_spacing": 1.5,  # 小最小间距
                "desired_time_headway": 1.0,  # 短跟车时距
                "max_acceleration": 1.5
            }
        },
        {
            "vehicle_id": "CAV_Conservative",
            "is_cav": True,
            "x_front": 10.0,
            "speed": 8.0,
            "idm_params": {
                "desired_speed": 12.0,  # 低期望速度
                "minimum_spacing": 3.0,  # 大最小间距
                "desired_time_headway": 2.5,  # 长跟车时距
                "max_acceleration": 0.8
            }
        }
    ]
    
    env = SingleLaneFollowingEnv(num_vehicles=3, cav_indices=[1, 2], dt=0.1)
    obs, info = env.reset(options={"manual_vehicles": manual_vehicles})
    
    print("CAV IDM参数对比:")
    for vehicle in env.sim.vehicles:
        if vehicle.is_cav:
            params = vehicle.idm_params
            print(f"  {vehicle.vehicle_id}: v0={params.v0:.1f}, s0={params.s0:.1f}, T={params.T:.1f}")
    
    print(f"\n--- 让两辆CAV都使用IDM自动驾驶 ---")
    for step in range(6):
        # 两辆CAV都使用IDM
        action = {}
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取CAV状态
        aggressive_cav = env.sim.vehicles[1]
        conservative_cav = env.sim.vehicles[2]
        
        print(f"步骤 {step}: 激进CAV(v={aggressive_cav.speed:.1f}, a={aggressive_cav.acceleration:.3f}), "
              f"保守CAV(v={conservative_cav.speed:.1f}, a={conservative_cav.acceleration:.3f})")
        
        if terminated:
            break


def main():
    """主函数"""
    print("🚗 CAV IDM参数设置和使用演示\n")
    
    try:
        # 演示1: 手动设置CAV IDM参数
        demo_cav_with_manual_idm()
        
        # 演示2: 异质IDM环境中的CAV
        demo_cav_heterogeneous_idm()
        
        # 演示3: 不同IDM参数的CAV行为对比
        demo_cav_idm_comparison()
        
        print(f"\n🎉 演示完成!")
        print(f"\n💡 总结:")
        print(f"✓ CAV可以设置个性化的IDM参数")
        print(f"✓ 当没有外部控制时，CAV会使用其IDM参数自动驾驶")
        print(f"✓ 可以实现外部控制和IDM自动驾驶的混合模式")
        print(f"✓ 不同IDM参数会导致不同的驾驶行为")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()