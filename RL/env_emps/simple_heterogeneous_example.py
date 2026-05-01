"""
简单的异质IDM参数使用示例

演示如何快速启用和使用异质IDM参数功能
"""

import os
import sys

# 允许直接运行本文件：将项目根目录加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv


def example_1_random_heterogeneous():
    """示例1: 随机生成的异质IDM参数"""
    print("=== 示例1: 随机异质IDM参数 ===")
    
    env = SingleLaneFollowingEnv(
        seed=42,
        num_vehicles=4,
        cav_indices=[1],  # 第1辆车为CAV
        dt=0.1,
        heterogeneous_idm=True  # 启用异质IDM参数
    )
    
    obs, info = env.reset()
    
    print("生成的车辆IDM参数：")
    for vehicle in env.sim.vehicles:
        print(f"  {vehicle.vehicle_id} ({'CAV' if vehicle.is_cav else 'HV'}): {vehicle.idm_params}")
    
    print("\n运行10步仿真...")
    for step in range(10):
        action = [0.0]  # CAV保持稳定
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 5 == 0:
            print(f"步骤 {step}: {env.render().split(chr(10))[0]}")  # 只显示时间行
        if terminated:
            break


def example_2_driver_types():
    """示例2: 预定义驾驶员类型"""
    print("\n=== 示例2: 预定义驾驶员类型 ===")
    
    env = SingleLaneFollowingEnv(
        seed=42,
        num_vehicles=4,
        cav_indices=[2],  # 第2辆车为CAV
        dt=0.1,
        heterogeneous_idm=True,
        driver_types=["aggressive", "normal", "conservative", "cautious"]
    )
    
    obs, info = env.reset()
    
    print("驾驶员类型和IDM参数：")
    driver_types = ["aggressive", "normal", "conservative", "cautious"]
    for i, vehicle in enumerate(env.sim.vehicles):
        driver_type = driver_types[i] if i < len(driver_types) else "normal"
        print(f"  {vehicle.vehicle_id} ({driver_type}): 期望速度={vehicle.idm_params.v0:.1f}, 跟车时距={vehicle.idm_params.T:.1f}")
    
    print("\n运行10步仿真...")
    for step in range(10):
        action = [0.0]  # CAV保持稳定
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 5 == 0:
            print(f"步骤 {step}: {env.render().split(chr(10))[0]}")
        if terminated:
            break


def example_3_custom_variation():
    """示例3: 自定义参数变化范围"""
    print("\n=== 示例3: 自定义参数变化 ===")
    
    # 自定义IDM参数变化配置
    custom_config = {
        "desired_speed": {"std": 4.0, "min": 8.0, "max": 25.0},      # 期望速度变化较大
        "desired_time_headway": {"std": 0.6, "min": 0.5, "max": 3.0} # 跟车时距变化较大
    }
    
    env = SingleLaneFollowingEnv(
        seed=42,
        num_vehicles=4,
        cav_indices=[1],
        dt=0.1,
        heterogeneous_idm=True,
        idm_variation_config=custom_config
    )
    
    obs, info = env.reset()
    
    print("自定义变化范围生成的IDM参数：")
    for vehicle in env.sim.vehicles:
        params = vehicle.idm_params
        print(f"  {vehicle.vehicle_id}: 期望速度={params.v0:.1f} m/s, 跟车时距={params.T:.2f} s")
    
    print("\n运行10步仿真...")
    for step in range(10):
        action = [0.0]
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 5 == 0:
            print(f"步骤 {step}: {env.render().split(chr(10))[0]}")
        if terminated:
            break


def example_4_manual_idm():
    """示例4: 手动指定IDM参数"""
    print("\n=== 示例4: 手动指定IDM参数 ===")
    
    # 手动定义车辆和它们的IDM参数
    manual_vehicles = [
        {
            "vehicle_id": "Leader",
            "is_cav": False,
            "x_front": 100.0,
            "speed": 15.0,
            "idm_params": {
                "desired_speed": 20.0,        # 激进领车
                "minimum_spacing": 1.5,
                "desired_time_headway": 1.0,
                "max_acceleration": 1.8
            }
        },
        {
            "vehicle_id": "CAV1",
            "is_cav": True,
            "x_front": 75.0,
            "speed": 12.0
            # CAV没有IDM参数，使用外部控制
        },
        {
            "vehicle_id": "Follower",
            "is_cav": False,
            "x_front": 50.0,
            "speed": 10.0,
            "idm_params": {
                "desired_speed": 12.0,        # 保守跟车
                "minimum_spacing": 3.0,
                "desired_time_headway": 2.5,
                "max_acceleration": 0.8
            }
        }
    ]
    
    env = SingleLaneFollowingEnv(
        seed=42,
        num_vehicles=3,
        cav_indices=[1],
        dt=0.1
    )
    
    obs, info = env.reset(options={"manual_vehicles": manual_vehicles})
    
    print("手动指定的车辆和IDM参数：")
    for vehicle in env.sim.vehicles:
        if vehicle.idm_params:
            print(f"  {vehicle.vehicle_id}: 期望速度={vehicle.idm_params.v0:.1f}, 最小间距={vehicle.idm_params.s0:.1f}")
        else:
            print(f"  {vehicle.vehicle_id}: CAV (无IDM参数)")
    
    print("\n运行10步仿真...")
    for step in range(10):
        action = [0.0]  # CAV保持稳定
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 5 == 0:
            print(f"步骤 {step}: {env.render().split(chr(10))[0]}")
        if terminated:
            break


def main():
    """运行所有示例"""
    print("🚗 异质IDM参数使用示例\n")
    
    try:
        example_1_random_heterogeneous()
        example_2_driver_types()
        example_3_custom_variation()
        example_4_manual_idm()
        
        print("\n🎉 所有示例运行完成!")
        print("\n💡 使用方法总结:")
        print("1. 启用异质IDM: heterogeneous_idm=True")
        print("2. 驾驶员类型: driver_types=['aggressive', 'normal', ...]")
        print("3. 自定义变化: idm_variation_config={...}")
        print("4. 手动指定: manual_vehicles中添加idm_params")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()