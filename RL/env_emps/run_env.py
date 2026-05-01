import os
import sys
import time
import numpy as np

# 允许直接运行本文件：将项目根目录加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv


def main():
	# 多 CAV 示例：cav_indices=[1,3]
	env = SingleLaneFollowingEnv(seed=42, num_vehicles=8, cav_indices=[4], dt=0.1, jerk_coeff=0.05)
	# 提高初始车间距，降低初始碰撞概率
	opts = {"base_gap": 25.0, "v0": 6}
	reset_out = env.reset(options=opts)
	if isinstance(reset_out, tuple) and len(reset_out) == 2:
		obs, info = reset_out
	else:
		obs = reset_out 
	print("Initial:")
	print(f"CAV IDs: {env.cav_ids}")
	print(f"Number of CAVs: {len(env.cav_ids)}")
	try:
		print(env.render())
	except Exception:
		pass

	k = len(env.cav_ids)
	for t in range(1000):
		# 动作向量长度自适应 CAV 数量：第一个轻微加速，其余保持
		actions = np.zeros((k,), dtype=np.float32)
		if k > 0:
			actions[0] = 1
			# actions[1] = 1
		step_out = env.step(actions)
		# step_out = env.step({})
		if len(step_out) == 5:
			obs, reward, terminated, truncated, info = step_out
		else:
			obs, reward, terminated, info = step_out
		try:
			print(env.render())
		except Exception:
			pass
		if terminated:
			coll = info.get("collision") if isinstance(step_out, tuple) else None
			if coll:
				print(f"Collision detected between follower {coll[0]} and leader {coll[1]}.")
			else:
				print("Collision detected, terminating.")
			break
		# time.sleep(0.01)

	# 仿真结束后绘图
	try:
		env.plot_timeseries()
	except Exception as e:
		pass
		#print("Plot failed:", e)


if __name__ == "__main__":
	main()
