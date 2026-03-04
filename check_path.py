import os
import re

# 任务列表（从run_yamls.sh的14-48行）
tasks = [
    "maniskill_libero openpi maniskill_ppo_openpi",
    "maniskill_libero openpi maniskill_ppo_openpi_pi05",
    "maniskill_libero openvla maniskill_ppo_openvla",
    "maniskill_libero openvla-oft maniskill_ppo_openvlaoft",
    "maniskill_libero mlp maniskill_ppo_mlp",
    "maniskill_libero openpi libero_goal_ppo_openpi",
    "maniskill_libero openpi libero_goal_ppo_openpi_pi05",
    "maniskill_libero gr00t libero_10_ppo_gr00t",
    "behavior openpi behavior_ppo_openpi",
    "calvin openpi calvin_abc_d_ppo_openpi",
    "calvin openpi calvin_abcd_d_ppo_openpi_pi05",
    "robotwin openvla-oft robotwin_place_empty_cup_ppo_openvlaoft",
    "isaaclab gr00t isaaclab_franka_stack_cube_ppo_gr00t",
    "frankasim mlp frankasim_ppo_mlp",
    "maniskill_libero openpi gsenv_ppo_openpi_pi05",
    "maniskill_libero openpi maniskill_async_ppo_openpi",
    "maniskill_libero openpi maniskill_async_ppo_openpi_pi05",
    "maniskill_libero openvla maniskill_async_ppo_openvla",
    "maniskill_libero openvla-oft maniskill_async_ppo_openvlaoft",
    "maniskill_libero openpi libero_spatial_async_ppo_openpi",
    "maniskill_libero openpi libero_object_async_ppo_openpi_pi05",
    "maniskill_libero openpi realworld_45_ppo_openpi",
    "maniskill_libero openpi realworld_50_ppo_openpi_pi05",
    "maniskill_libero openvla maniskill_grpo_openvla",
    "maniskill_libero openvla-oft maniskill_grpo_openvlaoft",
    "maniskill_libero openpi libero_10_grpo_openpi",
    "maniskill_libero openpi libero_spatial_grpo_openpi_pi05",
    "maniskill_libero openvla-oft libero_10_grpo_openvlaoft",
    "maniskill_libero mlp libero_spatial_0_grpo_mlp",
    "robotwin openvla-oft robotwin_beat_block_hammer_grpo_openvlaoft",
    "wan openvla-oft wan_libero_goal_grpo_openvlaoft",
    "maniskill_libero mlp maniskill_sac_mlp",
    "frankasim mlp frankasim_sac_cnn_async",
]

config_dir = "examples/embodiment/config"
missing_paths = []
missing_configs = []

def find_model_paths_in_file(yaml_file):
    """从yaml文件中提取所有model_path"""
    paths = []
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式查找所有model_path
        pattern = r'model_path:\s*(?:["\']([^"\']+)["\']|([^\s\n#]+))'
        matches = re.finditer(pattern, content)
        
        for match in matches:
            path = match.group(1) if match.group(1) else match.group(2)
            if path and not path.startswith('${') and path != 'null':
                path = path.strip().rstrip(',').strip()
                if path and path != 'null':
                    paths.append(path)
    except Exception as e:
        pass
    
    return paths

# 检查所有任务
for task in tasks:
    parts = task.split()
    if len(parts) >= 3:
        yaml_name = parts[2]
        yaml_file = os.path.join(config_dir, f"{yaml_name}.yaml")

        if os.path.exists(yaml_file):
            model_paths = find_model_paths_in_file(yaml_file)
            for path in model_paths:
                if path and path != 'null':
                    if not os.path.exists(path):
                        missing_paths.append((yaml_name, path))
        else:
            missing_configs.append(yaml_name)

# 输出结果
print("=" * 80)
print("run_yamls.sh (14-48行) 任务配置检查报告")
print("=" * 80)

if missing_configs:
    print(f"\n【不存在的YAML配置文件】({len(missing_configs)}个):")
    for i, config in enumerate(sorted(set(missing_configs)), 1):
        print(f"  {i:2d}. {config}.yaml")

if missing_paths:
    print(f"\n【不存在的model path】({len(missing_paths)}个):")
    # 去重并按yaml名称排序
    seen = {}
    for yaml_name, path in missing_paths:
        if yaml_name not in seen:
            seen[yaml_name] = []
        if path not in seen[yaml_name]:
            seen[yaml_name].append(path)

    for i, (yaml_name, paths) in enumerate(sorted(seen.items()), 1):
        print(f"\n  {i}. {yaml_name}:")
        for path in paths:
            print(f"     - {path}")

    print(f"\n总计: {len(seen)} 个配置文件，{len(missing_paths)} 个不存在的路径")
else:
    if not missing_configs:
        print("\n✓ 所有存在的配置文件中的model path都存在！")
    else:
        print("\n✓ 所有存在的配置文件中的model path都存在！")

print("\n" + "=" * 80)