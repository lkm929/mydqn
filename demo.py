import os
import logging
import yaml
import argparse
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

# from sample_agent import MyAgent
from sample_agent_acorm import MyAgent
from pycmo.configs.config import get_config
from pycmo.env.cmo_env import CMOEnv
from pycmo.lib.protocol import SteamClientProps
from pycmo.lib.run_loop import run_loop_steam
from util.replay_buffer import ReplayBuffer

def load_config(config_path: str) -> Dict[str, Any]:
    """
    從 YAML 文件讀取配置。
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    使用命令行參數更新配置。
    """
    if args.scenario_name:
        config['scenario_name'] = args.scenario_name
    if args.player_side:
        config['player_side'] = args.player_side
    if args.scenario_script_folder_name:
        config['scenario_script_folder_name'] = args.scenario_script_folder_name
    if args.ac_name:
        config['agent']['ac_name'] = args.ac_name
    if args.target_name:
        config['agent']['target_name'] = args.target_name
    if args.debug_mode:
        config['debug_mode'] = args.debug_mode
    return config
def get_config1():
    parser = argparse.ArgumentParser(description="Run CMO agent with configurable parameters.")

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file')
    parser.add_argument('--scenario-name', type=str, help='Name of the scenario')
    parser.add_argument('--player-side', type=str, help='Player side (e.g., Taiwan)')
    parser.add_argument('--scenario-script-folder-name', type=str, help='Folder name containing agent Lua script')
    parser.add_argument('--ac-name', type=str, help='Name of the agent-controlled unit')
    parser.add_argument('--target-name', type=str, help='Name of the target unit')
    parser.add_argument('--debug-mode', type=bool, help='Debug mode')


    parser.add_argument('--N', type=int, default=1)  # 單位數量，動態確定
    parser.add_argument('--obs_dim', type=int, default=6) # 單位觀測維度
    parser.add_argument('--state_dim', type=int, default=6) # 單位狀態維度(全域)
    parser.add_argument('--action_dim', type=int, default=8) # 單位動作維度
#     parser.add_argument('--rnn_hidden_dim', type=int, default=64) # RNN隱藏層維度
#     parser.add_argument('--lr', type=float, default=0.001) # 學習率
#     parser.add_argument('--gamma', type=float, default=0.99) # 折扣因子
#     parser.add_argument('--epsilon', type=float, default=0.1) # epsilon
#     parser.add_argument('--buffer_size', type=int, default=10000) # 經驗回放緩存大小
#     parser.add_argument('--batch_size', type=int, default=64) # 批次大小
#     parser.add_argument('--max_train_steps', type=int, default=100000) # 最大訓練步數
    
    parser.add_argument("--max_train_steps", type=int, default=5000000, help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=10000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")

    parser.add_argument("--algorithm", type=str, default="ACORM", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=80000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.02, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=2, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--use_hard_update", type=bool, default=False, help="Whether to use hard update")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Whether to use learning rate decay")
    parser.add_argument("--lr_decay_steps", type=int, default=500, help="every steps decay steps")
    parser.add_argument("--lr_decay_rate", type=float, default=0.98, help="learn decay rate")
    parser.add_argument("--target_update_freq", type=int, default=100, help="Update frequency of the target network")
    parser.add_argument("--tau", type=float, default=0.005, help="If use soft update")
    parser.add_argument('--device', type=str, default='cuda:0')


    # RECL
    parser.add_argument("--agent_embedding_dim", type=int, default=128, help="The dimension of the agent embedding")
    parser.add_argument("--role_embedding_dim", type=int, default=64, help="The dimension of the role embedding")
    parser.add_argument("--use_ln", type=bool, default=False, help="Whether to use layer normalization")
    parser.add_argument("--cluster_num", type=int, default=int(3), help="the cluster number of knn")
    parser.add_argument("--recl_lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--agent_embedding_lr", type=float, default=1e-3, help="agent_embedding Learning rate")
    parser.add_argument("--train_recl_freq", type=int, default=200, help="Train frequency of the RECL network")
    parser.add_argument("--role_tau", type=float, default=0.005, help="If use soft update")
    parser.add_argument("--multi_steps", type=int, default=1, help="Train frequency of the RECL network")
    parser.add_argument("--role_mix_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the QMIX network")
    # attention
    parser.add_argument("--att_dim", type=int, default=128, help="The dimension of the attention net")
    parser.add_argument("--att_out_dim", type=int, default=64, help="The dimension of the attention net")
    parser.add_argument("--n_heads", type=int, default=4, help="multi-head attention")
    parser.add_argument("--soft_temperature", type=float, default=1.0, help="multi-head attention")
    parser.add_argument("--state_embed_dim", type=int, default=64, help="The dimension of the gru state net")

    # replay buffer
    parser.add_argument("--episode_limit", type=int, default=1000, help="The episode_limit of the replay buffer")

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    return args

def main():
    # 設置命令行參數解析
    args = get_config1()
    # 讀取基礎配置（從 pycmo 自帶的 config）
    base_config = get_config()
    
    config_path = os.path.join(r"C:\Users\iris\Desktop\code\pycmopen\scripts\MyDQN", args.config)
    print(config_path)
    # 讀取 YAML 配置
    yaml_config = load_config(config_path)

    # 使用命令行參數更新配置
    yaml_config = update_config_with_args(yaml_config, args)

    # 提取參數
    scenario_name = yaml_config['scenario_name']
    player_side = yaml_config['player_side']
    # player_side2 = yaml_config['player_side2']
    scenario_script_folder_name = yaml_config['scenario_script_folder_name']
    ac_name1 = yaml_config['agent']['ac_name1']
    ac_name2 = yaml_config['agent']['ac_name2']
    ac_name3 = yaml_config['agent']['ac_name3']
    target_name1 = yaml_config['agent']['target_name1']
    target_name2 = yaml_config['agent']['target_name2']
    target_name3 = yaml_config['agent']['target_name3']
    debug_mode = yaml_config['debug_mode']

    ac_name = [ac_name1,ac_name2,ac_name3]
    target_name = [target_name1,target_name2,target_name3]
    # player_side = [player_side1,player_side2]
    # 設置文件路徑
    command_version = base_config["command_mo_version"]
    observation_path = os.path.join(base_config['steam_observation_folder_path'], f'{scenario_name}.inst')
    action_path = os.path.join(base_config["scripts_path"], scenario_script_folder_name, "agent_action.lua")
    scen_ended_path = os.path.join(base_config['steam_observation_folder_path'], f'{scenario_name}_scen_has_ended.inst')

    # 初始化 Steam 客戶端屬性
    steam_client_props = SteamClientProps(
        scenario_name=scenario_name,
        agent_action_filename=action_path,
        command_version=command_version
    )
    print("ok1")

    # 初始化環境
    env = CMOEnv(
        player_side=player_side,
        steam_client_props=steam_client_props,
        observation_path=observation_path,
        action_path=action_path,
        scen_ended_path=scen_ended_path,
    )
    print("ok2")

    # 初始化 Agent
    replay_buffer = ReplayBuffer(args, args.buffer_size)
    agent = MyAgent(player_side=player_side, ac_name=ac_name, target_name=target_name, args=args, replay_buffer=replay_buffer)
    print("ok3")
    # 執行主循環
    run_loop_steam(env=env, agent=agent, max_steps=None)

if __name__ == "__main__":
    main()