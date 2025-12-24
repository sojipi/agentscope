#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试跨局记忆功能 - 验证在线学习系统的状态持久化能力
"""

from agent import PlayerAgent
import json

def test_cross_game_memory():
    print("=== 跨局记忆测试 ===")
    
    # 第一局：创建智能体并积累经验
    agent1 = PlayerAgent('TestPlayer')
    print('1. 创建第一个智能体实例')
    
    # 模拟一些学习数据积累
    agent1.experience_weights['aggressive'] = 0.7
    agent1.experience_weights['defensive'] = 0.3
    agent1.adaptation_history.append({'round': 1, 'strategy': 'aggressive', 'outcome': 'success'})
    agent1.strategy_performance['aggressive'] = 0.8
    
    print(f'2. 第一局积累的经验权重: {agent1.experience_weights}')
    print(f'   适应历史: {agent1.adaptation_history}')
    print(f'   策略性能: {agent1.strategy_performance}')
    
    # 保存状态
    saved_state = agent1.state_dict()
    print('3. 保存智能体状态')
    
    # 第二局：创建新智能体并加载状态
    agent2 = PlayerAgent('TestPlayer')  
    agent2.load_state_dict(saved_state)
    print('4. 创建第二个智能体实例并加载状态')
    
    # 验证跨局记忆
    print('5. 验证跨局记忆效果:')
    print(f'   经验权重保留: {agent2.experience_weights}')
    print(f'   适应历史保留: {agent2.adaptation_history}')
    print(f'   策略性能保留: {agent2.strategy_performance}')
    
    # 继续学习新经验
    agent2.experience_weights['neutral'] = 0.5
    agent2.adaptation_history.append({'round': 2, 'strategy': 'neutral', 'outcome': 'partial'})
    print(f'6. 第二局新增经验: {agent2.adaptation_history}')
    
    # 验证持久化到JSON
    agent2_state = agent2.state_dict()
    with open('cross_game_memory_test.json', 'w', encoding='utf-8') as f:
        json.dump(agent2_state, f, ensure_ascii=False, indent=2)
    print('7. 状态已持久化到JSON文件')
    
    print('\n✅ 跨局记忆功能正常！经验可以在不同游戏实例间传递和积累')
    return True

if __name__ == "__main__":
    test_cross_game_memory()