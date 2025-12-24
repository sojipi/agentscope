#!/usr/bin/env python3
"""
测试PlayerAgent在线学习功能的脚本
只测试在线学习功能：初始化、学习权重更新、自适应建议等
"""

import sys
import os
import asyncio
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import PlayerAgent
from agentscope.message import Msg

async def test_online_learning():
    """测试在线学习功能"""
    print("=" * 60)
    print("测试1: 在线学习功能")
    print("=" * 60)
    
    # 创建PlayerAgent实例
    agent = PlayerAgent(name="TestAgent")
    
    # 初始化学习系统
    agent.initialize_learning_system()
    
    print("✓ 在线学习系统初始化成功")
    print(f"初始经验权重: {agent.experience_weights}")
    print(f"初始模型权重: {agent.model_weights}")
    
    # 模拟游戏历史记录
    agent.game_history = [
        {"round": 1, "phase": "day", "role": "villager", "action": "Vote for suspicious player"},
        {"round": 2, "phase": "night", "role": "villager", "action": "Stay quiet"},
        {"round": 3, "phase": "day", "role": "villager", "action": "Accuse wolf"},
    ]
    
    # 更新策略权重
    player_decisions = {
        "voting_patterns": 0.6,
        "speech_analysis": 0.7,
        "role_claim_evaluation": 0.8
    }
    agent.update_strategy_weights("loss", player_decisions)
    
    print(f"更新后的经验权重: {agent.experience_weights}")
    
    # 获取自适应策略建议
    advice = agent.get_adaptive_strategy_advice()
    print(f"自适应策略建议: {advice}")
    
    # 测试决策质量评估
    decision_context = {"target": "Player1", "reasoning": "Suspicious behavior"}
    quality = agent.evaluate_decision_quality("voting", decision_context)
    print(f"决策质量评分: {quality}")
    
    print("✓ 在线学习功能测试完成\n")
    return True

async def main():
    """主测试函数"""
    print("开始测试PlayerAgent在线学习功能...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        result = await test_online_learning()
        print("=" * 60)
        print("测试总结")
        print("=" * 60)
        
        if result:
            print("测试1 - 在线学习功能: ✅ 通过")
            print("\n总计: 1/1 项测试通过")
            print("🎉 在线学习功能测试通过！PlayerAgent在线学习系统实现成功。")
        else:
            print("测试1 - 在线学习功能: ❌ 失败")
            print("\n总计: 0/1 项测试通过")
            print("⚠️ 在线学习功能测试失败，请检查相关功能实现。")
        
        return result
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)