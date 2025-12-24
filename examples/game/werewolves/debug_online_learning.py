#!/usr/bin/env python3
"""Debug online learning functionality."""
import traceback
from agent import PlayerAgent

def test_online_learning_debug():
    print("=== 在线学习功能调试 ===")
    
    try:
        # 初始化智能体
        print("1. 初始化PlayerAgent...")
        agent = PlayerAgent("TestPlayer")
        print("✓ PlayerAgent初始化成功")
        
        # 检查initialize_learning_system方法是否存在
        print("2. 检查initialize_learning_system方法...")
        if hasattr(agent, 'initialize_learning_system'):
            print("✓ initialize_learning_system方法存在")
        else:
            print("❌ initialize_learning_system方法不存在")
            return False
        
        # 调用initialize_learning_system
        print("3. 调用initialize_learning_system...")
        agent.initialize_learning_system()
        print("✓ initialize_learning_system调用成功")
        
        # 检查experience_weights是否被正确初始化
        print("4. 检查experience_weights...")
        if hasattr(agent, 'experience_weights') and agent.experience_weights:
            print(f"✓ experience_weights: {agent.experience_weights}")
        else:
            print("❌ experience_weights未正确初始化")
            return False
        
        # 测试update_strategy_weights方法
        print("5. 测试update_strategy_weights...")
        test_decisions = {
            "voting_patterns": 0.7,
            "speech_analysis": 0.6,
            "role_claim_evaluation": 0.8
        }
        agent.update_strategy_weights("win", test_decisions)
        print("✓ update_strategy_weights调用成功")
        
        # 测试get_adaptive_strategy_advice方法
        print("6. 测试get_adaptive_strategy_advice...")
        if hasattr(agent, 'get_adaptive_strategy_advice'):
            advice = agent.get_adaptive_strategy_advice()
            print(f"✓ 策略建议: {advice}")
        else:
            print("❌ get_adaptive_strategy_advice方法不存在")
            return False
        
        # 测试evaluate_decision_quality方法
        print("7. 测试evaluate_decision_quality...")
        if hasattr(agent, 'evaluate_decision_quality'):
            context = {"target": "Player1", "reasoning": "test"}
            quality = agent.evaluate_decision_quality("voting", context)
            print(f"✓ 决策质量评分: {quality}")
        else:
            print("❌ evaluate_decision_quality方法不存在")
            return False
        
        print("=== 所有测试通过 ===")
        return True
        
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        print("错误详情:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_online_learning_debug()
    if success:
        print("\n✅ 在线学习功能测试成功")
    else:
        print("\n❌ 在线学习功能测试失败")