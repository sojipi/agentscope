#!/usr/bin/env python3
"""
测试提示词攻击功能的测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import PlayerAgent
import asyncio
from agentscope.message import Msg


async def test_prompt_attack_system():
    """测试提示词攻击系统的基本功能"""
    print("🧪 测试提示词攻击系统...")
    
    # 创建测试代理
    agent = PlayerAgent("TestPlayer1")
    
    # 初始化攻击系统
    print("1. 初始化攻击系统...")
    success = agent.initialize_attack_system()
    print(f"   攻击系统初始化: {'✅ 成功' if success else '❌ 失败'}")
    
    if not success:
        print("   攻击系统初始化失败，跳过后续测试")
        return
    
    # 测试目标分析
    print("\n2. 测试目标分析功能...")
    agent.suspicions = {"Player2": 0.8, "Player3": 0.3}
    agent.speech_patterns = {
        "Player2": ["I'm not sure about this", "maybe we should reconsider"],
        "Player3": ["I'm certain about this", "obviously Player2 is the wolf"]
    }
    
    susceptibility = agent.analyze_target_susceptibility("Player2")
    print(f"   Player2 易感性分析: {susceptibility:.2f}")
    
    susceptibility = agent.analyze_target_susceptibility("Player3")
    print(f"   Player3 易感性分析: {susceptibility:.2f}")
    
    # 测试攻击内容生成
    print("\n3. 测试攻击内容生成...")
    
    test_scenarios = [
        ("role_fakeout", "I think Player2 might be suspicious"),
        ("confusion_injection", "We need to vote someone today"),
        ("vote_manipulation", "I'm going to vote for Player2"),
        ("social_pressure", "Player2 hasn't spoken much"),
        ("false_logic", "If Player2 is not a wolf, then logic suggests otherwise")
    ]
    
    for strategy, content in test_scenarios:
        if strategy in agent.attack_strategies:
            attack_content = agent._generate_attack_content(strategy, content, "Player2")
            print(f"   {strategy}: {attack_content[:80]}...")
    
    # 测试完整攻击应用
    print("\n4. 测试完整攻击应用...")
    
    original_content = "I think we should vote Player2 today. They seem suspicious."
    enhanced_content = agent.apply_prompt_attack(original_content, "Player2")
    
    print(f"   原始内容: {original_content}")
    print(f"   增强内容: {enhanced_content}")
    print(f"   内容变化: {'✅ 已增强' if enhanced_content != original_content else '❌ 无变化'}")
    
    # 测试策略建议
    print("\n5. 测试策略建议...")
    advice = agent.get_attack_strategy_advice()
    print(f"   策略建议: {advice[:100]}...")
    
    # 测试效果追踪
    print("\n6. 测试效果追踪...")
    agent.update_attack_effectiveness("confusion_injection", "Player2", True)
    agent.update_attack_effectiveness("confusion_injection", "Player3", False)
    
    print(f"   混淆注入成功率: {agent.attack_success_rates['confusion_injection']:.2f}")
    print(f"   攻击历史记录数: {len(agent.attack_history)}")
    
    print("\n✅ 提示词攻击系统测试完成!")


async def test_role_specific_attacks():
    """测试不同角色的特定攻击策略"""
    print("\n🎭 测试角色特定攻击策略...")
    
    roles = ["werewolf", "seer", "witch", "hunter", "villager"]
    
    for role in roles:
        print(f"\n--- 测试 {role.upper()} 角色 ---")
        
        agent = PlayerAgent(f"Test{role.title()}")
        agent.role = role
        agent.round_num = 2
        agent.phase = "day"
        agent.my_position = 5
        
        # 初始化攻击系统
        agent.initialize_attack_system()
        
        # 测试角色特定策略选择
        test_content = "I think Player2 might be suspicious today"
        target = "Player2"
        
        strategy = agent._select_attack_strategy(test_content, target, 2.1, "middle")
        print(f"   推荐策略: {strategy}")
        
        # 生成攻击内容
        if strategy:
            attack_content = agent._generate_attack_content(strategy, test_content, target)
            print(f"   攻击内容: {attack_content[:80]}...")
        
        # 获取角色特定建议
        advice = agent.get_attack_strategy_advice()
        print(f"   策略建议: {advice[:60]}...")


async def test_integration_with_reply():
    """测试与reply方法的集成"""
    print("\n🔗 测试与reply方法的集成...")
    
    agent = PlayerAgent("IntegrationTest")
    agent.role = "werewolf"
    agent.learning_enabled = True
    agent.round_num = 2
    agent.phase = "day"
    agent.my_position = 5
    
    # 初始化系统
    agent.initialize_attack_system()
    agent.initialize_learning_system()
    
    # 创建测试消息
    test_msg = Msg(
        name="TestGame",
        content="What do you think about Player2?",
        role="user"
    )
    
    print("   生成带攻击的回复...")
    try:
        response = await agent.reply(test_msg)
        if response:
            content = response.get_text_content() or ""
            print(f"   回复长度: {len(content)} 字符")
            print(f"   回复预览: {content[:100]}...")
            
            # 检查是否包含攻击元素
            attack_indicators = ["Player2", "suspicious", "vote", "vote", "think"]
            has_attack = any(indicator in content.lower() for indicator in attack_indicators)
            print(f"   包含攻击元素: {'✅ 是' if has_attack else '❌ 否'}")
        else:
            print("   ❌ 未生成回复")
    except Exception as e:
        print(f"   ❌ 集成测试失败: {e}")


async def main():
    """主测试函数"""
    print("🚀 开始提示词攻击功能测试")
    print("=" * 50)
    
    try:
        await test_prompt_attack_system()
        await test_role_specific_attacks()
        await test_integration_with_reply()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())