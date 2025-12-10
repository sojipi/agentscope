import re

# Test death player extraction patterns
print("Testing death player extraction patterns:")
print("=" * 50)

test_cases = [
    "主持人：昨天晚上Player2被淘汰了",
    "主持人：昨晚Player3和Player4死亡",
    "主持人：Player5出局",
    "主持人：淘汰了Player6",
    "Player7 was eliminated during the night",
    "Player8 died",
    "The players eliminated are Player9 and Player10",
    "游戏结束，胜利方是狼人：Player1, Player2, Player3"  # Should not match any deaths
]

# Patterns match "PlayerX was eliminated" or "淘汰了PlayerX" or "PlayerX死亡"
en_patterns = [r"(Player\d+)\s+(?:was\s+)?eliminated", r"(Player\d+)\s+(?:was\s+)?died"]
cn_patterns = [r"淘汰(了)?(Player\d+)", r"(Player\d+)(?:被)?淘汰", r"(Player\d+)死亡", r"死亡(了)?(Player\d+)"]

for test in test_cases:
    print(f"\nInput: {test}")
    dead_players = []
    for pattern in en_patterns + cn_patterns:
        matches = re.findall(pattern, test, re.I)
        for match in matches:
            if isinstance(match, tuple):
                # Extract the PlayerX from tuple matches
                dead_players.extend([m for m in match if m.startswith("Player")])
            else:
                dead_players.append(match)
    
    if dead_players:
        print(f"✓ Found dead players: {list(set(dead_players))}")
    else:
        print(f"✗ No dead players found")

print("\n" + "=" * 50)
print("Testing new game detection:")
test_game_msgs = [
    "主持人：游戏开始！参与玩家：Player1, Player2, Player3",
    "Welcome! new game! The players are Player1, Player2, Player3, Player4, Player5",
    "新的一局游戏开始了",
    "参与玩家有：Player1到Player9"
]

for test in test_game_msgs:
    print(f"\nInput: {test}")
    if ("players are" in test.lower() and "new game" in test.lower()) or \
       ("游戏开始" in test or "新的一局" in test or "参与玩家" in test):
        players = re.findall(r"Player\d+", test)
        print(f"✓ New game detected, players: {players}")
    else:
        print(f"✗ Not a new game message")