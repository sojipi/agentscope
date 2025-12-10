import re
from agent import PlayerAgent

# Test the improved death player extraction
agent = PlayerAgent(name="Player1")

# Test case 1: Normal death message
test_msg1 = "主持人：昨天晚上Player2被淘汰了"
print("Test 1:", test_msg1)
agent._extract_game_info("主持人", test_msg1)
print(f"Dead players: {agent.dead_players}")
print(f"Alive players: {agent.alive_players}")
print()

# Test case 2: Multiple deaths
test_msg2 = "主持人：昨晚Player3和Player4死亡"
agent = PlayerAgent(name="Player1")  # Reset
agent.alive_players = ["Player1", "Player2", "Player3", "Player4", "Player5"]
agent._extract_game_info("主持人", test_msg2)
print("Test 2:", test_msg2)
print(f"Dead players: {agent.dead_players}")
print(f"Alive players after death: {agent.alive_players}")
print()

# Test case 3: New game should reset state
agent = PlayerAgent(name="Player1")
agent.teammates = ["Player2", "Player3"]
agent.dead_players = ["Player4", "Player5"]
agent.known_roles = {"Player2": "werewolf", "Player3": "werewolf"}
test_msg3 = "主持人：游戏开始！参与玩家：Player1, Player2, Player3, Player4, Player5, Player6, Player7, Player8, Player9"
agent._extract_game_info("主持人", test_msg3)
print("Test 3: New game reset")
print(f"Teammates after reset: {agent.teammates}")
print(f"Dead players after reset: {agent.dead_players}")
print(f"Known roles after reset: {agent.known_roles}")
print(f"Alive players: {agent.alive_players}")
print(f"Position: {agent.my_position}")
print()

# Test case 4: Werewolf teammate extraction
agent = PlayerAgent(name="Player1")
agent.role = "werewolf"
test_msg4 = "【仅狼人可见】你的队友是Player2和Player3"
agent._extract_game_info("主持人", test_msg4)
print("Test 4: Werewolf teammate extraction")
print(f"Teammates: {agent.teammates}")
print(f"Known roles: {agent.known_roles}")
print()

# Test case 5: Build context
agent = PlayerAgent(name="Player1")
agent.role = "werewolf"
agent.teammates = ["Player2", "Player3"]
agent.known_roles = {"Player2": "werewolf", "Player3": "werewolf", "Player4": "villager"}
agent.alive_players = ["Player1", "Player2", "Player3", "Player4", "Player5"]
agent.dead_players = ["Player6"]
agent.round_num = 1
agent.phase = "day"
agent.my_position = 1

context = agent._build_context()
print("Test 5: Context output:")
print(context)