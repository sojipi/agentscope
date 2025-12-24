# -*- coding: utf-8 -*-
"""Random name generator for werewolf game players."""

import random
from typing import List


class RandomNameGenerator:
    """Generate random names for players."""
    
    def __init__(self):
        # 中文姓氏
        self.chinese_surnames = [
            "李", "王", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
            "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗",
            "梁", "宋", "郑", "谢", "韩", "唐", "冯", "于", "董", "萧",
            "程", "曹", "袁", "邓", "许", "傅", "沈", "曾", "彭", "吕"
        ]
        
        # 中文名字
        self.chinese_names = [
            "伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋",
            "勇", "艳", "杰", "娟", "涛", "明", "超", "秀英", "霞", "平",
            "刚", "桂英", "建华", "志强", "秀兰", "国强", "德华", "文华", "志华", "小华",
            "雨", "雪", "春", "夏", "秋", "冬", "山", "水", "花", "月"
        ]
        
        # 英文名字
        self.english_names = [
            "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Helen",
            "Isaac", "Julia", "Kevin", "Laura", "Michael", "Nancy", "Oscar", "Peggy",
            "Quinn", "Rachel", "Sam", "Tina", "Ulysses", "Victoria", "William", "Xena",
            "Yuri", "Zara", "Alex", "Ben", "Chris", "David", "Emma", "Frank", "Grace",
            "Henry", "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Peter", "Quincy"
        ]
        
        # 英文姓氏
        self.english_surnames = [
            "Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
            "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia",
            "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall",
            "Allen", "Young", "Hernandez", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams"
        ]
    
    def generate_random_name(self) -> str:
        """Generate a random name."""
        name_type = random.choice(["chinese", "english", "mixed"])
        
        if name_type == "chinese":
            # 生成中文名字
            surname = random.choice(self.chinese_surnames)
            given_name = random.choice(self.chinese_names)
            return surname + given_name
            
        elif name_type == "english":
            # 生成英文名字
            first_name = random.choice(self.english_names)
            last_name = random.choice(self.english_surnames)
            return f"{first_name}{last_name}"
            
        else:  # mixed
            # 生成混合名字（中英混合）
            chinese_surname = random.choice(self.chinese_surnames)
            english_given_name = random.choice(self.english_names)
            return chinese_surname + english_given_name
    
    def generate_multiple_names(self, count: int, exclude_names: List[str] = None) -> List[str]:
        """Generate multiple random names, ensuring uniqueness."""
        if exclude_names is None:
            exclude_names = []
        
        names = set()
        max_attempts = count * 10  # 防止无限循环
        attempts = 0
        
        while len(names) < count and attempts < max_attempts:
            name = self.generate_random_name()
            if name not in exclude_names and name not in names:
                names.add(name)
            attempts += 1
        
        # 如果尝试次数过多还没有生成足够的名字，添加序号后缀
        while len(names) < count:
            name = f"Player{len(names) + 1}"
            if name not in exclude_names:
                names.add(name)
        
        return list(names)


# 全局实例
name_generator = RandomNameGenerator()


def generate_player_names(count: int = 9) -> List[str]:
    """Generate random player names.
    
    Args:
        count: Number of names to generate
        
    Returns:
        List of random player names
    """
    return name_generator.generate_multiple_names(count)