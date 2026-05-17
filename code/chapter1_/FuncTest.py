# 定义两个函数
def func_a(x):
    return f"调用 func_a, 参数: {x}"

def func_b(x):
    return f"调用 func_b, 参数: {x}"

# 存储到字典
tools = {
    "a": func_a,
    "b": func_b
}

# 动态调用
tool_name = "a"
kwargs = {"x": "hello"}

# 步骤 1: 从字典获取函数对象
# 步骤 2: 参数解包，kwargs = {"city": "北京"}，**kwargs 展开为 city="北京"
# 步骤 3: 调用函数，等价于 func_a(x="hello")
result = tools[tool_name](**kwargs)
print(result)
# 输出: 调用 func_a, 参数: hello
