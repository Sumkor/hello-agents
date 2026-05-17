import collections

# 示例语料库，与上方案例讲解中的语料库保持一致
corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens = len(tokens)

# --- 第一步：计算 P(datawhale) ---
count_datawhale = tokens.count('datawhale')
p_datawhale = count_datawhale / total_tokens
print(f"第一步: P(datawhale) = {count_datawhale}/{total_tokens} = {p_datawhale:.3f}")

# --- 第二步：计算 P(agent|datawhale) ---
# 先计算 bigrams 用于后续步骤
# 打包获的元组：
# ('datawhale', 'agent')
# ('agent', 'learns')
# ('learns', 'datawhale')
# ('datawhale', 'agent')
# ('agent', 'works')
bigrams = zip(tokens, tokens[1:])
# 计算每个元组出现的次数
# Counter({
#     ('datawhale', 'agent'): 2,
#     ('agent', 'learns'): 1,
#     ('learns', 'datawhale'): 1,
#     ('agent', 'works'): 1
# })
bigram_counts = collections.Counter(bigrams)
# count_datawhale_agent取得2
count_datawhale_agent = bigram_counts[('datawhale', 'agent')]
# count_datawhale 已在第一步计算
p_agent_given_datawhale = count_datawhale_agent / count_datawhale
print(f"第二步: P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = {p_agent_given_datawhale:.3f}")

# --- 第三步：计算 P(learns|agent) ---
count_agent_learns = bigram_counts[('agent', 'learns')]
count_agent = tokens.count('agent')
p_learns_given_agent = count_agent_learns / count_agent
print(f"第三步: P(learns|agent) = {count_agent_learns}/{count_agent} = {p_learns_given_agent:.3f}")

# --- 最后：将概率连乘 ---
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(f"最后: P('datawhale agent learns') ≈ {p_datawhale:.3f} * {p_agent_given_datawhale:.3f} * {p_learns_given_agent:.3f} = {p_sentence:.3f}")

# 句子概率的作用：核心是验证语言规律、辅助生成/纠错、支撑下游任务，本质是量化语言序列的“自然度”。
# 语义理解的能力：概率能间接反映语义的“合理性”“通顺度”（即语义的基本逻辑），但无法覆盖语义的全部维度（情感、深层意图、指代等）。

# N-gram模型：通过统计“前N-1个词”的条件概率预测下一个词，能捕捉局部语义关联（比如“吃”后接“饭”“面”的概率高），
# 但对长距离依赖（如“昨天我在书店看到一本关于人工智能的书”）处理差，语义理解的“全局性”不足。
# RNN/LSTM/Transformer：通过循环结构或自注意力，能捕捉长距离语义依赖（比如“人工智能”和“书”的关联），使句子概率的计算更贴合“完整语义”，
# 但本质上仍基于“序列概率”，而非显式的语义解析。
