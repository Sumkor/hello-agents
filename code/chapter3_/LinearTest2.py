import torch
import torch.nn as nn

def test_three_linear_are_different():
    torch.manual_seed(42)

    W_q = nn.Linear(4, 4)
    W_k = nn.Linear(4, 4)
    W_v = nn.Linear(4, 4)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    Q = W_q(x)
    K = W_k(x)
    V = W_v(x)

    # 1. 它们是三个不同对象
    assert W_q is not W_k
    assert W_q is not W_v
    assert W_k is not W_v

    # 2. 它们的参数不是同一块内存
    assert W_q.weight.data_ptr() != W_k.weight.data_ptr()
    assert W_q.weight.data_ptr() != W_v.weight.data_ptr()
    assert W_k.weight.data_ptr() != W_v.weight.data_ptr()

    # 3. 参数值通常也不同
    assert not torch.equal(W_q.weight, W_k.weight)
    assert not torch.equal(W_q.weight, W_v.weight)
    assert not torch.equal(W_k.weight, W_v.weight)

    # 4. 同一个输入经过三层，输出通常不同
    assert not torch.allclose(Q, K)
    assert not torch.allclose(Q, V)
    assert not torch.allclose(K, V)

    print("W_q.weight =\n", W_q.weight)
    print("W_k.weight =\n", W_k.weight)
    print("W_v.weight =\n", W_v.weight)
    print("Q =", Q)
    print("K =", K)
    print("V =", V)
    print("测试通过：三个 Linear 虽然写法一样，但参数彼此独立。")

    # 看参数到底长什么样
    print(W_q.state_dict().keys())
    print(W_q.weight.shape)
    print(W_q.bias.shape)

test_three_linear_are_different()