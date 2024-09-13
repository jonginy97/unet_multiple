import torch
import torch.nn as nn
import torch.optim as optim

# 단순한 모델 정의
model = nn.Linear(1, 1)


criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 입력과 실제 출력
input = torch.tensor([[1.0]])
target = torch.tensor([[2.0]])

# Forward Pass
output = model(input)

print(f'Initial weights: {model.weight.data}')
print(f'Initial bias: {model.bias.data}')

loss = criterion(output, target)

print(f'Loss: {loss.item()}')

# Backward Pass
optimizer.zero_grad()  # 기존 기울기 초기화
loss.backward()        # 역전파를 통해 기울기 계산


# 미분 계산된 기울기 출력
print(f'Gradient of weight: {model.weight.grad}')
print(f'Gradient of bias: {model.bias.grad}')

optimizer.step()       # 옵티마이저를 통해 가중치 업데이트

# derived from the loss.backward() call
# dLoss/dw = -2 * 1 * -1 = 2
# dLoss/db = -2 * 1 * -1 = 2




# Updated weights and bias

print(f'Updated weights: {model.weight.data}')
print(f'Updated bias: {model.bias.data}')