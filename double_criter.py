import torch
import torch.nn as nn
import torch.optim as optim

# 예제 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

model = SimpleModel()
criterion1 = nn.MSELoss()  # 손실 함수 1
criterion2 = nn.L1Loss()   # 손실 함수 2
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 입력과 실제 출력
input = torch.tensor([[1.0]])
target1 = torch.tensor([[2.0]])
target2 = torch.tensor([[1.0]])

# Forward Pass
output1, output2 = model(input)
loss1 = criterion1(output1, target1)
loss2 = criterion2(output2, target2)

# Total Loss
total_loss = loss1 + loss2

# Backward Pass
optimizer.zero_grad()  # 기존 기울기 초기화
total_loss.backward()  # 역전파를 통해 기울기 계산

# 미분 계산된 기울기 출력
print(f'Gradient of weight (fc1): {model.fc1.weight.grad}')
print(f'Gradient of bias (fc1): {model.fc1.bias.grad}')
print(f'Gradient of weight (fc2): {model.fc2.weight.grad}')
print(f'Gradient of bias (fc2): {model.fc2.bias.grad}')








optimizer.step()       # 옵티마이저를 통해 가중치 업데이트

print(f'Updated weights (fc1): {model.fc1.weight.data}')
print(f'Updated bias (fc1): {model.fc1.bias.data}')
print(f'Updated weights (fc2): {model.fc2.weight.data}')
print(f'Updated bias (fc2): {model.fc2.bias.data}')