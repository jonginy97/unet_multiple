class Example:
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return self.value + x

# 인스턴스 생성
example = Example(10)

# 인스턴스를 함수처럼 호출
result = example(5)
print(result)  # 출력: 15