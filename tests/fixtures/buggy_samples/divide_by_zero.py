"""缺陷样例：计算平均值时未处理空序列，`len(values)` 为 0 时抛 ZeroDivisionError。

预期：每轮审查均报同类【严重】（语义型缺陷，脚本化修复不改动），
至修复轮次上限以残留清单收尾（退出码 1）。
"""


def average(values):
    """返回列表中数值的平均值。"""
    return sum(values) / len(values)


def main():
    """演示入口。"""
    print(average([1, 2, 3]))


if __name__ == "__main__":
    main()
