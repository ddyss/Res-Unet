from collections import defaultdict
c = int(input())
for _ in range(c):
    n, d = map(int, input().split())
    data = []
    for _ in range(n):
        tmp = list(map(int, input().split()))
        data.append([tmp[:-1],tmp[-1]])
    m, k = map(int, input().split())
    hidden = list(map(int, input().split()))
    g = {}
    for _ in range(k):
        s1, s2, s3 = input().split()
        s3 = int(s3)
        g[(s1, s2)] = s3
    inp = []
    for k in g.keys():
        inp.append(k)

    t1 = len(hidden)
    t2 = max(hidden)
    tmp = [[0] * (t1 + 1) for _ in range(t2)]

    loss_cur = 0
    for d in data:
        shuju = d[0]
        target = d[1]
        #首先获得输入层到隐层的计算结果
        #然后获得每一个隐层的计算结果
        #最后获得最后一个隐层到输出层的计算结果

        #计算loss_cur
        loss_cur += abs(loss - target)

    n = len(inp)
    loss_new = float('inf')
    for i in range(n):
        new_inp = inp[:i] + inp[i + 1:]
        tmp_loss = 0
        for d in data:
            shuju = d[0]
            target = d[1]

    for i in data:
        shuju = i[0]
        target = i[1]
        tmp = defaultdict(list)
        for a in range(1, d + 1):
            for b in range(1, hidden[0] + 1):
                if ((f'i_{a}',f'h_1_{b}') in inp):
                    q = f'i_{a}'
                    w = f'h_1{b}'
                    if not tmp[w]:
                        tmp[w] = shuju[a - 1] * g[(q, w)]
                    else:
                        tmp[w] += shuju[a - 1] * g[{q,w}]
