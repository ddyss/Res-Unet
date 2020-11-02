c = 1#int(input())
for _ in range(c):
    n, m = 7, 3#map(int, input().split())
    t = 1
    flag = 1
    r = 0
    g = [[0] * n for _ in range(n)]
    while r < n:
        if flag > 0:
            for i in range(r, n - r):
                g[r][i] = t
                t += 1
            for i in range(r + 1, n - r - 1):
                g[i][n - r - 1] = t
                t += 1
            for i in range(n - r - 1, r - 1, -1):
                g[n - r - 1][i] = t
                t += 1
            for i in range(n - r - 2, r, -1):
                g[i][r] = t
                t += 1
        else:
            for i in range(r, n - r):
                g[i][r] = t
                t += 1
            for i in range(r + 1, n - r - 1):
                g[n - r - 1][i] = t
                t += 1
            for i in range(n - r - 1, r - 1, -1):
                g[i][n - r - 1] = t
                t += 1
            for i in range(n - r - 2, r, -1):
                g[r][i] = t
                t += 1
        r += 1
        flag *= -1
    # if n % 2:
    #     g[n//2][n//2] = n * n
    # for _ in range(m):
    #     x, y = map(int, input().split())
    #     print(g[x][y])
print(g)


