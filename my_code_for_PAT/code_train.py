while 1:
    n=input()
    if n!="":
        n=int(n)
        a=[int(i) for i in input().split()]
        b=sorted(a)
        c=[a[i] for i in range(n) if a[i]!=b[i]]#c 代表的是与之不同的一段，可能在中间
        print(c)
        start=a.index(c[0])#c 首部在a中对应的索引
        end=a.index(c[-1])#c 尾部在a中对应的索引
        print(a)
        print(start,end)
        l = 0
        r = n - 1
        if a[:start]+list(reversed(a[start:end+1]))+a[end+1:]==b:
            print('yes')
        else:
            print('no')
        # i = 0
        # while i < n:
        #     if list(reversed(a[:i])) + a[i:n] == b or a[:i] + list(reversed(a[i:n])) == b:
        #         print("yes")
        #     i += 1
        # print("no")
    else:
        break
# 2 1 3 4 5
# [2, 1]
# [2, 1, 3, 4, 5]
# 0 1
# yes

# 2 1 3 5 4
# [2, 1, 5, 4]
# [2, 1, 3, 5, 4]
# 0 4
# no

# 8
# 1 2 5 9 4 3 6 7
# [5, 9, 4, 3, 6, 7]
# [1, 2, 5, 9, 4, 3, 6, 7]
# 2 7
# no

# 9
# 6 4 5 3 8 9 2 1 7
# [6, 4, 5, 3, 8, 9, 2, 1, 7]
# [6, 4, 5, 3, 8, 9, 2, 1, 7]
# 0 8
# no