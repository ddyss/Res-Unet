def quick_sort(lists, i, j):
    if i < j:
        pivot = lists[i]
        low = i
        high = j
        while i < j:
            while i < j and lists[j] >= pivot:# 一直找，直到出现小于基准值的值，跳出循环，把小的值放到pivot前面去
                j -= 1
            # lists[i]=lists[j] #先找到大于的数后，放前面 过去  https://zhuanlan.zhihu.com/p/110052556

            while i < j and lists[i] <= pivot:
                i += 1
            # lists[j]=lists[i] #找到小于的数后，再放后面 回来
            lists[j], lists[i] = lists[i], lists[j] #把上面两行合并为一行
        lists[i] = pivot #每次的pivot是确定的，最开始的pivot找到了其现在所在的位置i，放到lists中
        # lists[j] = pivot
        quick_sort(lists, low, i - 1)  #递归
        quick_sort(lists, i + 1, high)
    return lists  #直到满足条件i>=j返回


res = [65, 18, 11, 52, 12, 1, 37, 51, 31, 88, 40, 88, 49, 100, 20, 40, 14, 4, 69, 96, 27]
print(quick_sort(res,0,len(res)-1))
