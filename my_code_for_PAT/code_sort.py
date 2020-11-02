import random
#
# res = []
# for i in range(1, 20):
#     res.append(random.randint(1, 100))
#
# print(res)
# res = (random.randint(1,100) for _ in range(20)) #type generator


"""
def quick_sort(lists, i, j):
    if i >= j:
        return
    pivot = lists[i]
    low = i
    high = j
    while i < j:
        while i < j and lists[j] >= pivot:# 一直找，直到出现小于基准值的值，跳出循环，把小的值放到pivot前面去
            j -= 1
        lists[i]=lists[j] #先找到大于的数后，放前面 过去  https://zhuanlan.zhihu.com/p/110052556

        while i < j and lists[i] <= pivot:
            i += 1
        lists[j]=lists[i] #找到小于的数后，再放后面 回来
    lists[j] = pivot #每次的pivot是确定的，现在的i/j，就是pivot应该待的位置，放到lists中
    quick_sort(lists, low, i - 1)  #递归
    quick_sort(lists, i + 1, high)
    return lists  #直到满足条件i>=j返回
# """
"""
def quick_sort(nums):
    n = len(nums)

    def quick(left, right):
        if left >= right:
            return nums
        pivot = left  #这种情况下必须只是索引，不能是值
        i = left
        j = right
        while i < j:
            while i < j and nums[j] > nums[pivot]:
                j -= 1
            while i < j and nums[i] <= nums[pivot]:
                i += 1
            nums[i], nums[j] = nums[j], nums[i] #可以相互交换，这样可读性更好
        nums[pivot], nums[j] = nums[j], nums[pivot] #这个才对应知乎https://zhuanlan.zhihu.com/p/110052556的动图
        quick(left, j - 1)
        quick(j + 1, right)
        return nums

    return quick(0, n - 1)


# """



"""
def quick_sort(array, l, r):
    if l < r:
        q = partition(array, l, r)
        quick_sort(array, l, q - 1)
        quick_sort(array, q + 1, r)


def partition(array, l, r):
    x = array[r]
    i = l - 1
    for j in range(l, r):
        if array[j] <= x:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1
"""



'''
def quick_sort(L):  # 时间复杂度最坏是n^2，平均是nlog（n），空间复杂度最坏是n，平均是log（n）
    n = len(L)
    if n <= 1:
        return L

    pivot = L[0]  # int
    print(pivot, type(pivot))
    lesser = [item for item in L[1:] if item <= pivot]
    greater = [item for item in L[1:] if item >= pivot]

    return quick_sort(lesser) + [pivot] + quick_sort(greater)  # list + int  #递归就是终止条件
# '''

# out = quick_sort(res, 0, len(res) - 1)

# print(quick_sort(res))
# for i in quick_sort(res,0,len(res)-1):
#     print(i,end=' ')

"""
res = [38, 21, 64, 41, 93, 90, 50, 99, 94, 22, 68, 1, 16, 32, 12, 43, 5, 73, 23]
def selection_sort(nums):
    n = len(nums)
    count = 0
    for i in range(n):
        for j in range(i+1,n): #这属于反向冒泡排序，（选择排序selection sort）把最小值放到最前面，然后第二小的值放在第二个，类推
            if nums[i] > nums[j]:
                nums[i],nums[j] = nums[j],nums[i]
            else:
                continue     #优化  171 减少到 101

            count += 1
    print(count)
    return nums
print(selection_sort(res))

def bubble_sort(nums):
    n = len(nums)
    count = 0
    for i in range(n):
        for j in range(1,n-i): #冒泡，最大值放到最后面，第二大的值放在倒数第二个
            if nums[j-1] > nums[j]:
                nums[j-1],nums[j] = nums[j],nums[j-1]
            else:
                continue  #优化  171 减少到 101
            count += 1
    print(count)
    return nums

print(bubble_sort(res))

# """

"""
res = [38, 21, 64, 41, 93, 90, 50, 99, 94, 22, 68, 1, 16, 32, 12, 43, 5, 73, 23]

def insert_sort(nums):
    n = len(nums)
    for i in range(1, n):
        while i > 0 and nums[i - 1] > nums[i]:
            nums[i - 1], nums[i] = nums[i], nums[i - 1]
            i -= 1
    return nums

print(insert_sort(res))
# """

res = [38, 21, 64, 41, 93, 90, 50, 99, 94, 22, 68, 1, 16, 32, 12, 43, 5, 73, 23]
print(sorted(res,key=1))