def quick_sort(nums):
    n = len(nums)

    def quick(left, right):
        if left >= right:
            return nums

        pivot = left
        i = left
        j = right
        while i < j:
            while i < j and nums[j] >= nums[pivot]:
                j -= 1
            while i < j and nums[i] <= pivot:
                i += 1
            nums[i], nums[j] = nums[j], nums[i]

        pivot, nums[j] = nums[j], pivot
        quick(left, j - 1)
        quick(j + 1, right)
        return nums

    return quick(0, n - 1)

res = [65, 18, 11, 52, 12, 1, 37, 51, 31, 88, 40, 88, 49, 100, 20, 40, 14, 4, 69, 96, 27]
print(quick_sort(res))