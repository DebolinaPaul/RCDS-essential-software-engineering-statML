def pivot_sort(arr):
    """
    Sort a list in ascending order using pivot sort (no quicksort here).
    
    Args:
        arr: List of comparable elements to sort
        
    Returns:
        A new sorted list in ascending order
    """
    if len(arr) <= 1:
        return arr
    
    # Choose pivot as middle element
    pivot = arr[len(arr) // 2]
    
    # Partition into three lists: less than, equal to, and greater than pivot
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    
    # Recursively sort and combine
    return pivot_sort(less) + equal + pivot_sort(greater)


# Test the function
if __name__ == "__main__":
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9],
        [1],
        [],
        [3, 3, 1, 2, 3, 1],
    ]
    
    for test in test_cases:
        print(f"Original: {test}")
        print(f"Sorted:   {pivot_sort(test)}\n")
