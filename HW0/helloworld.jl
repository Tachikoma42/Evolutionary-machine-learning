using Random


    tobeSorted = String["ojSpWT","Av5ZyF","HQXaXF","l2638j","h7GP1G"  ]

# for i in 1:5 
#     push!(tobeSorted, randstring(6))
# end
# print("Before sorting:\n")
# for j in tobeSorted
#     print(j)
#     print('\n')
# end
# print('\n')

function bubbleSort(tobeSorted)
    for i in 1:length(tobeSorted)-1
        for j in 1:length(tobeSorted)-1
            if cmp(tobeSorted[j],tobeSorted[j+1]) == 1
                # equal to 1 means that the first element is bigger than the second element
                # so we need to swap them
                # 0 , -1 means that the first element is smaller or equal to the second element
                # no need to swap
                tobeSorted[j],tobeSorted[j+1] = tobeSorted[j+1],tobeSorted[j]
            end
                
        end
    end
    return tobeSorted
end

afterSort = bubbleSort(tobeSorted)
# afterSort = sort(tobeSorted, alg = MergeSort) 

# print("After sorting:\n")
# for j in afterSort
#     print(j)
#     print('\n')
# end
if afterSort == sort(tobeSorted, alg = MergeSort)
    print("Sorting is correct")
else
    print("Sorting is incorrect\n")

 for j in afterSort
    print(j)
    print('\n')
end
print('\n')
    for j in sort(tobeSorted, alg = MergeSort)
        print(j)
    print('\n')
    end
 end
