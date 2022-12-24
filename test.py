import numpy as np


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num
def mostCommon(lst):
   flatList = [el for sublist in lst for el in sublist]
   return max(flatList, key=flatList.count)


# Driver code
lst = [[10, 20, 30], [20, 50, 10], [30, 50, 10],[10, 20, 30]]
print(lst)
sub=list(np.transpose(lst))
for i in range (len(sub)):
    #flatList = [el for sublist in s for el in sublist]
    a=max(list(sub[i]), key=list(sub[i]).count)
    #a=mostCommon(list(s))
    print(a)

