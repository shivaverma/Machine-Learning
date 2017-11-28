#!/usr/bin/python
import heapq


def comp(self, other):

    return cmp(self.error, other.error)


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).
        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    h = []
    for i in range(len(ages)):
        error = -abs(predictions[i]-net_worths[i])
        heapq.heappush(h, (error, ages[i], net_worths[i]))

    for i in range(9):
        heapq.heappop(h)      # removing outlier with higher error

    cleaned_data = []

    for i in h:
        tup = (i[1], i[2], i[0])
        cleaned_data.append(tup)

    return cleaned_data

