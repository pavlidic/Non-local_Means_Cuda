#ifndef SORT_H
#define SORT_H

// credits: GeeksForGeeks. Link: https://www.geeksforgeeks.org/quick-sort/

void swap(float* a, float* b){
    float t = *a;
    *a = *b;
    *b = t;
}

float partition (float arr[], int low, int high)
{
    float pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element and indicates the right position of pivot found so far

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(float arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}


#endif //CUDATEST_SORT_H
