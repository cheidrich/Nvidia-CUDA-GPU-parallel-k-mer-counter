#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "warpcore/include/warpcore.cuh"
#include <cooperative_groups.h>



#include <stdio.h>
#include <fstream>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <string>



int khost;
__device__ int k;
__device__ int arrayPosition = 0;



__global__ void countReadsFA(char file[], uint64_t* size, int* totalReadCount)
{
    int readCount = 0;
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t numberOfThreads = blockDim.x * gridDim.x;
    uint64_t chunksize = *size / numberOfThreads;
    uint64_t begin = tid * chunksize;

    // Distribute Leftover
    if (tid < (*size % numberOfThreads)) {
        chunksize++;
        begin += tid;
    }
    else {
        begin += (*size % numberOfThreads);
    }

    // Search and count > symbols
    for (size_t i = 0; i < chunksize; i++)
    {
        if (file[begin + i] == '>')
        {
            readCount++;
        }
    }

    atomicAdd(totalReadCount, readCount);
}

__global__ void countReadsFQ(char file[], uint64_t* size, int* totalReadCount)
{
    int readCount = 0;
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t numberOfThreads = blockDim.x * gridDim.x;
    uint64_t chunksize = *size / numberOfThreads;
    uint64_t begin = tid * chunksize;

    // distribute Leftover
    if (tid < (*size % numberOfThreads)) {
        chunksize++;
        begin += tid;
    }
    else {
        begin += (*size % numberOfThreads);
    }

    // search for @ read markers
    for (size_t i = 0; i < chunksize; i++)
    {
        if (file[begin + i] == '@')
        {
            if (begin + i != 0)
            {
                /* Check whether @ is at the beginning of the line and 
                *  symbol at the beginning of the next line is not @
                *  to know whether @ is a quality score or marker
                */
                if (file[begin + i - 1] == '\n')
                {
                    while (file[begin + i] != '\n')
                    {
                        i++;
                    }
                    i++;
                    if (file[begin + i] != 0)
                    {
                        if (file[begin + i] != '@')
                        {
                            readCount++;
                        }
                    }
                }
            }
            else
            {
                readCount++;
            }
        }
    }

    atomicAdd(totalReadCount, readCount);
}

__global__ void findReadPositionsFA(char file[], uint64_t readPositions[], uint64_t* size)
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t numberOfThreads = blockDim.x * gridDim.x;
    int localPosition = 0;
    uint64_t chunksize = *size / numberOfThreads;
    uint64_t begin = tid * chunksize;



    // distribute Leftover
    if (tid < (*size % numberOfThreads)) {
        chunksize++;
        begin += tid;
    }
    else {
        begin += (*size % numberOfThreads);
    }

    // search for Position of > symbols
    for (size_t i = 0; i < chunksize; i++)
    {
        if (file[begin + i] == '>')
        {
            localPosition = atomicAdd(&arrayPosition, 1);
            atomicCAS(&readPositions[localPosition], readPositions[localPosition], begin + i);
        }
    }
}

__global__ void findReadPositionsFQ(char file[], uint64_t readPositions[], uint64_t* size)
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t numberOfThreads = blockDim.x * gridDim.x;
    int localPosition = 0;
    uint64_t chunksize = *size / numberOfThreads;
    uint64_t begin = tid * chunksize;



    // distribute Leftover
    if (tid < (*size % numberOfThreads)) {
        chunksize++;
        begin += tid;
    }
    else {
        begin += (*size % numberOfThreads);
    }

    // search for @ read markers
    for (size_t i = 0; i < chunksize; i++)
    {
        if (file[begin + i] == '@')
        {
            if (begin + i != 0)
            {
                /* Check whether @ is at the beginning of the line and
                *  symbol at the beginning of the next line is not @
                *  to know whether @ is a quality score or marker
                */
                if (file[begin + i - 1] == '\n')
                {
                    while (file[begin + i] != '\n')
                    {
                        i++;
                    }
                    i++;
                    if (file[begin + i] != 0)
                    {
                        if (file[begin + i] != '@')
                        {
                            localPosition = atomicAdd(&arrayPosition, 1);
                            atomicCAS(&readPositions[localPosition], readPositions[localPosition], begin + i);
                        }
                    }
                }
            }
            else
            {
                localPosition = atomicAdd(&arrayPosition, 1);
                atomicCAS(&readPositions[localPosition], readPositions[localPosition], begin + i);
            }
        }
    }
}

__device__ uint64_t calcKmerKey(char sequence[]) {
    uint64_t count = 0;
    int temp = 0;
    uint64_t power = 1;
    for (size_t i = 0; i < k; i++)
    {
        switch (sequence[i])
        {
        case 'A': case 'a':
            temp = 0;
            break;
        case 'C': case 'c':
            temp = 1;
            break;
        case 'G': case 'g':
            temp = 2;
            break;
        case 'T': case 't':
            temp = 3;
            break;
        default:
            return 0;
        }
        count += power * temp;
        power *= 4;
    }
    return count;
}

__global__ void countKmersFA(char file[], uint64_t readPositions[], int* numberOfReads, warpcore::CountingHashTable<std::uint64_t, std::uint64_t> cht, warpcore::BloomFilter<std::uint64_t, warpcore::hashers::MurmurHash<uint64_t>, std::uint64_t> bf)
{
    using namespace warpcore;

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t numberOfThreads = blockDim.x * gridDim.x;
    uint64_t chunksize = *numberOfReads / numberOfThreads;
    uint64_t begin = tid * chunksize;
    char sequence[31];
    std::uint64_t key = 0;



    // Cooperative Group and Thread Divergence Variables
    auto cg1 = cg::tiled_partition<cht.cg_size()>(cg::this_thread_block());
    auto cg2 = cg::tiled_partition<bf.cg_size()>(cg::this_thread_block());
    uint32_t activeMask1 = 1;
    uint32_t activeMask2 = 1;
    uint32_t validKeyMask = 1;
    bool validKey = false;
    bool activeThread1 = true;
    bool activeThread2 = true;



    // distribute Leftover
    if (tid < (*numberOfReads % numberOfThreads)) {
        chunksize++;
        begin += tid;
    }
    else {
        begin += (*numberOfReads % numberOfThreads);
    }

    if (chunksize == 0)
    {
        activeThread1 = false;
    }


    char* iterator;
    int i = 0;


    /* avoids thread divergence, equivalent to: for (size_t i = 0; i < chunksize; i++)
    *  iterates over all reads assigned
    */
    while (activeMask1)
    {
        iterator = &file[readPositions[begin + i]];

        // go from description to beginning of the actual sequence
        while (*iterator != '\n')
        {
            iterator++;
        }
        iterator++;

        // fillup sequence array, let first one empty to fit iterative loop design of rest of code
        for (int j = 1; j < k; j++)
        {
            if (*iterator == 'A' || *iterator == 'C' || *iterator == 'G' || *iterator == 'T') {
                sequence[j] = *iterator;
            }
            else
            {
                j--;
            }
            iterator++;
        }

        /* avoids thread divergence, equivalent to: while (*iterator != '>' && *iterator != 0)
        *  iterates over all k-mers in the read
        */
        while (activeMask2)
        {
            validKey = false;
            if ((*iterator == 'A' || *iterator == 'C' || *iterator == 'G' || *iterator == 'T') && activeThread1 == true && activeThread2 == true)
            {
                for (size_t j = 0; j < k; j++)
                {
                    if (j == k - 1)
                    {
                        sequence[j] = *iterator;
                    }
                    else
                    {
                        sequence[j] = sequence[j + 1];
                    }
                }
                key = calcKmerKey(sequence);
                validKey = bf.insert_and_query(key, cg2);
                // to disable filter replace ^ with: validKey = true;
            }

            // insert k-mers with cooperative group
            validKeyMask = cg1.ballot(validKey);
            while (validKeyMask)
            {
                const auto leader = __ffs(validKeyMask) - 1;
                const auto filtered_key = cg1.shfl(key, leader);
                cht.insert(filtered_key, cg1);
                validKeyMask ^= 1UL << leader;
            }

            if (*iterator != '>' && *iterator != 0)
            {
                iterator++;
            }
            else
            {
                // set bool for 2. while loop mask (avoids thread divergence)
                activeThread2 = false;
            }
            activeMask2 = cg1.ballot(activeThread2);
        }

        activeThread2 = true;
        i++;

        // set bool for 1. while loop mask (avoids thread divergence)
        if (!(i < chunksize))
        {
            activeThread1 = false;
        }
        activeMask1 = cg1.ballot(activeThread1);
    }
}

__global__ void countKmersFQ(char file[], uint64_t readPositions[], int* numberOfReads, warpcore::CountingHashTable<std::uint64_t, std::uint64_t> cht, warpcore::BloomFilter<std::uint64_t, warpcore::hashers::MurmurHash<uint64_t>, std::uint64_t> bf)
{
    using namespace warpcore;

    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t numberOfThreads = blockDim.x * gridDim.x;
    uint64_t chunksize = *numberOfReads / numberOfThreads;
    uint64_t begin = tid * chunksize;
    char sequence[31];
    std::uint64_t key = 0;


    // Cooperative Group and Thread Divergence Variables
    auto cg1 = cg::tiled_partition<cht.cg_size()>(cg::this_thread_block());
    auto cg2 = cg::tiled_partition<bf.cg_size()>(cg::this_thread_block());
    uint32_t activeMask1 = 1;
    uint32_t activeMask2 = 1;
    uint32_t validKeyMask = 1;
    bool validKey = false;
    bool activeThread1 = true;
    bool activeThread2 = true;



    // distribute Leftover
    if (tid < (*numberOfReads % numberOfThreads)) {
        chunksize++;
        begin += tid;
    }
    else {
        begin += (*numberOfReads % numberOfThreads);
    }

    if (chunksize == 0)
    {
        activeThread1 = false;
    }


    char* iterator;
    int i = 0;
    
    /* avoids thread divergence, equivalent to: for (size_t i = 0; i < chunksize; i++)
    *  iterates over all reads assigned
    */
    while (activeMask1)
    {
        iterator = &file[readPositions[begin + i]];

        // go from description to beginning of the actual sequence
        while (*iterator != '\n')
        {
            iterator++;
        }
        iterator++;

        // fillup sequence array, let first one empty to fit iterative loop design of rest of code
        for (int j = 1; j < k; j++)
        {
            if (*iterator != '\n' && *iterator != ' ' && *iterator != '\r') {
                sequence[j] = *iterator;
            }
            else
            {
                j--;
            }
            iterator++;
        }

        /* avoids thread divergence, equivalent to: while (*iterator != '+' && *iterator != 0)
        *  iterates over all k-mers in the read
        */
        while (activeMask2)
        {
            validKey = false;
            if ((*iterator == 'A' || *iterator == 'C' || *iterator == 'G' || *iterator == 'T') && activeThread1 == true && activeThread2 == true)
            {
                for (size_t j = 0; j < k; j++)
                {
                    if (j == k - 1)
                    {
                        sequence[j] = *iterator;
                    }
                    else
                    {
                        sequence[j] = sequence[j + 1];
                    }
                }
                key = calcKmerKey(sequence);
                validKey = bf.insert_and_query(key, cg2);
                // to disable filter replace ^ with: validKey = true;
            }

            // insert k-mers with cooperative group
            validKeyMask = cg1.ballot(validKey);
            while (validKeyMask)
            {
                const auto leader = __ffs(validKeyMask) - 1;
                const auto filtered_key = cg1.shfl(key, leader);
                cht.insert(filtered_key, cg1);
                validKeyMask ^= 1UL << leader;
            }

            if (*iterator != '+' && *iterator != 0)
            {
                iterator++;
            }
            else
            {
                // set bool for 2. while loop mask (avoids thread divergence)
                activeThread2 = false;
            }
            activeMask2 = cg1.ballot(activeThread2);
        }

        activeThread2 = true;
        i++;

        // set bool for 1. while loop mask (avoids thread divergence)
        if (!(i < chunksize))
        {
            activeThread1 = false;
        }
        activeMask1 = cg1.ballot(activeThread1);
    }
}


void calcKmerFromKey(char* out, uint64_t key) {
    char bases[] = "ACGT";
    for (int i = 0; i < khost; i++)
    {
        out[i] = bases[key % 4];
        key /= 4;
        out[i + 1] = 0;
    }
}


void printKmerCounts(std::uint64_t keys[], std::uint64_t counts[], int numberOfKmers, int threshold = 0) {
    char* bases;
    bases = new char[khost];
    int counter = 0;
    printf("Of the first %d %d-mer all above threshold %d: \n", numberOfKmers, khost, threshold);
    for (int i = 0; i < numberOfKmers; i++)
    {
        if (counts[i] >= threshold)
        {
            calcKmerFromKey(bases, keys[i]);
            //add 1 to account for filter loss
            printf("Count of %d-mer %s is %d \n", khost, bases, ((int)counts[i]) + 1);
            counter++;
        }
    }
}



int main()
{
    // User Input Dialogue

    std::string directory;
    printf("Enter File Directory: \n");
    std::cin >> directory;
    char charFastqOrFasta;
    bool boolFastq = false;
    printf("Enter File Format (Fastq = q / Fasta = a): \n");
    std::cin >> charFastqOrFasta;
    if (charFastqOrFasta == 'q')
    {
        boolFastq = true;
    }
    int ktemp;
    printf("Enter k Value (2 - 31): \n");
    std::cin >> ktemp;
    int threshold = 0;
    printf("Enter counter threshold (0 recommended): \n");
    std::cin >> threshold;
    bool printAllBool = false;
    char printAllChar;
    printf("Print all counts above threshold? (y = all / n = only first 10 ; n recommended): \n");
    std::cin >> printAllChar;
    if (printAllChar == 'y')
    {
        printAllBool = true;
    }
    khost = ktemp;
    cudaMemcpyToSymbol(k, &ktemp, sizeof(int));





    // Read in File

    std::ifstream t;
    uint64_t fileSize;
    t.open(directory);
    t.seekg(0, std::ios::end);
    fileSize = t.tellg();
    t.seekg(0, std::ios::beg);
    printf("File Size: %llu Bytes \n", fileSize);
    char* file = nullptr;
    cudaMallocHost(&file, fileSize * sizeof(char));
    t.read(file, fileSize);
    t.close();
    char* d_file;
    uint64_t* d_file_size = 0;





    // Cuda Parameters

    cudaError_t cudaStatus;
    uint64_t numberOfThreads = 1024;
    uint64_t numberOfBlocks = 2325860;
    int totalReadCount = 0;
    int* d_totalReadCount = 0;





    // Copy Data to Device

    cudaStatus = cudaMalloc(&d_totalReadCount, sizeof(int));
    cudaStatus = cudaMalloc(&d_file_size, sizeof(uint64_t));
    cudaStatus = cudaMalloc(&d_file, fileSize);
    cudaStatus = cudaMemcpy(d_file, file, fileSize, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_file_size, &fileSize, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_totalReadCount, &totalReadCount, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!1");
    }





    // Take execution Time

    long startTime = clock();


    


    // Set and launch 1. Kernel

    if (fileSize < numberOfBlocks * numberOfThreads)
    {
        numberOfBlocks = fileSize / 1024 + 1;
    }

    if (boolFastq == true)
    {
        countReadsFQ << <numberOfBlocks, numberOfThreads >> > (d_file, d_file_size, d_totalReadCount);
    }
    else
    {
        countReadsFA << <numberOfBlocks, numberOfThreads >> > (d_file, d_file_size, d_totalReadCount);
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("kernel 1 (countReads) : ");
        printf(cudaGetErrorName(cudaStatus));
        printf("\n");
        printf(cudaGetErrorString(cudaStatus));
        printf("\n");
    }




    // Get Number of Reads from Device, create according Array for Positions and copy all (back) to Device

    cudaStatus = cudaMemcpy(&totalReadCount, d_totalReadCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!2");
    }
    printf("Read Count: %llu\n", totalReadCount);
    uint64_t* readPositions = new uint64_t[totalReadCount];
    uint64_t* d_readPositions = 0;
    cudaStatus = cudaMalloc(&d_readPositions, sizeof(uint64_t) * totalReadCount);
    cudaStatus = cudaMemcpy(d_readPositions, readPositions, sizeof(uint64_t) * totalReadCount, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!3");
    }




    // Set and launch 2. Kernel

    if (boolFastq == true)
    {
        findReadPositionsFQ << <numberOfBlocks, numberOfThreads >> > (d_file, d_readPositions, d_file_size);
    }
    else
    {
        findReadPositionsFA << <numberOfBlocks, numberOfThreads >> > (d_file, d_readPositions, d_file_size);
    }
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("kernel 2 (findReadPositions) : ");
        printf(cudaGetErrorName(cudaStatus));
        printf("\n");
        printf(cudaGetErrorString(cudaStatus));
        printf("\n");
    }





    // Calculate, set, and init Hash Table and Bloom Filter and their Capacity and Size

    uint64_t maxKmers = fileSize - (khost - 1) * totalReadCount;
    uint64_t possibleKmers = pow(4, khost);
    uint64_t capacityFilter = possibleKmers;
    uint64_t capacityTable = maxKmers;
    if (fileSize > 6000000000)
    {
        capacityTable /= 12;
    }
    else if (fileSize > 4000000000)
    {
        capacityTable /= 8;
    }
    else if (fileSize > 1000000000)
    {
        capacityTable /= 6;
    }

    if (possibleKmers < capacityTable)
    {
        capacityTable = possibleKmers;
    }
    if (maxKmers < possibleKmers)
    {
        capacityFilter = maxKmers;
    }
    warpcore::CountingHashTable<std::uint64_t, std::uint64_t> cht((capacityTable));
    const uint8_t k2 = 6;
    const uint64_t numberOfBits = ceil((capacityFilter * log(0.02)) / log(1 / pow(2, log(2))));
    warpcore::BloomFilter<std::uint64_t, warpcore::hashers::MurmurHash<uint64_t>, std::uint64_t> bf = warpcore::BloomFilter<std::uint64_t, warpcore::hashers::MurmurHash<uint64_t>, std::uint64_t>(numberOfBits, k2);





    // Set and launch 3. Kernel

    cudaStatus = cudaMemcpy(d_totalReadCount, &totalReadCount, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!4");
    }

    if (totalReadCount < 2325860 * numberOfThreads)
    {
        numberOfBlocks = totalReadCount / 1024 + 1;
    }
    else
    {
        numberOfBlocks = 2325860;
    }

    if (boolFastq == true)
    {
        countKmersFQ << <numberOfBlocks, numberOfThreads >> > (d_file, d_readPositions, d_totalReadCount, cht, bf);
    }
    else
    {
        countKmersFA << <numberOfBlocks, numberOfThreads >> > (d_file, d_readPositions, d_totalReadCount, cht, bf);
    }

    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("kernel 3 (countReads) : ");
        printf(cudaGetErrorName(cudaStatus));
        printf("\n");
        printf(cudaGetErrorString(cudaStatus));
        printf("\n");
    }





    // Get Data back from Device plus necessary Allocations

    int hash_size = (int)cht.size();

    std::uint64_t* keys_out_h;
    std::uint64_t* counts_out_h;
    std::uint64_t* keys_out_d;
    std::uint64_t* counts_out_d;
    std::uint64_t output_size_h;

    cudaMalloc(&keys_out_d, sizeof(std::uint64_t) * hash_size);
    cudaMalloc(&counts_out_d, sizeof(std::uint64_t) * hash_size);
    cudaMallocHost(&keys_out_h, sizeof(std::uint64_t) * hash_size);
    cudaMallocHost(&counts_out_h, sizeof(std::uint64_t) * hash_size);

    cht.retrieve_all(keys_out_d, counts_out_d, output_size_h);
    printf("Output Size: %d \n", (int)output_size_h);

    cudaStatus = cudaMemcpy(keys_out_h, keys_out_d, sizeof(std::uint64_t) * output_size_h, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(counts_out_h, counts_out_d, sizeof(std::uint64_t) * output_size_h, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!5\n");
    }





    // take time of execution from first to third kernel

    long endTime = clock();
    long time = endTime - startTime;
    printf("Time: %ld Seconds or %ld Ticks \n", (time / (long)CLOCKS_PER_SEC), time);




    // print k-mers

    if (printAllBool == true)
    {
        printKmerCounts(keys_out_h, counts_out_h, hash_size, threshold);
    }
    else
    {
        printKmerCounts(keys_out_h, counts_out_h, 10, threshold);
    }





    cudaFree(d_readPositions);
    cudaFree(d_file);
    cudaFree(d_file_size);
    cudaFree(keys_out_d);
    cudaFree(counts_out_d);
    cudaFreeHost(file);

    delete d_file, readPositions, d_readPositions, keys_out_h, counts_out_h, keys_out_d, counts_out_d;

    return 0;
}
