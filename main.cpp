/* main.cpp
 *
 * Contains host logic:
 *   - File I/O
 *   - GPU memory allocation
 *   - Kernel launches
 *   - Retrieving results
 *   - Cleanup
 *
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <cstdint>

/* Declarations of global host/device variables (from globals.cu) */
extern int khost;
__device__ extern int k;
__device__ extern int arrayPosition;

/* Declarations of host helper functions (from kmer_utils.cu) */
void calcKmerFromKey(char* out, uint64_t key);
void printKmerCounts(std::uint64_t keys[],
                     std::uint64_t counts[],
                     int numberOfKmers,
                     int threshold);

/* Declarations of kernels (FASTA & FASTQ) */
__global__ void countReadsFA(char file[],
                             uint64_t* size,
                             int* totalReadCount);
__global__ void findReadPositionsFA(char file[],
                                    uint64_t readPositions[],
                                    uint64_t* size);
__global__ void countKmersFA(char file[],
                             uint64_t readPositions[],
                             int* numberOfReads,
                             warpcore::CountingHashTable<std::uint64_t,
                                                         std::uint64_t> cht,
                             warpcore::BloomFilter<std::uint64_t,
                                                   warpcore::hashers::MurmurHash<uint64_t>,
                                                   std::uint64_t> bf);

__global__ void countReadsFQ(char file[],
                             uint64_t* size,
                             int* totalReadCount);
__global__ void findReadPositionsFQ(char file[],
                                    uint64_t readPositions[],
                                    uint64_t* size);
__global__ void countKmersFQ(char file[],
                             uint64_t readPositions[],
                             int* numberOfReads,
                             warpcore::CountingHashTable<std::uint64_t,
                                                         std::uint64_t> cht,
                             warpcore::BloomFilter<std::uint64_t,
                                                   warpcore::hashers::MurmurHash<uint64_t>,
                                                   std::uint64_t> bf);

/* CUDA error-check macro (optional but recommended) */
#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e)        \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

int main() {
    /* 1. User dialog */
    std::string directory;
    std::cout << "Enter File Directory:\n";
    std::cin >> directory;

    char charFastqOrFasta;
    bool boolFastq = false;
    std::cout << "Enter File Format (Fastq = q / Fasta = a):\n";
    std::cin >> charFastqOrFasta;
    if (charFastqOrFasta == 'q') boolFastq = true;

    int ktemp;
    std::cout << "Enter k Value (2 - 31):\n";
    std::cin >> ktemp;

    int threshold = 0;
    std::cout << "Enter counter threshold (0 recommended):\n";
    std::cin >> threshold;

    bool printAllBool = false;
    char printAllChar;
    std::cout << "Print all counts above threshold? (y = all / n = only first 10; n recommended):\n";
    std::cin >> printAllChar;
    if (printAllChar == 'y') printAllBool = true;

    /* 2. Set khost and copy to device */
    khost = ktemp;
    cudaMemcpyToSymbol(k, &ktemp, sizeof(int));

    /* 3. Load file (host) */
    std::ifstream t(directory, std::ios::binary | std::ios::in);
    if (!t) {
        std::cerr << "Failed to open file!\n";
        return 1;
    }
    t.seekg(0, std::ios::end);
    uint64_t fileSize = t.tellg();
    t.seekg(0, std::ios::beg);
    std::cout << "File Size: " << fileSize << " Bytes\n";

    /* Allocate page-locked host memory to speed up H2D copy */
    char* file = nullptr;
    cudaMallocHost(&file, fileSize * sizeof(char));
    t.read(file, fileSize);
    t.close();

    /* 4. Allocate device memory & copy host→device */
    char* d_file;
    uint64_t* d_file_size = nullptr;
    cudaMalloc(&d_file, fileSize);
    cudaMalloc(&d_file_size, sizeof(uint64_t));
    cudaMemcpy(d_file, file, fileSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_file_size, &fileSize, sizeof(uint64_t), cudaMemcpyHostToDevice);

    /* 5. Kernel 1: Count reads */
    uint64_t numberOfThreads = 1024;
    uint64_t numberOfBlocks  = 2325860;
    int totalReadCount = 0;
    int* d_totalReadCount = nullptr;
    cudaMalloc(&d_totalReadCount, sizeof(int));
    cudaMemcpy(d_totalReadCount, &totalReadCount, sizeof(int), cudaMemcpyHostToDevice);

    if (fileSize < numberOfBlocks * numberOfThreads) {
        numberOfBlocks = fileSize / numberOfThreads + 1;
    }

    if (boolFastq) {
        countReadsFQ<<<numberOfBlocks, numberOfThreads>>>(d_file, d_file_size, d_totalReadCount);
    } else {
        countReadsFA<<<numberOfBlocks, numberOfThreads>>>(d_file, d_file_size, d_totalReadCount);
    }
    cudaCheckError();

    /* 6. Retrieve totalReadCount */
    cudaMemcpy(&totalReadCount, d_totalReadCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Read Count: " << totalReadCount << "\n";

    /* 7. Allocate and initialize readPositions array */
    uint64_t* readPositions = new uint64_t[totalReadCount];
    uint64_t* d_readPositions = nullptr;
    cudaMalloc(&d_readPositions, sizeof(uint64_t) * static_cast<size_t>(totalReadCount));
    cudaMemcpy(d_readPositions, readPositions, sizeof(uint64_t) * static_cast<size_t>(totalReadCount), cudaMemcpyHostToDevice);

    /* 8. Kernel 2: Find read positions */
    if (boolFastq) {
        findReadPositionsFQ<<<numberOfBlocks, numberOfThreads>>>(d_file, d_readPositions, d_file_size);
    } else {
        findReadPositionsFA<<<numberOfBlocks, numberOfThreads>>>(d_file, d_readPositions, d_file_size);
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    /* 9. Initialize hash table & Bloom filter */
    uint64_t maxKmers      = fileSize - (static_cast<uint64_t>(khost) - 1) * totalReadCount;
    uint64_t possibleKmers = static_cast<uint64_t>(std::pow(4, khost));
    uint64_t capacityFilter = possibleKmers;
    uint64_t capacityTable  = maxKmers;
    if (fileSize > 6000000000ULL) capacityTable /= 12;
    else if (fileSize > 4000000000ULL) capacityTable /= 8;
    else if (fileSize > 1000000000ULL) capacityTable /= 6;
    if (possibleKmers < capacityTable) capacityTable = possibleKmers;
    if (maxKmers < possibleKmers) capacityFilter = maxKmers;

    warpcore::CountingHashTable<uint64_t, uint64_t> cht(capacityTable);
    const uint8_t k2 = 6;
    uint64_t numberOfBits = static_cast<uint64_t>(
        std::ceil((static_cast<long double>(capacityFilter) * std::log(0.02L)) /
                  std::log(1.0L / std::pow(2.0L, std::log(2.0L))))
    );
    warpcore::BloomFilter<uint64_t,
                          warpcore::hashers::MurmurHash<uint64_t>,
                          uint64_t> bf(numberOfBits, k2);

    /* 10. Kernel 3: Count K-mers */
    cudaMemcpy(d_totalReadCount, &totalReadCount, sizeof(int), cudaMemcpyHostToDevice);
    if (totalReadCount < 2325860ULL * numberOfThreads) {
        numberOfBlocks = totalReadCount / numberOfThreads + 1;
    } else {
        numberOfBlocks = 2325860;
    }

    if (boolFastq) {
        countKmersFQ<<<numberOfBlocks, numberOfThreads>>>(d_file, d_readPositions, d_totalReadCount, cht, bf);
    } else {
        countKmersFA<<<numberOfBlocks, numberOfThreads>>>(d_file, d_readPositions, d_totalReadCount, cht, bf);
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    /* 11. Retrieve results */
    int hash_size = static_cast<int>(cht.size());
    uint64_t* keys_out_d   = nullptr;
    uint64_t* counts_out_d = nullptr;
    uint64_t* keys_out_h   = nullptr;
    uint64_t* counts_out_h = nullptr;
    uint64_t output_size_h = 0;

    cudaMalloc(&keys_out_d, sizeof(uint64_t) * static_cast<size_t>(hash_size));
    cudaMalloc(&counts_out_d, sizeof(uint64_t) * static_cast<size_t>(hash_size));
    cudaMallocHost(&keys_out_h, sizeof(uint64_t) * static_cast<size_t>(hash_size));
    cudaMallocHost(&counts_out_h, sizeof(uint64_t) * static_cast<size_t>(hash_size));

    cht.retrieve_all(keys_out_d, counts_out_d, output_size_h);
    std::cout << "Output Size: " << output_size_h << "\n";

    cudaMemcpy(keys_out_h, keys_out_d, sizeof(uint64_t) * static_cast<size_t>(output_size_h), cudaMemcpyDeviceToHost);
    cudaMemcpy(counts_out_h, counts_out_d, sizeof(uint64_t) * static_cast<size_t>(output_size_h), cudaMemcpyDeviceToHost);

    /* 12. Print K-mer results */
    if (printAllBool) {
        printKmerCounts(keys_out_h, counts_out_h, hash_size, threshold);
    } else {
        printKmerCounts(keys_out_h, counts_out_h, std::min(hash_size, 10), threshold);
    }

    /* 13. Cleanup memory */
    cudaFree(d_readPositions);
    cudaFree(d_file);
    cudaFree(d_file_size);
    cudaFree(keys_out_d);
    cudaFree(counts_out_d);
    cudaFree(d_totalReadCount);
    cudaFreeHost(file);
    delete[] readPositions;
    cudaFreeHost(keys_out_h);
    cudaFreeHost(counts_out_h);

    return 0;
}
