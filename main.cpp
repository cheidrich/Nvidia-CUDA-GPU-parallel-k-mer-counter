/* main.cpp
 *
 * Host logic:
 *  - parse user input
 *  - allocate memory
 *  - launch CUDA kernels
 *  - retrieve and print results
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include "src/globals.cu"    // bring in externs for khost, k, arrayPosition
#include "src/kmer_utils.cu" // bring in prototypes for calcKmerFromKey/printKmerCounts

/* Declarations of kernels */
__global__ void countReadsFA(char file[], uint64_t* size, int* totalReadCount);
__global__ void findReadPositionsFA(char file[], uint64_t readPositions[], uint64_t* size);
__global__ void countKmersFA(char file[], uint64_t readPositions[], int* numberOfReads,
    warpcore::CountingHashTable<uint64_t,uint64_t> cht,
    warpcore::BloomFilter<uint64_t,warpcore::hashers::MurmurHash<uint64_t>,uint64_t> bf);

__global__ void countReadsFQ(char file[], uint64_t* size, int* totalReadCount);
__global__ void findReadPositionsFQ(char file[], uint64_t readPositions[], uint64_t* size);
__global__ void countKmersFQ(char file[], uint64_t readPositions[], int* numberOfReads,
    warpcore::CountingHashTable<uint64_t,uint64_t> cht,
    warpcore::BloomFilter<uint64_t,warpcore::hashers::MurmurHash<uint64_t>,uint64_t> bf);

/* Simple CUDA error check */
#define cudaCheck() do { cudaError_t e = cudaGetLastError(); if(e!=cudaSuccess){ \
    std::cerr<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; exit(1); }} while(0)

int main() {
    std::string path;
    std::cout<<"File path: "; std::cin>>path;
    char fmt; std::cout<<"Format (a=FASTA, q=FASTQ): "; std::cin>>fmt;
    bool isFastQ = (fmt=='q');

    int ktemp; std::cout<<"k (2–31): "; std::cin>>ktemp;
    int threshold; std::cout<<"Threshold: "; std::cin>>threshold;
    bool printAll = false; char c; std::cout<<"Print all? (y/n): "; std::cin>>c;
    if(c=='y') printAll=true;

    /* set k on host and copy to device */
    khost = ktemp;
    cudaMemcpyToSymbol(k, &ktemp, sizeof(int));

    /* read file into pinned host memory */
    std::ifstream f(path, std::ios::binary|std::ios::ate);
    if(!f){ std::cerr<<"Cannot open file\n"; return 1; }
    size_t sz=f.tellg(); f.seekg(0);
    char* h_file=nullptr; cudaMallocHost(&h_file, sz);
    f.read(h_file, sz); f.close();

    /* allocate and copy to device */
    char* d_file; uint64_t* d_size;
    cudaMalloc(&d_file, sz); cudaMalloc(&d_size, sizeof(uint64_t));
    cudaMemcpy(d_file, h_file, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_size, &sz, sizeof(uint64_t), cudaMemcpyHostToDevice);

    /* count reads */
    int h_count=0, *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int threads=1024, blocks=(sz+threads-1)/threads;
    if(isFastQ) countReadsFQ<<<blocks,threads>>>(d_file,d_size,d_count);
    else        countReadsFA<<<blocks,threads>>>(d_file,d_size,d_count);
    cudaCheck();

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    /* allocate readPositions */
    uint64_t* h_pos = new uint64_t[h_count];
    uint64_t* d_pos;
    cudaMalloc(&d_pos, h_count*sizeof(uint64_t));
    cudaMemcpy(d_pos, h_pos, h_count*sizeof(uint64_t), cudaMemcpyHostToDevice);

    /* find positions */
    if(isFastQ) findReadPositionsFQ<<<blocks,threads>>>(d_file,d_pos,d_size);
    else        findReadPositionsFA<<<blocks,threads>>>(d_file,d_pos,d_size);
    cudaDeviceSynchronize(); cudaCheck();

    /* init warpcore structures */
    uint64_t maxKmers = sz - (ktemp-1)*h_count;
    uint64_t possible=uint64_t(std::pow(4,ktemp));
    uint64_t capTab=maxKmers, capFil=possible;
    if(possible<capTab) capTab=possible;
    if(maxKmers<possible) capFil=maxKmers;
    warpcore::CountingHashTable<uint64_t,uint64_t> cht(capTab);
    uint64_t bits = ceil((long double)capFil*log(0.02L)/log(1.0L/pow(2.0L,log(2.0L))));
    warpcore::BloomFilter<uint64_t,warpcore::hashers::MurmurHash<uint64_t>,uint64_t> bf(bits,6);

    /* count kmers */
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);
    blocks = (h_count+threads-1)/threads;
    if(isFastQ) countKmersFQ<<<blocks,threads>>>(d_file,d_pos,d_count,cht,bf);
    else        countKmersFA<<<blocks,threads>>>(d_file,d_pos,d_count,cht,bf);
    cudaDeviceSynchronize(); cudaCheck();

    /* retrieve results */
    size_t hs = cht.size();
    uint64_t *d_keys, *d_vals;
    uint64_t *h_keys=new uint64_t[hs], *h_vals=new uint64_t[hs];
    cudaMalloc(&d_keys,hs*sizeof(uint64_t));
    cudaMalloc(&d_vals,hs*sizeof(uint64_t));
    cht.retrieve_all(d_keys,d_vals,hs);
    cudaMemcpy(h_keys,d_keys,hs*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals,d_vals,hs*sizeof(uint64_t),cudaMemcpyDeviceToHost);

    /* print */
    printKmerCounts(h_keys,h_vals, printAll? hs: std::min(hs, size_t(10)), threshold);

    /* cleanup */
    cudaFree(d_file); cudaFree(d_size);
    cudaFree(d_count); cudaFree(d_pos);
    cudaFree(d_keys); cudaFree(d_vals);
    cudaFreeHost(h_file);
    delete[] h_pos; delete[] h_keys; delete[] h_vals;
    return 0;
}
